"""Backward passes through ddx (design §11.1).

``loss.backward()`` on a torch-sql tensor doesn't run PyTorch's autograd.
It finds the leaf tensors the loss depends on, asks ddx for the gradient of
the loss with respect to all their tables at once (one program, one backward
pass), and sets each leaf's ``.grad`` to a tensor holding its gradient's
table. ``torch.optim`` reads only ``.grad``, so it works unchanged.

**Which leaves.** PyTorch's autograd still records its graph during the
forward pass, though nothing runs it. Its ``AccumulateGrad`` nodes are the
leaves that require a gradient and that the loss depends on, after
``torch.no_grad()``, ``.detach()`` and ``requires_grad=False`` have had
their say, so torch-sql reads the leaves from there.

**Stopping gradients** (§11.3). A node marked ``stop_gradient``
(``.detach()``, or a value computed under ``torch.no_grad()`` from one that
requires a gradient) is saved as a table first. A table that isn't a
``wrt`` table is a constant to ddx, so no gradient flows through it. ddx has
a function for this, ``ddx_stop_gradient``, but ddxdb registers it only on
DataFusion, and only for DOUBLE (xqlsystems/ddx#260); saving a table works
on every engine.
"""

from __future__ import annotations

import weakref

import torch

from . import graph as g


def leaves(loss: torch.Tensor) -> list[torch.Tensor]:
    """The tensors that would accumulate a gradient if PyTorch's autograd
    ran ``loss.backward()``, in a stable order."""
    if loss.grad_fn is None:
        return [loss] if loss.requires_grad else []
    found, seen, stack = [], set(), [loss.grad_fn]
    while stack:
        fn = stack.pop()
        if fn is None or id(fn) in seen:
            continue
        seen.add(id(fn))
        variable = getattr(fn, "variable", None)
        if variable is not None:  # an AccumulateGrad node
            found.append(variable)
        stack.extend(next_fn for next_fn, _ in fn.next_functions)
    return found


def backward(loss, gradient=None, retain_graph=None, create_graph=False, inputs=None):
    """``loss.backward()``, for a torch-sql tensor."""
    from .device import Tensor

    if create_graph:
        raise NotImplementedError("backward(create_graph=True): higher-order gradients")
    if not loss.requires_grad:
        raise RuntimeError("element 0 of tensors does not require grad and does not have a grad_fn")
    env = loss._env
    if gradient is not None:
        # The vector-Jacobian product along `gradient` is the gradient of
        # sum(loss * gradient).
        if not isinstance(gradient, Tensor):
            gradient = env.tensor(gradient.detach().to(loss.dtype))
        objective = (loss * gradient).sum()
    elif loss.numel() != 1:
        raise RuntimeError("grad can be implicitly created only for scalar outputs")
    else:
        objective = loss.reshape(())

    targets = leaves(loss)
    if inputs is not None:
        inputs = [inputs] if isinstance(inputs, torch.Tensor) else list(inputs)
        targets = [t for t in targets if any(t is i for i in inputs)]
    for t in targets:
        if not isinstance(t, Tensor):
            raise RuntimeError("a torch-sql loss depends on a leaf that isn't a torch-sql tensor")
        if t._env is not env:
            raise RuntimeError("a loss depends on tensors from two torch-sql environments")
    if not targets:
        return

    # Each leaf as a table, which ddx differentiates with respect to.
    tables = []
    for t in targets:
        env.sync(t)
        tables.append(t._node)
    # Each stop-gradient node as a table, which ddx treats as a constant.
    # The environment keeps saved tables only while something refers to
    # them, so `held` keeps these until the gradients exist.
    held = [
        env.materialize(node)
        for node in g.topological([objective._node], env.saved)
        if isinstance(node, g.Select) and node.stop_gradient
    ]

    root = env.saved(objective._node)
    reached = {n.id for n in g.topological([root], env.saved) if isinstance(n, g.Table)}
    wrt = [table for table in tables if table.id in reached]
    grads = {}
    if wrt:
        out = env.engine.grad(env.query(objective._node), [f"{t.name}.v" for t in wrt])
        # The program computed the loss too: reading it costs no query of
        # the forward pass.
        value = _adopt(env, g.Table((), objective.dtype, name=out.value))
        env._saved[objective._node.id] = value
        objective._node = value
        if loss._node is not value and env.saved(loss._node) is value:
            loss._node = value
        for table in wrt:
            name = out.gradients[table.name]
            grads[table.id] = _adopt(env, g.Table(table.shape, table.dtype, name=name))
    for t, table in zip(targets, tables, strict=True):
        node = grads.get(table.id)
        if node is None:  # a leaf the SQL doesn't read: its gradient is 0
            grad = torch.zeros_like(t)
        else:
            grad = Tensor(node, env)
        with torch.no_grad():
            t.grad = grad if t.grad is None else t.grad + grad
    del held


def _adopt(env, table: g.Table) -> g.Table:
    """Take ownership of a table ddx left on the engine: it's dropped when
    the node is."""
    weakref.finalize(table, _drop, env.engine, table.name)
    return table


def _drop(engine, name):
    try:
        engine.drop(name)
    except Exception:  # pragma: no cover - the engine may be gone at exit
        pass
