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
from . import sql


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
    # The objective is part of backward(), not of the caller's program: it
    # carries a gradient even when backward() is called under no_grad().
    with torch.enable_grad():
        objective = _objective(loss, gradient, env)
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

    # Each leaf as a table, which ddx differentiates with respect to. The
    # loss's graph refers to each leaf's node from before the sync.
    tables, resolve = [], {}
    for t in targets:
        node = t._node
        env.sync(t)
        if any(t._node is other for other in tables):
            raise AssertionError("two leaves share one table; their gradients would mix")
        tables.append(t._node)
        resolve[node.id] = t._node
    # Each stop-gradient node as a table, which ddx treats as a constant.
    # Every other node is computed again, so that ddx differentiates it,
    # even if its value was read before and is saved.
    stack, seen = [objective._node], set()
    while stack:
        node = stack.pop()
        if node.id in seen or node.id in resolve:
            continue
        seen.add(node.id)
        if isinstance(node, g.Select) and node.stop_gradient:
            resolve[node.id] = env.materialize(node)
        else:
            stack.extend(node.inputs)

    def read(node):
        return resolve.get(node.id, node)

    reached = {n.id for n in g.topological([objective._node], read) if isinstance(n, g.Table)}
    wrt = [table for table in tables if table.id in reached]
    grads = {}
    if wrt:
        # SUM over the one row: ddx can then show the loss is one row, which
        # it can't for a scalar cross joined with another (a detached scalar).
        summed = g.Select(
            (), objective.dtype, (objective._node,), value=g.Agg("SUM", g.Col(0, "v")), group=True
        )
        loss_sql = sql.query(summed, resolve)
        out = env.engine.grad(loss_sql, [f"{t.name}.v" for t in wrt])
        # The program computed the loss too: reading it costs no query of
        # the forward pass.
        value = _adopt(env, g.Table((), objective.dtype, name=out.value))
        env._saved[objective._node.id] = value
        objective._table = value
        if loss._node is objective._node:
            loss._table = value
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


def _objective(loss, gradient, env):
    """The scalar backward() differentiates."""
    from .device import Tensor

    if gradient is not None:
        # The vector-Jacobian product along `gradient` is the gradient of
        # sum(loss * gradient).
        if not isinstance(gradient, Tensor):
            gradient = env.tensor(gradient.detach().to(loss.dtype))
        return (loss * gradient).sum()
    if loss.numel() != 1:
        raise RuntimeError("grad can be implicitly created only for scalar outputs")
    return loss.reshape(())


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
