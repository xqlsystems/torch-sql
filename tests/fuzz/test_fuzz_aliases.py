"""Random programs of views, ``.detach()`` aliases and in-place ops, on the
CPU and on ``device="sql"`` (design §8.6).

In PyTorch a view and a ``.detach()`` alias share their base's storage, so
an in-place op on any of them shows through the others, and the gradient
still follows each tensor's own history. A program here is a list of
actions over a growing list of tensors: take an alias, compute a new tensor,
or write one in place (with or without ``torch.no_grad()``). At the end,
every tensor's value must match PyTorch's, and so must the gradient of a
loss over all of them. A write through a view may be refused (P3).
"""

from __future__ import annotations

import os

import pytest
import torch
from harness import close, make_engine, refusals_are_skips, scale, weights
from hypothesis import assume, event, given, note
from hypothesis import strategies as st

import torch_sql
from torch_sql import engines

F64 = torch.float64
SHAPE = (2, 3)

#: Aliases of a tensor of shape SHAPE, (name, function, the result's shape).
ALIASES = {
    "detach": (lambda t: t.detach(), SHAPE),
    "data": (lambda t: t.data, SHAPE),
    "slice": (lambda t: t[:, 1:], (2, 2)),
    "select": (lambda t: t[1], (3,)),
    "t": (lambda t: t.t(), (3, 2)),
    "view": (lambda t: t.view(6), (6,)),
    "unsqueeze": (lambda t: t.unsqueeze(0), (1, 2, 3)),
}

#: In-place ops, which keep the target's shape.
WRITES = {
    "add_": lambda t: t.add_(0.5),
    "mul_": lambda t: t.mul_(-1.5),
    "zero_": lambda t: t.zero_(),
    "clamp_": lambda t: t.clamp_(-0.5, 0.75),
    "copy_": lambda t: t.copy_(t * 2 + 1),
    "lerp_": lambda t: t.lerp_(t * 0 + 1, 0.25),
}

#: New tensors from one tensor.
COMPUTE = {
    "mul": lambda t: t * 1.25,
    "tanh": lambda t: torch.tanh(t),
    "clone": lambda t: t.clone(),
}


@st.composite
def alias_programs(draw):
    """Actions over tensors 0 and 1 (a leaf, and a tensor computed from a
    second leaf) and the tensors the actions make. Each tensor has shape
    SHAPE unless it is an alias of another shape."""
    shapes = [SHAPE, SHAPE]
    actions = []
    for _ in range(draw(st.integers(1, 8))):
        k = draw(st.integers(0, len(shapes) - 1))
        kind = draw(st.sampled_from(["alias", "compute", "write", "write"]))
        if kind == "alias" and shapes[k] == SHAPE:
            name = draw(st.sampled_from(sorted(ALIASES)))
            actions.append(("alias", name, k))
            shapes.append(ALIASES[name][1])
        elif kind == "compute" and shapes[k] == SHAPE:
            actions.append(("compute", draw(st.sampled_from(sorted(COMPUTE))), k))
            shapes.append(SHAPE)
        elif kind == "write":
            name = draw(st.sampled_from(sorted(WRITES)))
            actions.append(("write", name, k, draw(st.booleans())))
    data = draw(
        st.lists(
            st.floats(-2, 2, allow_nan=False).map(lambda v: round(v, 3)), min_size=12, max_size=12
        )
    )
    return actions, data


def interpret(program, device):
    """Run the program, returning every tensor and the two leaves."""
    actions, data = program
    # Not reshape(), whose result is a view: an in-place op on a view of a
    # tensor that doesn't require a gradient changes how PyTorch regenerates
    # the gradient of the view's own views.
    leaves = [
        torch.tensor([data[0:3], data[3:6]], dtype=F64),
        torch.tensor([data[6:9], data[9:12]], dtype=F64),
    ]
    if device != "cpu":
        leaves = [x.to(device) for x in leaves]
    leaves = [x.requires_grad_() for x in leaves]
    tensors = [leaves[0], leaves[1] * 1.0]
    for action in actions:
        if action[0] == "alias":
            tensors.append(ALIASES[action[1]][0](tensors[action[2]]))
        elif action[0] == "compute":
            tensors.append(COMPUTE[action[1]](tensors[action[2]]))
        else:
            _, name, k, no_grad = action
            target = tensors[k]
            if not no_grad and target.requires_grad and target.grad_fn is None:
                # PyTorch refuses an in-place op on a leaf that requires a
                # gradient, outside no_grad.
                no_grad = True
            if no_grad:
                with torch.no_grad():
                    WRITES[name](target)
            else:
                WRITES[name](target)
    return tensors, leaves


def known(program) -> str | None:
    """A finding already reported, which a soak should look past. Set
    ``TORCH_SQL_FUZZ_KNOWN=0`` to see them again."""
    if os.environ.get("TORCH_SQL_FUZZ_KNOWN", "1") == "0":
        return None
    actions, _ = program
    root = [0, 1]  # the tensor whose storage each tensor shares
    depth = [0, 0]  # how many aliases each tensor is from its root
    tracked = [True, True]  # whether autograd sees a write to it
    history = [False, True]  # whether its root has a grad_fn
    used = set()  # roots that another tensor was computed from
    for action in actions:
        k = action[2]
        if action[0] == "alias":
            root.append(root[k])
            depth.append(depth[k] + 1)
            tracked.append(tracked[k] and action[1] not in ("detach", "data"))
            history.append(history[k])
        elif action[0] == "compute":
            used.add(root[k])
            root.append(len(root))
            depth.append(0)
            tracked.append(True)
            history.append(True)
        else:
            if root[k] == 0 and 0 in used:
                return "a leaf written after it was used loses those uses' gradients"
            if history[k] and (action[3] or not tracked[k]):
                return "a write autograd doesn't record erases the tensor's history"
            if any(d >= 2 for r, d in zip(root, depth, strict=True) if r == root[k]):
                return "a view of an alias derives its base's node again"
    return None


def loss_of(tensors):
    return sum(
        (t * weights(t.shape).to(t.device if t.device.type == "cpu" else "sql")).sum()
        for t in tensors
        if t.requires_grad
    )


@pytest.mark.parametrize("engine", sorted(engines.ENGINES))
@given(program=alias_programs())
def test_aliases_match_pytorch(engine, program):
    note(repr(program[0]))
    why = known(program)
    if why:
        event(f"known finding: {why}")
        assume(False)
    try:
        tensors, leaves = interpret(program, "cpu")
        values = [t.detach().clone() for t in tensors]
        loss = loss_of(tensors)
        grads = None
        if isinstance(loss, torch.Tensor):
            grads = torch.autograd.grad(loss, leaves, allow_unused=True)
    except RuntimeError as e:
        # PyTorch refuses some programs: a write to a tensor that backward
        # needs, a write through a view of a leaf.
        event(f"PyTorch refuses: {str(e)[:50]}")
        assume(False)
    with refusals_are_skips(), torch_sql.Environment(make_engine(engine)):
        tensors_sql, leaves_sql = interpret(program, "sql")
        got = [t.detach().cpu() for t in tensors_sql]
        assert [t.requires_grad for t in tensors_sql] == [t.requires_grad for t in tensors]
        loss_sql = loss_of(tensors_sql)
        if grads is not None:
            loss_sql.backward()
            got_grads = [x.grad.cpu() if x.grad is not None else None for x in leaves_sql]
    s = scale(values)
    for k, (a, b) in enumerate(zip(got, values, strict=True)):
        close(a, b, s, f"tensor {k}")
    if grads is None:
        event("values only")
        return
    event("gradient compared")
    for k, (a, b) in enumerate(zip(got_grads, grads, strict=True)):
        if b is None:
            assert a is None or not a.abs().any(), f"leaf {k}: a gradient where PyTorch has none"
        else:
            assert a is not None, f"leaf {k}: no gradient where PyTorch has {b.tolist()}"
            close(a, b, s, f"gradient of leaf {k}")


# -- Pinned findings ---------------------------------------------------------
# Each was found by the property above and shrunk by hand. Each is a strict
# xfail: it passes once the bug is fixed, then fails until the marker goes.


def _gradient(engine, program):
    """``program(device)`` returns a leaf and a loss; the leaf's gradient on
    the CPU and on ``engine``."""
    leaf, loss = program("cpu")
    loss.backward()
    expected = leaf.grad
    with torch_sql.Environment(make_engine(engine)):
        leaf, loss = program("sql")
        loss.backward()
        got = leaf.grad.cpu() if leaf.grad is not None else torch.zeros_like(expected)
    return got, expected


def _leaf(device):
    x = torch.tensor([1.0, 2.0, 3.0], dtype=F64)
    return (x.to("sql") if device == "sql" else x).requires_grad_()


@pytest.mark.pinned
@pytest.mark.xfail(strict=True, reason="a write autograd doesn't record erases the history")
@pytest.mark.parametrize("engine", sorted(engines.ENGINES))
@pytest.mark.parametrize("how", ["no_grad", "detach"])
def test_untracked_write_keeps_history(engine, how):
    """An in-place op autograd doesn't record, under ``torch.no_grad()`` or
    through ``.detach()``, changes a tensor's value but not its gradient: in
    PyTorch, ``y`` below still has ``MulBackward``, and ``w.grad`` is 2.
    torch-sql gives ``y`` the write's node, which is cut from ``y``'s history
    (a stop-gradient), so ``w.grad`` is 0."""

    def program(device):
        w = _leaf(device)
        y = w * 2
        if how == "no_grad":
            with torch.no_grad():
                y.add_(1)
        else:
            y.detach().add_(1)
        return w, y.sum()

    got, expected = _gradient(engine, program)
    assert torch.equal(got, expected), f"got {got.tolist()}, PyTorch {expected.tolist()}"


@pytest.mark.pinned
@pytest.mark.xfail(strict=True, reason="a leaf written after use loses those uses' gradients")
@pytest.mark.parametrize("engine", sorted(engines.ENGINES))
def test_leaf_written_after_use(engine):
    """``c = w * 2`` reads the leaf's node before the write; backward()
    differentiates with respect to the leaf's node after it, so the gradient
    through ``c`` is lost: 1 where PyTorch gives 3. Had ``c`` saved ``w``
    (``c = w * w``), PyTorch would raise "one of the variables needed for
    gradient computation has been modified by an inplace operation";
    torch-sql gives 1 there too, with no error."""

    def program(device):
        w = _leaf(device)
        c = w * 2
        with torch.no_grad():
            w.add_(0.5)
        return w, w.sum() + c.sum()

    got, expected = _gradient(engine, program)
    assert torch.equal(got, expected), f"got {got.tolist()}, PyTorch {expected.tolist()}"


@pytest.mark.pinned
@pytest.mark.xfail(strict=True, reason="a view of an alias derives its base's node again")
@pytest.mark.parametrize("engine", sorted(engines.ENGINES))
def test_view_of_a_parameter_after_a_write(engine):
    """An ``nn.Parameter`` on the device is an alias of the tensor it wraps.
    A view of it (``w[:2]``) re-derives its node through the parameter's own
    derivation, making a second node for the parameter's value, which
    backward() doesn't treat as the leaf: after an in-place update (an
    optimizer step), the view's gradient is 0."""

    def program(device):
        w = _leaf(device)
        w = torch.nn.Parameter(w.detach())
        v = w[:2]
        with torch.no_grad():
            w.mul_(2)
        return w, (v * v).sum()

    got, expected = _gradient(engine, program)
    assert torch.equal(got, expected), f"got {got.tolist()}, PyTorch {expected.tolist()}"
