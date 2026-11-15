"""Simulation tests: a few training steps with a random optimizer, on the CPU
and on ``device="sql"``, step by step (design §11.5).

The optimizer's arithmetic is ordinary tensor ops that torch-sql lowers, so
any optimizer configuration ``torch.optim`` accepts should give PyTorch's
parameters after every step, on every engine and every cut budget.
"""

from __future__ import annotations

import copy

import pytest
import torch
from harness import make_engine, refusals_are_skips
from hypothesis import given, note
from hypothesis import strategies as st

import torch_sql
from torch_sql import engines

F64 = torch.float64


@st.composite
def optimizers(draw):
    kind = draw(st.sampled_from(["SGD", "Adam", "AdamW"]))
    foreach = draw(st.sampled_from([None, False, True]))
    wd = draw(st.sampled_from([0.0, 0.01]))
    if kind == "SGD":
        momentum = draw(st.sampled_from([0.0, 0.9]))
        nesterov = momentum > 0 and draw(st.booleans())
        kwargs = dict(lr=0.1, momentum=momentum, nesterov=nesterov, weight_decay=wd)
    else:
        kwargs = dict(lr=0.01, weight_decay=wd, amsgrad=draw(st.booleans()))
    return kind, {**kwargs, "foreach": foreach}


@pytest.mark.parametrize("engine", sorted(engines.ENGINES))
@given(
    optimizer=optimizers(),
    steps=st.integers(1, 3),
    clip=st.sampled_from([None, 0.5]),
    budget=st.sampled_from(["default", 2]),
    seed=st.integers(0, 2**16),
    width=st.sampled_from([1, 2]),
)
def test_training_matches_pytorch(engine, optimizer, steps, clip, budget, seed, width):
    kind, kwargs = optimizer
    note(f"{kind}({kwargs}), clip={clip}")
    if engine == "duckdb" and budget != "default":
        budget = None

    def train(device):
        torch.manual_seed(seed)
        model = torch.nn.Sequential(
            # width 1: a one-element bias, as a regression head has (#27).
            torch.nn.Linear(3, 4),
            torch.nn.Tanh(),
            torch.nn.Linear(4, width),
        ).double()
        x, y = torch.randn(5, 3, dtype=F64), torch.randn(5, width, dtype=F64)
        if device == "sql":
            model, x, y = model.to("sql"), x.to("sql"), y.to("sql")
        opt = getattr(torch.optim, kind)(model.parameters(), **kwargs)
        history = []
        for _ in range(steps):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model(x), y)
            loss.backward()
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            history.append([p.detach().cpu().clone() for p in model.parameters()])
        return history

    expected = train("cpu")
    with refusals_are_skips(), torch_sql.Environment(make_engine(engine, budget)):
        got = train("sql")
    for step, (a, b) in enumerate(zip(got, expected, strict=True)):
        for k, (p, q) in enumerate(zip(a, b, strict=True)):
            torch.testing.assert_close(
                p, q, rtol=1e-10, atol=1e-12, msg=lambda m, s=step, k=k: f"step {s} param {k}: {m}"
            )


@st.composite
def more_optimizers(draw):
    """:func:`optimizers`, and the others ``torch.optim`` has whose step is
    ordinary tensor ops."""
    kind = draw(
        st.sampled_from(["SGD, Adam or AdamW", "RMSprop", "Adagrad", "Adamax", "NAdam", "RAdam"])
    )
    if kind == "SGD, Adam or AdamW":
        return draw(optimizers())
    kwargs = dict(lr=0.01, weight_decay=draw(st.sampled_from([0.0, 0.01])))
    return kind, {**kwargs, "foreach": draw(st.sampled_from([None, False, True]))}


#: Losses, by name: each takes the model's output and a target of the shape
#: :func:`_targets` makes for it.
LOSSES = {
    "mse": torch.nn.functional.mse_loss,
    "cross_entropy": torch.nn.functional.cross_entropy,
    "bce_with_logits": torch.nn.functional.binary_cross_entropy_with_logits,
}


def _targets(loss: str, n: int):
    if loss == "cross_entropy":
        return 3, torch.randint(0, 3, (n,))
    if loss == "bce_with_logits":
        return 1, torch.randint(0, 2, (n, 1)).float()
    return 2, torch.randn(n, 2)


@pytest.mark.parametrize("engine", sorted(engines.ENGINES))
@given(
    optimizer=more_optimizers(),
    steps=st.integers(1, 3),
    clip=st.sampled_from([None, 0.5]),
    loss=st.sampled_from(sorted(LOSSES)),
    seed=st.integers(0, 2**16),
)
def test_float32_training_matches_pytorch(engine, optimizer, steps, clip, loss, seed):
    """Training in float32, PyTorch's default dtype, as a new user would: a
    model with ``LayerNorm`` and GELU, a classification or regression loss,
    and any optimizer. Each float32 op rounds as PyTorch's does, so the
    parameters agree to float32's precision after each step."""
    kind, kwargs = optimizer
    note(f"{kind}({kwargs}), clip={clip}, loss={loss}")

    def train(device):
        torch.manual_seed(seed)
        x = torch.randn(6, 3)
        width, y = _targets(loss, 6)
        model = torch.nn.Sequential(
            torch.nn.Linear(3, 4),
            torch.nn.LayerNorm(4),
            torch.nn.GELU(approximate="tanh"),  # exact GELU needs erf (#28)
            torch.nn.Linear(4, width),
        )
        if device == "sql":
            model, x, y = model.to("sql"), x.to("sql"), y.to("sql")
        opt = getattr(torch.optim, kind)(model.parameters(), **kwargs)
        history = []
        for _ in range(steps):
            opt.zero_grad()
            value = LOSSES[loss](model(x), y)
            value.backward()
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            history.append(
                [value.detach().cpu()] + [p.detach().cpu().clone() for p in model.parameters()]
            )
        return history

    expected = train("cpu")
    with refusals_are_skips(), torch_sql.Environment(make_engine(engine)):
        got = train("sql")
    for step, (a, b) in enumerate(zip(got, expected, strict=True)):
        for k, (p, q) in enumerate(zip(a, b, strict=True)):
            what = "the loss" if k == 0 else f"param {k - 1}"
            assert p.dtype == q.dtype == torch.float32, f"step {step} {what}: {p.dtype}"
            torch.testing.assert_close(
                p, q, rtol=1e-4, atol=1e-5, msg=lambda m, s=step, w=what: f"step {s} {w}: {m}"
            )


@pytest.mark.parametrize("engine", sorted(engines.ENGINES))
def test_adam_amsgrad_foreach(engine):
    """``Adam(amsgrad=True, foreach=True)`` calls ``_foreach_maximum_``. The
    one rule for ``_foreach_*`` ops (spike S5) used to look up the in-place
    plain op, ``aten.maximum_``, which doesn't exist, and raised
    AttributeError."""
    with torch_sql.Environment(make_engine(engine)):
        torch.manual_seed(0)
        model = torch.nn.Linear(2, 1).double().to("sql")
        opt = torch.optim.Adam(model.parameters(), amsgrad=True, foreach=True)
        model(torch.randn(3, 2, dtype=F64).to("sql")).sum().backward()
        opt.step()


@pytest.mark.pinned
@pytest.mark.parametrize("engine", sorted(engines.ENGINES))
def test_clipping_a_one_element_gradient(engine):
    """``clip_grad_norm_`` takes the norm of each gradient. PyTorch's
    reference ``vector_norm`` of a one-element tensor calls
    ``aten.flatten.using_ints``, which had no lowering (#27), so a model
    with a one-element parameter (``Linear(n, 1)``'s bias) couldn't clip.
    The property above treats that refusal as a skip, so this pins it."""
    torch.manual_seed(0)
    cpu = torch.nn.Linear(2, 1).double()
    x = torch.randn(3, 2, dtype=F64)
    with torch_sql.Environment(make_engine(engine)):
        dev = copy.deepcopy(cpu).to("sql")
        dev(x.to("sql")).sum().backward()
        norm = torch.nn.utils.clip_grad_norm_(dev.parameters(), 0.5)
        got = [p.grad.cpu() for p in dev.parameters()]
    cpu(x).sum().backward()
    want_norm = torch.nn.utils.clip_grad_norm_(cpu.parameters(), 0.5)
    torch.testing.assert_close(norm.cpu(), want_norm)
    for g, p in zip(got, cpu.parameters(), strict=True):
        torch.testing.assert_close(g, p.grad)
