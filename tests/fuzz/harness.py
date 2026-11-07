"""Running a random program on the CPU and on the device, and comparing.

Shared by the property tests in this directory.
"""

from __future__ import annotations

import math
import os
from contextlib import contextmanager

import torch
from fuzz_programs import F64, Program, Screens, leaf_tensors, output_of, run, scale
from hypothesis import assume, event, note

import torch_sql
from torch_sql import OperatorNotFound, engines

#: Rounding: a result may differ from PyTorch's by this much relative to the
#: largest magnitude the program computes (ddx's fuzzer measures the same
#: way). Summation order differs between engines, so an exact match isn't
#: expected (design §14).
RTOL = 1e-9


def weights(shape) -> torch.Tensor:
    """Fixed, irregular weights for a scalar loss ``sum(out * w)``, so every
    element of the output gets its own cotangent."""
    n = math.prod(shape)
    return torch.cos(torch.arange(n, dtype=F64) * 0.7 + 0.3).reshape(shape)


def close(got: torch.Tensor, expected: torch.Tensor, scale_: float, what: str):
    got = got.detach().to(F64)
    expected = expected.detach().to(F64)
    assert got.shape == expected.shape, (
        f"{what}: shape {tuple(got.shape)} != {tuple(expected.shape)}"
    )
    nan_got, nan_exp = torch.isnan(got), torch.isnan(expected)
    assert torch.equal(nan_got, nan_exp), (
        f"{what}: NaN where PyTorch has none, or none where it has one\n"
        f"got      {got.tolist()}\nexpected {expected.tolist()}"
    )
    inf_exp = torch.isinf(expected)
    assert torch.equal(got[inf_exp], expected[inf_exp]), (
        f"{what}: infinities differ\ngot      {got.tolist()}\nexpected {expected.tolist()}"
    )
    finite = ~(nan_exp | inf_exp)
    err = (got[finite] - expected[finite]).abs()
    tol = RTOL * scale_
    assert not err.numel() or float(err.max()) <= tol, (
        f"{what}: off by {float(err.max()):.3g} (tolerance {tol:.3g})\n"
        f"got      {got.tolist()}\nexpected {expected.tolist()}"
    )


def make_engine(name: str, budget="default") -> engines.Engine:
    e = engines.engine(name)
    if budget != "default":
        e.cut_budget = budget
    return e


#: Findings already reported, which a soak should look past to find new
#: ones: (engine, the start of the error's message, an op the program must
#: use). Each has a pinned test in test_fuzz_device.py; remove its entry here
#: when it's fixed. Set ``TORCH_SQL_FUZZ_KNOWN=0`` to see them all again.
KNOWN = [
    ("duckdb", "Out of Range Error: input value", {"sin", "cos"}),
]


#: Known wrong values: (engine, ops, why). A program that uses one of the
#: ops and computes a NaN anywhere is skipped on that engine.
KNOWN_NAN = [
    ("datafusion", {"relu", "maximum"}, "DataFusion orders a negative NaN below every number"),
]


def _known_nan(program: Program, engine: str, values) -> str | None:
    if os.environ.get("TORCH_SQL_FUZZ_KNOWN", "1") == "0":
        return None
    ops = {s.op for s in program.steps}
    if not any(bool(torch.isnan(v.detach()).any()) for v in values):
        return None
    return next((why for e, uses, why in KNOWN_NAN if e == engine and ops & uses), None)


def _known(program: Program, engine: str, error: Exception) -> bool:
    if os.environ.get("TORCH_SQL_FUZZ_KNOWN", "1") == "0":
        return False
    ops = {s.op for s in program.steps}
    return any(engine == e and str(error).startswith(msg) and ops & uses for e, msg, uses in KNOWN)


@contextmanager
def refusals_are_skips(program: Program | None = None, engine: str | None = None):
    """A refusal is allowed (P3); the test moves on to another program. So
    is a known finding (:data:`KNOWN`)."""
    try:
        yield
    except (NotImplementedError, OperatorNotFound) as e:
        note(f"refused: {e}")
        event(f"refused: {str(e)[:60]}")
        assume(False)
    except Exception as e:
        if program is None or not _known(program, engine, e):
            raise
        event(f"known finding: {str(e)[:60]}")
        assume(False)


def oracle(program: Program, *, grad: bool):
    """The CPU's values, the output's gradient with respect to each leaf, and
    the screens."""
    screens = Screens()
    leaves = leaf_tensors(program, "cpu", requires_grad=grad)
    values = run(program, leaves, screens)
    grads = None
    if grad and values[output_of(program)].requires_grad:
        out = values[output_of(program)]
        loss = (out * weights(out.shape)).sum()
        grads = torch.autograd.grad(loss, leaves, allow_unused=True)
        grads = [
            torch.zeros_like(x) if g is None else g for x, g in zip(leaves, grads, strict=True)
        ]
    return values, grads, screens


def on_device(program: Program, engine: engines.Engine, *, grad: bool, read=(), requires=None):
    """The output's value and, with ``grad``, each leaf's gradient, on
    ``engine``. ``read`` lists values to read before the output, to check
    that reading a value changes nothing."""
    with torch_sql.Environment(engine):
        leaves = leaf_tensors(program, "sql", requires_grad=grad or bool(requires))
        values = run(program, leaves)
        for k in read:
            values[k].detach().cpu()
        out = values[output_of(program)]
        if requires is not None:
            assert out.requires_grad == requires, (
                f"the output's requires_grad is {out.requires_grad}; PyTorch's is {requires}"
            )
        grads = None
        if grad and out.requires_grad:
            loss = (out * weights(out.shape).to("sql")).sum()
            loss.backward()
            grads = [
                torch.zeros(x.shape, dtype=F64) if x.grad is None else x.grad.cpu() for x in leaves
            ]
        value = out.detach().cpu()
    return value, grads


def check(program: Program, engine_name: str, *, grad: bool, budget="default", read=()):
    note(repr(program))
    values, expected_grads, screens = oracle(program, grad=grad)
    out = values[output_of(program)]
    s = scale(values)
    if screens.overflow:
        event("skipped: a sum near overflow")
        assume(False)
    known = _known_nan(program, engine_name, values)
    if known:
        event(f"known finding: {known}")
        assume(False)
    requires = out.requires_grad if grad else None
    grad = (
        grad
        and out.requires_grad
        and not screens.kink
        and not screens.domain
        and bool(torch.isfinite(out).all())
    )
    with refusals_are_skips(program, engine_name):
        got, grads = on_device(
            program, make_engine(engine_name, budget), grad=grad, read=read, requires=requires
        )
    close(got, out, s, "value")
    event("gradient compared" if grad else "value only")
    if grad:
        gs = scale([*expected_grads, *values])
        for k, (a, b) in enumerate(zip(grads, expected_grads, strict=True)):
            close(a, b, gs, f"gradient of leaf {k}")
    return grad
