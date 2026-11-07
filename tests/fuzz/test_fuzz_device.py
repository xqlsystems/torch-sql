"""Property-based tests: random programs on ``device="sql"`` against PyTorch.

Differential tests compare each engine with PyTorch on the CPU. Metamorphic
tests change something that must not change the answer (the cut budget,
reading a value early, the order of a table's rows) and compare torch-sql
with itself.
"""

from __future__ import annotations

import math
import random

import pyarrow as pa
import pytest
import torch
from fuzz_programs import Program, output_of, programs, scale
from harness import check, close, make_engine, on_device, oracle, refusals_are_skips
from hypothesis import assume, given, note
from hypothesis import strategies as st

from torch_sql import engines
from torch_sql.device import environment

ENGINES = sorted(engines.ENGINES)


# -- Differential: the device against PyTorch -------------------------------


@pytest.mark.parametrize("engine", ENGINES)
@given(programs())
def test_values_match_pytorch(engine, program):
    """Every mode, NaN and infinities included: the same numbers as PyTorch
    on the CPU (design §1.3 item 4)."""
    check(program, engine, grad=False)


@pytest.mark.parametrize("engine", ENGINES)
@given(programs(modes=("smooth",)))
def test_gradients_match_autograd(engine, program):
    """Away from ties, kinks and domain edges, ddx's gradients are autograd's."""
    check(program, engine, grad=True)


# -- Metamorphic: torch-sql against itself ----------------------------------


@given(programs(modes=("smooth", "ties")), st.sampled_from([1, 2, 3, 5, 8, 40]))
def test_cut_budget_changes_nothing(program, budget):
    """Cut points are a plan-size decision (§9.6): saving more values as
    tables must not change a value or a gradient."""
    note(repr(program))
    _, _, screens = oracle(program, grad=False)
    grad = program.mode == "smooth" and oracle(program, grad=True)[1] is not None
    with refusals_are_skips():
        base, base_grads = on_device(program, make_engine("duckdb", None), grad=grad)
        cut, cut_grads = on_device(program, make_engine("datafusion", budget), grad=grad)
    s = scale([base])
    close(cut, base, s, f"value with budget {budget}")
    if grad and not screens.kink:
        for k, (a, b) in enumerate(zip(cut_grads, base_grads, strict=True)):
            close(a, b, max(s, scale([b])), f"gradient {k}")


@pytest.mark.parametrize("engine", ENGINES)
@given(st.data())
def test_reading_a_value_early_changes_nothing(engine, data):
    """Reading an intermediate value saves it as a table (§9.6). A later
    read, or a backward pass that differentiates through it, must give the
    same answer as if it had never been read."""
    program: Program = data.draw(programs(modes=("smooth",)))
    n = output_of(program)
    read = data.draw(st.lists(st.integers(len(program.leaves), n), max_size=3, unique=True))
    check(program, engine, grad=True, read=read)


@pytest.mark.parametrize("engine", ENGINES)
@given(program=programs(modes=("smooth", "ties")), rng=st.randoms(use_true_random=False))
def test_row_order_changes_nothing(monkeypatch_rows, engine, program, rng: random.Random):
    """An engine may return and store rows in any order, in any number of
    batches. Registering every table with its rows shuffled, in several
    batches, must not change a value or a gradient."""
    monkeypatch_rows(rng)
    check(program, engine, grad=program.mode == "smooth")


@pytest.fixture
def monkeypatch_rows(monkeypatch):
    original = environment.to_table

    def install(rng: random.Random):
        def shuffled(t):
            table = original(t)
            n = table.num_rows
            order = list(range(n))
            rng.shuffle(order)
            table = table.take(pa.array(order, pa.int64()))
            cuts = sorted(rng.sample(range(1, n), min(3, n - 1))) if n > 1 else []
            bounds = [0, *cuts, n]
            batches = [
                b
                for lo, hi in zip(bounds, bounds[1:], strict=False)
                for b in table.slice(lo, hi - lo).to_batches()
            ]
            return pa.Table.from_batches(batches, schema=table.schema)

        monkeypatch.setattr(environment, "to_table", shuffled)

    return install


# -- Pinned findings ---------------------------------------------------------
# Each was found by the tests above and shrunk by hand.


@pytest.mark.pinned
@pytest.mark.parametrize(
    "engine",
    [
        pytest.param(
            "datafusion",
            marks=pytest.mark.xfail(
                strict=True, reason="DataFusion orders a negative NaN below every number"
            ),
        ),
        "duckdb",
    ],
)
@pytest.mark.parametrize("fn", ["relu", "maximum"])
def test_negative_nan_stays_nan(engine, fn):
    """0/0 on x86 is a NaN with its sign bit set. SQL engines order NaN above
    every number, but DataFusion orders by IEEE total order, which puts a
    negative NaN *below* every number; a CASE written for the first
    ordering then picks the wrong branch."""
    import torch_sql

    with torch_sql.Environment(make_engine(engine)):
        z = torch.zeros(1, dtype=torch.float64, device="sql")
        nan = z / z
        out = torch.relu(nan) if fn == "relu" else torch.maximum(nan, z + 1)
        assert torch.isnan(out.cpu()).all(), f"{fn}(0/0) = {out.cpu().tolist()}"


@pytest.mark.pinned
@pytest.mark.xfail(strict=True, reason="an in-place op doesn't reach the tensor's aliases")
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("alias", ["slice", "detach", "view"])
def test_writing_a_tensor_updates_its_aliases(engine, alias):
    """In PyTorch a view, ``.detach()`` and ``.view()`` share their base's
    storage, so an in-place op on the base shows through them. torch-sql
    replaces the base's node and leaves the alias reading the old one."""
    import torch_sql

    def program(device):
        x = torch.zeros(4, dtype=torch.float64, device=device)
        y = {"slice": lambda: x[:2], "detach": x.detach, "view": lambda: x.view(2, 2)}[alias]()
        x.add_(1)
        return y

    expected = program("cpu")
    with torch_sql.Environment(make_engine(engine)):
        got = program("sql").cpu()
    assert torch.equal(got, expected), f"{alias}: got {got.tolist()}, PyTorch {expected.tolist()}"


@pytest.mark.pinned
@pytest.mark.xfail(strict=True, reason="an in-place op doesn't reach the tensor's aliases")
@pytest.mark.parametrize("engine", ENGINES)
def test_ema_of_weights_updates(engine):
    """The same finding, through PyTorch's own EMA, which Samudra's optimizer
    path uses (design §11.5): ``AveragedModel.update_parameters`` lerps
    ``p.detach()`` in place, so on torch-sql the average never moves."""
    from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

    import torch_sql

    def averaged(device):
        torch.manual_seed(0)
        model = torch.nn.Linear(2, 1).double()
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.5))
        if device == "sql":
            model, ema = model.to("sql"), ema.to("sql")
        for _ in range(2):
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(1.0)
            ema.update_parameters(model)
        return [p.detach().cpu() for p in ema.module.parameters()]

    expected = averaged("cpu")
    with torch_sql.Environment(make_engine(engine)):
        got = averaged("sql")
    for a, b in zip(got, expected, strict=True):
        torch.testing.assert_close(a, b)


@pytest.mark.pinned
@pytest.mark.xfail(strict=True, reason="DuckDB's sin and cos raise on an infinity")
@pytest.mark.parametrize("fn", ["sin", "cos"])
def test_trig_of_infinity_is_nan(fn):
    """PyTorch's sin(inf) and cos(inf) are NaN. DuckDB raises an
    OutOfRangeException mid-query, an engine error for an op torch-sql says it
    lowers, where ``log`` and ``sqrt`` already guard the same way with CASE."""
    import torch_sql

    with torch_sql.Environment(make_engine("duckdb")):
        x = torch.tensor([math.inf, 1.0], dtype=torch.float64, device="sql")
        out = getattr(torch, fn)(x).cpu()
    assert torch.isnan(out[0]) and out[1] == getattr(math, fn)(1.0)


@pytest.mark.pinned
@pytest.mark.parametrize(
    "engine",
    [
        pytest.param(
            "datafusion",
            marks=pytest.mark.xfail(
                strict=True, reason="DataFusion pushes a filter below an ungrouped aggregate"
            ),
        ),
        "duckdb",
    ],
)
def test_select_from_stacked_scalars(engine):
    """``torch.stack([a.sum(), b.sum()])[0]`` returns two rows on DataFusion
    55, one of them NULL. The SQL is right; DataFusion's optimizer isn't:

        SELECT * FROM (SELECT 1 AS k, SUM(v) AS s FROM t) WHERE k = 0

    returns the row (1, NULL). The filter is pushed below the aggregate,
    whose input becomes empty, and an ungrouped aggregate over no rows
    still returns one row. ``losses[0:1]``, which joins an index table
    instead of filtering, is right."""
    import torch_sql

    with torch_sql.Environment(make_engine(engine)):
        a = torch.ones(3, dtype=torch.float64, device="sql")
        losses = torch.stack([a.sum(), (2 * a).sum()])
        assert losses[0].item() == 3.0 and losses[1].item() == 6.0


@pytest.mark.parametrize("engine", ENGINES)
@given(programs(modes=("smooth",)))
def test_backward_twice_accumulates(engine, program):
    """``.grad`` accumulates, as in PyTorch: two ``backward()`` calls on two
    losses give the sum of their gradients."""
    from fuzz_programs import leaf_tensors, run
    from harness import weights

    import torch_sql

    note(repr(program))
    _, expected, screens = oracle(program, grad=True)
    assume(expected is not None and not screens.kink and not screens.domain)
    with refusals_are_skips(), torch_sql.Environment(make_engine(engine)):
        leaves = leaf_tensors(program, "sql", requires_grad=True)
        out = run(program, leaves)[output_of(program)]
        w = weights(out.shape).to("sql")
        (out * w).sum().backward()
        (out * w * 2.0).sum().backward()
        got = [
            torch.zeros(x.shape, dtype=x.dtype) if x.grad is None else x.grad.cpu() for x in leaves
        ]
    for k, (a, b) in enumerate(zip(got, expected, strict=True)):
        close(a, 3 * b, max(1.0, scale([b])) * 3, f"accumulated gradient of leaf {k}")
