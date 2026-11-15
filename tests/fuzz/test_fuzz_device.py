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


@pytest.mark.parametrize("engine", ENGINES)
@given(programs(dtype=torch.float32))
def test_float32_values_match_pytorch(engine, program):
    """float32, PyTorch's default dtype, in every mode: each op rounds to
    float32 as PyTorch's do, so values agree to a few float32 roundings
    (:data:`harness.TOLERANCE`), and each result is float32."""
    check(program, engine, grad=False)


@pytest.mark.parametrize("engine", ENGINES)
@given(programs(modes=("smooth",), dtype=torch.float32))
def test_float32_gradients_match_autograd(engine, program):
    """ddx's gradients of a float32 program are float32, and autograd's."""
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


# -- Fixed findings ----------------------------------------------------------
# Each was found by the tests above, shrunk by hand, and fixed in the stack.
# They stay as regression tests. Each docstring says what used to happen.


@pytest.mark.parametrize(
    "engine",
    [
        "datafusion",
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


@pytest.mark.parametrize(
    "engine",
    [
        "datafusion",
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


@pytest.mark.parametrize(
    "engine",
    [
        "datafusion",
        "duckdb",
    ],
)
def test_amax_of_infinities(engine):
    """On DataFusion 55, a grouped ``MAX`` over values that are all -inf is
    ``f64::MIN`` (-1.797e308), and a grouped ``MIN`` over +inf is
    ``f64::MAX``; without a ``GROUP BY`` they're right. torch-sql's ``amax``
    and ``amin`` with a ``dim`` return those numbers. An attention row that
    a mask fills with -inf meets this."""
    import torch_sql

    with torch_sql.Environment(make_engine(engine)):
        x = torch.full((2, 3), -math.inf, dtype=torch.float64, device="sql")
        assert x.amax(1).cpu().tolist() == [-math.inf, -math.inf]
        assert (-x).amin(1).cpu().tolist() == [math.inf, math.inf]


@pytest.mark.parametrize(
    "engine",
    [
        "datafusion",
        "duckdb",
    ],
)
def test_gradient_of_a_sum_of_an_expand(engine):
    """``w.expand(2).sum()`` is ``2 * w``. On DuckDB, ddx raises ``NotScalar:
    the loss v does not depend on any wrt column``. Multiplying by a weight
    first (``(w.expand(2) * c).sum()``) works."""
    import torch_sql

    with torch_sql.Environment(make_engine(engine)):
        w = torch.tensor([-1.0], dtype=torch.float64, device="sql").requires_grad_()
        w.expand(2).sum().backward()
        assert w.grad.cpu().tolist() == [2.0]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("fn", ["logsigmoid", "bce_with_logits"])
def test_gradient_at_a_logit_of_zero(engine, fn):
    """PyTorch decomposes ``logsigmoid`` as ``min(x, 0) - log1p(exp(-|x|))``,
    and ``binary_cross_entropy_with_logits`` as ``(1 - y) * x -
    logsigmoid(x)``. The sum is smooth, but each term has a kink at 0, and
    ddx took 0 for each kink's derivative, where the two must cancel: the
    gradient at 0 was 0 for ``logsigmoid``, where PyTorch's is 0.5. A
    classifier head initialized to zeros gives logits of exactly 0 on its
    first step (the review of #29)."""
    f = {
        "logsigmoid": torch.nn.functional.logsigmoid,
        "bce_with_logits": lambda x: torch.nn.functional.binary_cross_entropy_with_logits(
            x, torch.full_like(x, 0.3), reduction="none"
        ),
    }[fn]
    got, expected = _gradients(f, [0.0, 1.0], engine)
    torch.testing.assert_close(got, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("engine", ENGINES)
def test_a_float32_infinity_keeps_the_other_gradients(engine):
    """A float32 result that passes float32's range is rounded to an
    infinity by a ``CASE`` whose branch is a constant, and ddx differentiated
    the constant: every gradient upstream of it was 0. PyTorch's rounding
    doesn't change a gradient, so ``log([0, 0.5]).sum()`` has a gradient of
    2 for the second element (the review of #29)."""
    got, expected = _gradients(torch.log, [0.0, 0.5], engine, torch.float32)
    assert got.tolist() == expected.tolist() == [math.inf, 2.0], got.tolist()


@pytest.mark.parametrize("engine", ENGINES)
def test_softplus_far_past_its_threshold(engine):
    """``softplus`` decomposes as ``where(x > 20, x, log1p(exp(x)))``. Past
    exp's overflow, the branch not taken had a gradient of inf / inf, and
    where's gradient multiplied it by 0, which gave NaN. PyTorch's is 1 (the
    review of #29)."""
    got, expected = _gradients(torch.nn.functional.softplus, [800.0, 1.0], engine)
    torch.testing.assert_close(got, expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("fn", ["sigmoid", "silu"])
def test_sigmoid_far_in_its_tail(engine, fn):
    """``sigmoid`` was lowered as ``1 / (1 + exp(-x))``, whose derivative,
    as written, is inf / inf below about -709, so the gradient was NaN
    (issue #35)."""
    f = {"sigmoid": torch.sigmoid, "silu": torch.nn.functional.silu}[fn]
    got, expected = _gradients(f, [-800.0, 1.0], engine)
    torch.testing.assert_close(got, expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize("engine", ENGINES)
def test_two_nested_binary_cross_entropies(engine):
    """``binary_cross_entropy_with_logits`` of its own output, after a
    ``softmax``, on one element, as the fuzzer drew it. ddxdb's DataFusion
    backend decodes Substrait, the serialized query plan, with a limit of
    100 protobuf messages, about 48 relations, and this step used to nest
    more, so ddxdb refused it. #29's decompositions nest less."""
    from fuzz_programs import Step

    program = Program(
        leaves=[(1,)],
        steps=[
            Step("softmax", (0,), (0, False)),
            Step("bce_logits", (1, 0)),
            Step("bce_logits", (2, 0)),
        ],
        data=[[-1.0]],
        mode="smooth",
    )
    _, expected, _ = oracle(program, grad=True)
    _, got = on_device(program, make_engine(engine), grad=True)
    close(got[0], expected[0], 1.0, "the gradient")


# -- Pinned findings ---------------------------------------------------------


@pytest.mark.pinned
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("fn", ["max", "min"])
def test_gradient_of_max_along_a_dim_at_a_tie(engine, fn):
    """``x.max(dim)`` returns values and indices, and PyTorch's gradient of
    the values goes to the returned index only. ``amax`` splits it among
    the tied elements, and when torch-sql lowered ``max.dim``'s values as
    ``amax``, a row ``[1, 3, 3]`` got ``[0, 0.5, 0.5]`` where PyTorch gives
    ``[0, 1, 0]`` (issue #20)."""
    import torch_sql

    def program(device):
        x = torch.tensor([[1.0, 3.0, 3.0], [0.0, 0.0, 0.0]], dtype=torch.float64)
        x = (x.to("sql") if device == "sql" else x).requires_grad_()
        values = getattr(x if fn == "max" else -x, fn)(1).values
        values.sum().backward()
        return x.grad.cpu()

    expected = program("cpu")
    with torch_sql.Environment(make_engine(engine)):
        got = program("sql")
    assert torch.equal(got, expected), f"got {got.tolist()}, PyTorch {expected.tolist()}"


@pytest.mark.pinned
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("how", ["argmax", "max_indices"])
def test_gradient_through_a_gather_by_argmax(engine, how):
    """``argmax`` joins ``x`` with its maximum and reads only ``x``'s
    columns, and DataFusion's optimizer makes that join a semi join, which
    ddx can't differentiate: ``UnsupportedExpression: a JOIN_TYPE_RIGHT_SEMI
    join on the path from a wrt table to the output``. backward() saves an
    integer node as a table, as reading it first would, which keeps the
    join out of ddx's program (issue #22)."""
    import torch_sql

    with torch_sql.Environment(make_engine(engine)):
        w = torch.tensor([[1.0, 5.0, 2.0], [7.0, 0.0, 3.0]], dtype=torch.float64)
        w = w.to("sql").requires_grad_()
        if how == "argmax":
            index = w.detach().argmax(1, keepdim=True)
        else:
            index = w.max(1, keepdim=True).indices
        w.gather(1, index).sum().backward()
        assert w.grad.cpu().tolist() == [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]


def _gradients(fn, values, engine, dtype=torch.float64):
    """``fn``'s gradient at ``values``, from PyTorch and from ``engine``."""
    import torch_sql

    def grad(device):
        x = torch.tensor(values, dtype=dtype)
        x = (x.to("sql") if device == "sql" else x).requires_grad_()
        fn(x).sum().backward()
        return x.grad.cpu()

    expected = grad("cpu")
    with torch_sql.Environment(make_engine(engine)):
        return grad("sql"), expected


@pytest.mark.pinned
@pytest.mark.parametrize(
    "engine",
    [
        "datafusion",
        pytest.param(
            "duckdb",
            marks=pytest.mark.xfail(
                strict=True, reason="xqlsystems/ddx#272: DuckDB raises casting a gradient to REAL"
            ),
        ),
    ],
)
@pytest.mark.parametrize("fn", ["reciprocal", "exp"])
def test_a_float32_gradient_past_its_range(engine, fn):
    """``1/x`` at 1e-30, a normal float32, has a gradient of -1e60, which is
    -inf in float32; ``exp(89)``'s is 4.5e38. ddx computes a gradient in
    DOUBLE and casts it to REAL, which DuckDB refuses for a value past
    REAL's range (``ConversionException``, xqlsystems/ddx#272)."""
    f, x, want = {
        "reciprocal": (torch.reciprocal, [1e-30, 1.0], [-math.inf, -1.0]),
        "exp": (torch.exp, [89.0, 0.0], [math.inf, 1.0]),
    }[fn]
    got, expected = _gradients(f, x, engine, torch.float32)
    assert got.tolist() == expected.tolist() == want, got.tolist()


@pytest.mark.pinned
@pytest.mark.parametrize(
    "engine",
    [
        "duckdb",
        pytest.param(
            "datafusion",
            marks=pytest.mark.xfail(strict=True, reason="#36: Arrow refuses 0 to a negative power"),
        ),
    ],
)
def test_a_negative_power_of_zero(engine):
    """``x.pow(-1)`` of 0 is inf in PyTorch; Arrow's ``power`` raises (#36)."""
    import torch_sql

    x = torch.tensor([0.0, 2.0], dtype=torch.float64)
    with torch_sql.Environment(make_engine(engine)):
        got = x.to("sql").pow(-1).cpu()
    assert got.tolist() == x.pow(-1).tolist(), got.tolist()


@pytest.mark.pinned
@pytest.mark.xfail(strict=True, reason="review of #29, round 3: sigmoid's float32 gradient")
@pytest.mark.parametrize("engine", ENGINES)
def test_binary_cross_entropy_of_a_saturated_sigmoid(engine):
    """In float32, ``sigmoid(x)`` rounds to exactly 1 above x of about 16.6.
    ``binary_cross_entropy``'s backward divides by ``max(p * (1 - p),
    1e-12)``, which is then 1e-12. PyTorch multiplies that by sigmoid's
    backward, ``out * (1 - out)`` of the same rounded output, which is 0.
    torch-sql computes sigmoid's derivative in DOUBLE from ``x``, about 2e-9
    at 20, so the gradient is about 1,400 where PyTorch's is 0."""

    def f(x):
        p = torch.sigmoid(x)
        return torch.nn.functional.binary_cross_entropy(
            p, torch.full_like(p, 0.3), reduction="none"
        )

    got, expected = _gradients(f, [17.0, 20.0, 1.0], engine, torch.float32)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)
