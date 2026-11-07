"""Hand-built graphs, written as SQL and run on every engine."""

import math

import pytest
import torch

from torch_sql import graph as g
from torch_sql import sql
from torch_sql.boundary.layout import to_dense, to_table

F64 = torch.float64


def table(engine, t):
    node = g.Table(tuple(t.shape), t.dtype)
    engine.register(node.name, to_table(t))
    return node


def run(engine, node):
    return to_dense(engine.query(sql.query(node)), node.shape, node.dtype)


def test_projection(engine):
    x = torch.randn(3, 4, dtype=F64)
    a = table(engine, x)
    y = g.Select(
        a.shape,
        F64,
        (a,),
        index=(g.Col(0, "i0"), g.Col(0, "i1")),
        value=g.call("tanh", g.Col(0, "v")) * 2.0 + 1.0,
    )
    torch.testing.assert_close(run(engine, y), torch.tanh(x) * 2 + 1, rtol=0, atol=1e-15)


def test_broadcast_join(engine):
    x, b = torch.randn(3, 4, dtype=F64), torch.randn(4, dtype=F64)
    a, c = table(engine, x), table(engine, b)
    y = g.Select(
        (3, 4),
        F64,
        (a, c),
        on=((g.Col(0, "i1"), g.Col(1, "i0")),),
        index=(g.Col(0, "i0"), g.Col(0, "i1")),
        value=g.Col(0, "v") + g.Col(1, "v"),
    )
    torch.testing.assert_close(run(engine, y), x + b, rtol=0, atol=0)


def test_cross_join(engine):
    x, b = torch.randn(3, dtype=F64), torch.randn(2, dtype=F64)
    a, c = table(engine, x), table(engine, b)
    y = g.Select(
        (3, 2),
        F64,
        (a, c),
        index=(g.Col(0, "i0"), g.Col(1, "i0")),
        value=g.Col(0, "v") * g.Col(1, "v"),
    )
    torch.testing.assert_close(run(engine, y), x[:, None] * b[None, :], rtol=0, atol=0)


def test_contraction(engine):
    x, w = torch.randn(5, 3, dtype=F64), torch.randn(3, 2, dtype=F64)
    a, b = table(engine, x), table(engine, w)
    y = g.Select(
        (5, 2),
        F64,
        (a, b),
        on=((g.Col(0, "i1"), g.Col(1, "i0")),),
        index=(g.Col(0, "i0"), g.Col(1, "i1")),
        value=g.Agg("SUM", g.Col(0, "v") * g.Col(1, "v")),
        group=True,
    )
    torch.testing.assert_close(run(engine, y), x @ w, rtol=1e-14, atol=1e-14)


def test_reduce_to_scalar(engine):
    x = torch.randn(4, 3, dtype=F64)
    a = table(engine, x)
    y = g.Select((), F64, (a,), value=g.Agg("AVG", g.Col(0, "v")), group=True)
    torch.testing.assert_close(run(engine, y), x.mean(), rtol=1e-14, atol=0)


def test_chain_of_ctes(engine):
    x = torch.randn(4, dtype=F64)
    a = table(engine, x)
    b = g.Select(a.shape, F64, (a,), index=(g.Col(0, "i0"),), value=g.Col(0, "v") * 3.0)
    c = g.Select(
        a.shape,
        F64,
        (b, a),
        on=((g.Col(0, "i0"), g.Col(1, "i0")),),
        index=(g.Col(0, "i0"),),
        value=g.Col(0, "v") - g.Col(1, "v"),
    )
    torch.testing.assert_close(run(engine, c), 2 * x, rtol=0, atol=1e-15)


def test_union_all_and_filter(engine):
    x = torch.randn(4, dtype=F64)
    a = table(engine, x)
    # cat([x[:2], x[2:]]) written as a filter with an offset, then a union.
    lo = g.Select(
        (2,),
        F64,
        (a,),
        where=g.call("<", g.Col(0, "i0"), 2),
        index=(g.Col(0, "i0"),),
        value=g.Col(0, "v"),
    )
    hi = g.Select(
        (2,),
        F64,
        (a,),
        where=g.call(">=", g.Col(0, "i0"), 2),
        index=(g.Col(0, "i0") - 2 + 2,),
        value=g.Col(0, "v"),
    )
    torch.testing.assert_close(run(engine, g.UnionAll((4,), F64, (lo, hi))), x, rtol=0, atol=0)


def test_values(engine):
    v = g.Values((2,), F64, rows=((0, 0.25), (1, 0.75)))
    torch.testing.assert_close(run(engine, v), torch.tensor([0.25, 0.75], dtype=F64))


@pytest.mark.parametrize("x", [0.1, 1e-300, 5e-324, 1.7976931348623157e308, -0.0, 3.0])
def test_float_literals_keep_their_bits(engine, x):
    v = g.Values((), F64, rows=((x,),))
    got = run(engine, v).item()
    assert got == x and math.copysign(1, got) == math.copysign(1, x)


@pytest.mark.parametrize("x", [math.inf, -math.inf])
def test_infinite_literals(engine, x):
    assert run(engine, g.Values((), F64, rows=((x,),))).item() == x


def test_float_literal_is_double_on_every_engine(engine):
    # 0.1 * 3 is 0.30000000000000004 in DOUBLE; DuckDB's DECIMAL gives 0.3.
    v = g.Values((), F64, rows=((3.0,),))
    y = g.Select((), F64, (v,), value=g.Col(0, "v") * 0.1)
    assert run(engine, y).item() == 0.1 * 3


def test_generated_names_are_prefixed():
    n = g.Select((), F64, (g.Values((), F64, rows=((1.0,),)),), value=g.Col(0, "v"))
    assert sql.cte_name(n).startswith(g.PREFIX)
    assert g.Table((), F64).name.startswith(g.PREFIX)


def test_select_checks_its_columns():
    a = g.Values((), F64, rows=((1.0,),))
    with pytest.raises(ValueError, match="input 1"):
        g.Select((), F64, (a,), value=g.Col(1, "v"))
    with pytest.raises(ValueError, match="aggregate"):
        g.Select((), F64, (a,), value=g.Agg("SUM", g.Col(0, "v")))


def test_deep_graph_does_not_recurse():
    node = g.Values((), F64, rows=((1.0,),))
    for _ in range(5000):
        node = g.Select((), F64, (node,), value=g.Col(0, "v") + 1.0)
    assert len(g.topological([node])) == 5001


def test_materialize(engine):
    x = torch.randn(3, dtype=F64)
    a = table(engine, x)
    y = g.Select(a.shape, F64, (a,), index=(g.Col(0, "i0"),), value=g.Col(0, "v") * 2.0)
    saved = g.Table(y.shape, F64)
    engine.materialize(saved.name, sql.query(y))
    torch.testing.assert_close(run(engine, saved), 2 * x, rtol=0, atol=0)
    engine.drop(saved.name)
