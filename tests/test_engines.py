"""ddx's gradient programs, on every engine."""

import pytest
import torch

from torch_sql import engines, sql
from torch_sql import graph as g
from torch_sql.boundary.layout import to_dense, to_table

F64 = torch.float64


def test_unknown_engine():
    with pytest.raises(ValueError, match="unknown engine"):
        engines.engine("postgres")


def test_grad_of_a_hand_built_loss(engine):
    # loss = sum((x @ w + b) ** 2), with b a scalar.
    x, w, b = (
        torch.randn(5, 3, dtype=F64),
        torch.randn(3, 2, dtype=F64),
        torch.tensor(0.5, dtype=F64),
    )
    tables = {}
    for name, t in (("x", x), ("w", w), ("b", b)):
        tables[name] = g.Table(tuple(t.shape), F64)
        engine.register(tables[name].name, to_table(t))
    tx, tw, tb = tables["x"], tables["w"], tables["b"]
    mm = g.Select(
        (5, 2),
        F64,
        (tx, tw),
        on=((g.Col(0, "i1"), g.Col(1, "i0")),),
        index=(g.Col(0, "i0"), g.Col(1, "i1")),
        value=g.Agg("SUM", g.Col(0, "v") * g.Col(1, "v")),
        group=True,
    )
    y = g.Select(
        (5, 2),
        F64,
        (mm, tb),
        index=(g.Col(0, "i0"), g.Col(0, "i1")),
        value=g.Col(0, "v") + g.Col(1, "v"),
    )
    loss = g.Select((), F64, (y,), value=g.Agg("SUM", g.Col(0, "v") * g.Col(0, "v")), group=True)

    out = engine.grad(sql.query(loss), [f"{tw.name}.v", f"{tb.name}.v"])
    value = torch.tensor(
        engine.query(f"SELECT v FROM {out.value}").column("v")[0].as_py(), dtype=F64
    )
    gw = to_dense(engine.query(f"SELECT * FROM {out.gradients[0]}"), w.shape, F64)
    gb = engine.query(f"SELECT * FROM {out.gradients[1]}")
    for name in out.tables():
        engine.drop(name)

    wr, br = w.clone().requires_grad_(), b.clone().requires_grad_()
    ref = ((x @ wr + br) ** 2).sum()
    ref.backward()
    torch.testing.assert_close(value, ref.detach(), rtol=1e-14, atol=0)
    torch.testing.assert_close(gw, wr.grad, rtol=1e-13, atol=1e-13)
    assert gb.column_names == ["i", "v"]
    torch.testing.assert_close(
        torch.tensor(gb.column("v")[0].as_py(), dtype=F64), br.grad, rtol=1e-13, atol=0
    )
