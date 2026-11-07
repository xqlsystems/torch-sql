"""Cut points, and backward passes across them (design §9.6)."""

import pytest
import torch

import torch_sql
from torch_sql import graph as g
from torch_sql import program

F64 = torch.float64


@pytest.fixture(params=[None, 1, 3, 10])
def budget(request):
    return request.param


@pytest.fixture
def env(engine, budget):
    """Every engine, with budgets small enough to cut nearly every node, and
    DuckDB with no cut points. DataFusion with none can't plan the deeper
    graphs here: that's what cut points are for (xarray-sql#267)."""
    if budget is None and engine.name == "datafusion":
        pytest.skip("DataFusion needs cut points")
    engine.cut_budget = budget
    with torch_sql.Environment(engine) as e:
        yield e


def mlp():
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Linear(3, 5), torch.nn.LayerNorm(5), torch.nn.Tanh(), torch.nn.Linear(5, 2)
    ).double()


def test_mlp_trains_across_cut_points(env):
    torch.manual_seed(1)
    x, y = torch.randn(8, 3, dtype=F64), torch.randn(8, 2, dtype=F64)
    ref, model = mlp(), mlp().to("sql")
    ref_opt = torch.optim.Adam(ref.parameters(), lr=0.01)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    sx, sy = x.to("sql"), y.to("sql")
    for _ in range(3):
        ref_opt.zero_grad()
        ref_loss = torch.nn.functional.mse_loss(ref(x), y)
        ref_loss.backward()
        ref_opt.step()
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(sx), sy)
        loss.backward()
        opt.step()
        torch.testing.assert_close(loss.item(), ref_loss.item(), rtol=1e-12, atol=1e-12)
        for p, r in zip(model.parameters(), ref.parameters(), strict=True):
            torch.testing.assert_close(p.cpu(), r.detach(), rtol=1e-12, atol=1e-12)


def test_a_value_read_twice_and_a_scalar_cut(env):
    # w is read along two paths and through a scalar, so cotangents of the
    # same cut point add up, and a 0-d cut point joins its cotangent.
    wc = torch.randn(4, dtype=F64, requires_grad=True)
    s = (wc * wc).sum()
    ((wc * s).tanh() + wc.exp() * s).sum().backward()
    w = wc.detach().to("sql").requires_grad_()
    s = (w * w).sum()
    ((w * s).tanh() + w.exp() * s).sum().backward()
    torch.testing.assert_close(w.grad.cpu(), wc.grad, rtol=1e-12, atol=1e-12)


def test_forward_values_across_cut_points(env):
    x = torch.randn(4, 5, dtype=F64)
    y = x
    for _ in range(6):
        y = torch.nn.functional.layer_norm(y * y + y, (5,))
    z = x.to("sql")
    for _ in range(6):
        z = torch.nn.functional.layer_norm(z * z + z, (5,))
    torch.testing.assert_close(z.cpu(), y, rtol=1e-12, atol=1e-12)


def test_cut_points_bound_every_query():
    a = g.Values((), F64, rows=((1.0,),))
    node = a
    for _ in range(20):  # each node reads the one before twice: 2^20 inlined
        node = g.Select((), F64, (node, node), value=g.Col(0, "v") + g.Col(1, "v"))
    cuts = program.cut_points(node, lambda n: n, 8, set())
    sizes, cut = {}, {c.id for c in cuts}
    for n in g.topological([node]):
        sizes[n.id] = 1 + sum(1 if i.id in cut else sizes[i.id] for i in n.inputs)
        assert sizes[n.id] <= 8 or n.id in cut
    assert program.cut_points(node, lambda n: n, None, set()) == []


def test_norm_inputs_are_cut_on_datafusion():
    with torch_sql.Environment("datafusion") as env:
        x = torch.randn(3, 4, dtype=F64, device="sql")
        h = x * 2
        torch.nn.functional.layer_norm(h, (4,))
        assert h._node in env.marked_cuts
