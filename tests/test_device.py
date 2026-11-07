"""The device: creating tensors, recording ops, reading values."""

import gc
import math

import pytest
import torch

import torch_sql

F64 = torch.float64
NAN, INF = math.nan, math.inf


@pytest.fixture
def env(engine):
    with torch_sql.Environment(engine) as e:
        yield e


def same(got, expected):
    torch.testing.assert_close(got.cpu(), expected, rtol=1e-13, atol=1e-13, equal_nan=True)


# -- Creating tensors -------------------------------------------------------


def test_factories_match_the_cpu(env):
    torch.manual_seed(0)
    a = torch.randn(3, 4, dtype=F64, device="sql")
    b = torch.arange(5, device="sql")
    c = torch.full((2, 2), 0.5, dtype=F64, device="sql")
    torch.manual_seed(0)
    assert isinstance(a, torch_sql.Tensor)
    same(a, torch.randn(3, 4, dtype=F64))
    same(b, torch.arange(5))
    same(c, torch.full((2, 2), 0.5, dtype=F64))


def test_to_sql_and_back(env):
    x = torch.randn(3, dtype=F64)
    y = x.to("sql")
    assert isinstance(y, torch_sql.Tensor) and y.device.type == "meta"
    out = y.cpu()
    assert type(out) is torch.Tensor and torch.equal(out, x)
    assert torch.equal(y.to("cpu"), x)
    assert y.tolist() == x.tolist()
    assert (y.numpy() == x.numpy()).all()


def test_device_parses():
    assert torch.device("sql").type == "sql"


def test_requires_grad_carries_over(env):
    x = torch.randn(3, dtype=F64, requires_grad=True).to("sql")
    assert x.requires_grad
    assert torch.zeros(2, dtype=F64, device="sql", requires_grad=True).requires_grad


def test_module_to_sql(env):
    torch.manual_seed(0)
    m = torch.nn.Linear(3, 2).double()
    w = m.weight.detach().clone()
    m.to("sql")
    assert type(m.weight) is torch_sql.Tensor
    assert isinstance(m.weight, torch.nn.Parameter)
    same(m.weight, w)


def test_module_created_on_sql_matches_the_cpu(env):
    # nn.Linear initializes its weights in place, with uniform_: torch-sql
    # draws on the CPU, so the same seed gives the same weights.
    torch.manual_seed(1)
    m = torch.nn.Linear(4, 3, dtype=F64, device="sql")
    torch.manual_seed(1)
    ref = torch.nn.Linear(4, 3, dtype=F64)
    same(m.weight, ref.weight.detach())
    same(m.bias, ref.bias.detach())


def test_mlp_forward(env):
    torch.manual_seed(0)
    mlp = torch.nn.Sequential(
        torch.nn.Linear(3, 5), torch.nn.Tanh(), torch.nn.Linear(5, 2)
    ).double()
    x = torch.randn(4, 3, dtype=F64)
    expected = mlp(x).detach()
    mlp.to("sql")
    same(mlp(x.to("sql")), expected)


# -- Recording ---------------------------------------------------------------


def test_nothing_runs_until_read(env, monkeypatch):
    x = torch.randn(3, 3, dtype=F64, device="sql")
    calls = []
    for name in ("query", "materialize"):
        orig = getattr(env.engine, name)
        monkeypatch.setattr(env.engine, name, lambda *a, o=orig, n=name: calls.append(n) or o(*a))
    y = (x @ x).tanh().sum(0)
    assert calls == []
    y.cpu()
    assert calls


def test_sync_keeps_a_table(env):
    x = torch.randn(3, dtype=F64, device="sql")
    y = x * 2
    torch_sql.sync(y)
    assert isinstance(y._node, torch_sql.graph.Table)
    same(y + 1, x.cpu() * 2 + 1)


def test_tables_are_dropped_with_their_tensors(env):
    x = torch.randn(3, dtype=F64, device="sql")
    name = x._node.name
    assert env.engine.exists(name)
    del x
    gc.collect()
    assert not env.engine.exists(name)


def test_in_place_replaces_the_node(env):
    x = torch.ones(3, dtype=F64, device="sql")
    y = x * 1
    x.add_(2).mul_(3)
    same(x, torch.full((3,), 9.0, dtype=F64))
    same(y, torch.ones(3, dtype=F64))  # read x before the update


def test_in_place_keeps_the_dtype(env):
    x = torch.zeros(3, dtype=torch.int64, device="sql")
    with pytest.raises(RuntimeError):  # PyTorch's own refusal
        x.add_(torch.ones(3, dtype=F64, device="sql"))


def test_writing_through_a_view_is_refused(env):
    x = torch.zeros(3, 3, dtype=F64, device="sql")
    with pytest.raises(NotImplementedError, match="view"):
        x[0].add_(1)
    with pytest.raises(NotImplementedError, match="view"):
        x[:, 1] = 2.0


def test_copy_into(env):
    x = torch.zeros(2, 3, dtype=F64, device="sql")
    x.copy_(torch.arange(3, dtype=F64))
    same(x, torch.arange(3, dtype=F64).expand(2, 3))


def test_load_state_dict(env):
    m = torch.nn.Linear(3, 2, dtype=F64, device="sql")
    ref = torch.nn.Linear(3, 2, dtype=F64)
    with torch.no_grad():
        ref.weight.copy_(torch.arange(6, dtype=F64).view(2, 3))
    m.load_state_dict(ref.state_dict())
    same(m.weight, torch.arange(6, dtype=F64).view(2, 3))
    same(m.bias, ref.bias.detach())


def test_is_not_meta_to_pytorch(env):
    # PyTorch skips copying into a parameter that says it's on meta, as
    # load_state_dict does; a torch-sql tensor holds data, so it says no.
    assert torch.ones(2, dtype=F64, device="sql").is_meta is False


def test_python_control_flow_reads_values(env):
    x = torch.tensor([1.0, -2.0], dtype=F64, device="sql")
    assert bool((x > 0).any()) is True
    assert bool((x > 0).all()) is False
    assert float(x.sum()) == -1.0
    assert (x.abs().max() > 1).item()


def test_reshape(env):
    x = torch.arange(24, dtype=F64)
    y = x.to("sql")
    same(
        y.view(2, 3, 4).permute(2, 0, 1).reshape(4, 6),
        x.view(2, 3, 4).permute(2, 0, 1).reshape(4, 6),
    )
    same(y.view(6, 4)[:, 1:3], x.view(6, 4)[:, 1:3])


def test_mixed_environments_are_refused(engine):
    a = torch_sql.Environment(engine)
    b = torch_sql.Environment(engine)
    x, y = a.tensor(torch.ones(2, dtype=F64)), b.tensor(torch.ones(2, dtype=F64))
    with pytest.raises(RuntimeError, match="two torch-sql environments"):
        x + y


def test_cpu_tensors_beside_sql_ones(env):
    x = torch.ones(2, dtype=F64, device="sql")
    same(x * torch.tensor(3.0, dtype=F64), torch.full((2,), 3.0, dtype=F64))
    with pytest.raises(RuntimeError, match="CPU|cpu"):
        x + torch.ones(2, dtype=F64)


def test_unsupported_dtype_is_refused(env):
    with pytest.raises(NotImplementedError, match="float32"):
        torch.ones(2, device="sql")


def test_exact_gelu_is_refused(env):
    with pytest.raises(NotImplementedError, match="erf"):
        torch.nn.functional.gelu(torch.ones(2, dtype=F64, device="sql"))
    same(
        torch.nn.functional.gelu(torch.ones(2, dtype=F64, device="sql"), approximate="tanh"),
        torch.nn.functional.gelu(torch.ones(2, dtype=F64), approximate="tanh"),
    )


def test_repr(env):
    assert "device='sql'" in repr(torch.ones(2, dtype=F64, device="sql"))


# -- IEEE floating point: the SQL must mean what PyTorch means --------------

SPECIAL = torch.tensor([NAN, INF, -INF, 0.0, -0.0, 1.0, -1.0, 2.5], dtype=F64)


@pytest.mark.parametrize(
    "fn",
    [
        torch.log,
        torch.sqrt,
        torch.rsqrt,
        torch.exp,
        torch.tanh,
        torch.abs,
        torch.relu,
        torch.sigmoid,
        torch.reciprocal,
        lambda x: x.clamp(-1, 2),
        lambda x: x.clamp(min=0),
    ],
    ids=[
        "log",
        "sqrt",
        "rsqrt",
        "exp",
        "tanh",
        "abs",
        "relu",
        "sigmoid",
        "reciprocal",
        "clamp",
        "clamp_min",
    ],
)
def test_unary_special_values(env, fn):
    same(fn(SPECIAL.to("sql")), fn(SPECIAL))


@pytest.mark.parametrize(
    "fn",
    [
        torch.eq,
        torch.ne,
        torch.lt,
        torch.le,
        torch.gt,
        torch.ge,
        torch.maximum,
        torch.minimum,
        torch.add,
        torch.mul,
        torch.div,
    ],
    ids=lambda f: f.__name__,
)
def test_binary_special_values(env, fn):
    a = SPECIAL[:, None].expand(8, 8).contiguous()
    b = SPECIAL[None, :].expand(8, 8).contiguous()
    same(fn(a.to("sql"), b.to("sql")), fn(a, b))


@pytest.mark.parametrize(
    "fn", [torch.amax, torch.amin, torch.sum, torch.mean], ids=lambda f: f.__name__
)
def test_reductions_propagate_nan(env, fn):
    x = torch.tensor([[1.0, NAN, 3.0], [1.0, 2.0, 3.0]], dtype=F64)
    same(fn(x.to("sql"), 1), fn(x, 1))


def test_integer_sum_stays_integer(env):
    # DuckDB's SUM of BIGINT is a DECIMAL; torch-sql casts it back.
    x = torch.arange(6).reshape(2, 3)
    y = x.to("sql").sum(1)
    assert y.dtype == torch.int64
    assert torch.equal(y.cpu(), x.sum(1))


def test_integer_tensor_against_a_float(env):
    x = torch.tensor([1, 2, 3])
    same(x.to("sql") < 1.5, x < 1.5)
    same(x.to("sql") == 2.0, x == 2.0)


def test_bitwise_of_integers_is_refused(env):
    with pytest.raises(NotImplementedError, match="integers"):
        torch.tensor([3], device="sql") & torch.tensor([5], device="sql")
    same(
        torch.tensor([True, False], device="sql") & torch.tensor([True, True], device="sql"),
        torch.tensor([True, False]),
    )
