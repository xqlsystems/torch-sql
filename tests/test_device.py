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


def test_sync_with_no_tensors_syncs_the_step(env):
    x = torch.randn(3, dtype=F64, device="sql")
    y = x * 2
    x.add_(1)
    torch_sql.sync()
    assert isinstance(y._node, torch_sql.graph.Table)
    assert isinstance(x._node, torch_sql.graph.Table)
    same(x - 1, y.cpu() / 2)


def test_sync_with_no_tensors_needs_an_environment():
    with pytest.raises(RuntimeError, match="active Environment"):
        torch_sql.sync()


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


@pytest.mark.parametrize(
    "alias",
    {
        "slice": lambda x: x[:2],
        "select": lambda x: x[1],
        "detach": lambda x: x.detach(),
        "view": lambda x: x.view(2, 2),
        "transpose": lambda x: x.view(2, 2).t(),
        "expand": lambda x: x.expand(3, 4),
        "chain": lambda x: x.detach()[1:].unsqueeze(0),
    }.items(),
    ids=lambda a: a[0],
)
def test_in_place_reaches_aliases(env, alias):
    # A view and .detach() share their base's storage, as in PyTorch (§8.6).
    def program(device):
        x = torch.zeros(4, dtype=F64, device=device)
        y = alias[1](x)
        x.add_(1).mul_(3)
        return y

    same(program("sql"), program("cpu"))


def test_writing_a_detached_alias_writes_the_base(env):
    x = torch.zeros(3, dtype=F64, device="sql")
    tail = x[1:]
    x.detach().add_(2)  # as AveragedModel's EMA does
    same(x, torch.full((3,), 2.0, dtype=F64))
    same(tail, torch.full((2,), 2.0, dtype=F64))


def test_deepcopy(env):
    import copy

    m = torch.nn.Linear(3, 2, dtype=F64, device="sql")
    c = copy.deepcopy(m)
    assert isinstance(c.weight, torch.nn.Parameter) and c.weight.requires_grad
    with torch.no_grad():
        c.weight.add_(1)
    same(c.weight, m.weight.cpu() + 1)


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


def test_true_division_of_integers_is_refused(env):
    # PyTorch gives float32, which torch-sql doesn't support yet (§8.2).
    a = torch.arange(1, 4, device="sql")
    with pytest.raises(NotImplementedError, match="float32"):
        a / a


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

# -NAN has its sign bit set, as the NaN x86 gives for 0/0 does, which
# DataFusion orders below -inf.
SPECIAL = torch.tensor([NAN, -NAN, INF, -INF, 0.0, -0.0, 1.0, -1.0, 2.5], dtype=F64)
N = len(SPECIAL)


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
        torch.sin,
        torch.cos,
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
        "sin",
        "cos",
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
    a = SPECIAL[:, None].expand(N, N).contiguous()
    b = SPECIAL[None, :].expand(N, N).contiguous()
    same(fn(a.to("sql"), b.to("sql")), fn(a, b))


@pytest.mark.parametrize(
    "fn", [torch.amax, torch.amin, torch.sum, torch.mean], ids=lambda f: f.__name__
)
def test_reductions_propagate_nan(env, fn):
    x = torch.tensor([[1.0, NAN, 3.0], [1.0, 2.0, 3.0]], dtype=F64)
    same(fn(x.to("sql"), 1), fn(x, 1))


def test_nan_from_zero_over_zero(env):
    z = torch.zeros(1, dtype=F64, device="sql")
    assert torch.isnan(torch.relu(z / z).cpu()).all()
    assert torch.isnan(torch.maximum(z / z, z + 1).cpu()).all()


@pytest.mark.parametrize("fn", [torch.amax, torch.amin], ids=lambda f: f.__name__)
def test_reductions_of_infinities(env, fn):
    # DataFusion's grouped MAX of values that are all -inf is f64::MIN.
    x = torch.tensor([[-INF, -INF], [INF, INF], [-INF, 1.0]], dtype=F64)
    same(fn(x.to("sql"), 1), fn(x, 1))
    same(fn(x.to("sql"), 1, keepdim=True), fn(x, 1, keepdim=True))


def test_select_from_stacked_scalars(env):
    # DataFusion pushes a filter on a constant index below an aggregate with
    # no GROUP BY, which then gives a row of NULLs; select joins instead.
    a = torch.ones(3, dtype=F64, device="sql")
    losses = torch.stack([a.sum(), (2 * a).sum()])
    assert (losses[0].item(), losses[1].item(), losses[-1].item()) == (3.0, 6.0, 6.0)


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
