"""backward() through ddx, and training with torch.optim (design §11)."""

import ddxdb
import pytest
import torch
from torch.testing._internal.common_methods_invocations import op_db
from torch.utils._pytree import tree_flatten, tree_map

import torch_sql

F64 = torch.float64

#: Ops with samples torch-sql or ddx refuses to differentiate, and why.
GRAD_REFUSED = {
    "div": "rounding_mode",
    "pow": "ddx differentiates power(x, y) only when x or y is a constant",
    "nn.functional.linear": "a 1-d input, which PyTorch's matmul squeezes in place",
}


@pytest.fixture
def env(engine):
    with torch_sql.Environment(engine) as e:
        yield e


def close(got, expected, tol=1e-12):
    torch.testing.assert_close(got.cpu(), expected.detach(), rtol=tol, atol=tol)


def mlp():
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Linear(3, 5), torch.nn.Tanh(), torch.nn.Linear(5, 2)
    ).double()


OPTIMIZERS = {
    "sgd": lambda p: torch.optim.SGD(p, lr=0.1),
    "sgd-momentum": lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9, nesterov=True),
    "adam": lambda p: torch.optim.Adam(p, lr=0.01),
    "adam-foreach": lambda p: torch.optim.Adam(p, lr=0.01, foreach=True),
    "adamw": lambda p: torch.optim.AdamW(p, lr=0.01, weight_decay=0.1),
    "adamw-foreach": lambda p: torch.optim.AdamW(p, lr=0.01, weight_decay=0.1, foreach=True),
}


@pytest.mark.parametrize("optimizer", sorted(OPTIMIZERS))
def test_mlp_trains_as_pytorch_does(env, optimizer):
    torch.manual_seed(1)
    x, y = torch.randn(8, 3, dtype=F64), torch.randn(8, 2, dtype=F64)
    ref, model = mlp(), mlp().to("sql")
    ref_opt, opt = (
        OPTIMIZERS[optimizer](ref.parameters()),
        OPTIMIZERS[optimizer](model.parameters()),
    )
    sx, sy = x.to("sql"), y.to("sql")
    for _ in range(5):
        ref_opt.zero_grad()
        ref_loss = torch.nn.functional.mse_loss(ref(x), y)
        ref_loss.backward()
        torch.nn.utils.clip_grad_norm_(ref.parameters(), 0.5)
        ref_opt.step()
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(sx), sy)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
        close(loss, ref_loss)
        for p, r in zip(model.parameters(), ref.parameters(), strict=True):
            close(p.grad, r.grad)
            close(p, r)


def test_a_step_ends_with_tables(env):
    model = mlp().to("sql")
    opt = torch.optim.Adam(model.parameters())
    model(torch.randn(2, 3, dtype=F64, device="sql")).sum().backward()
    opt.step()
    for p in model.parameters():
        assert isinstance(p._node, torch_sql.graph.Table)
        for state in opt.state[p].values():
            if isinstance(state, torch_sql.Tensor):
                assert isinstance(state._node, torch_sql.graph.Table)


def test_loss_value_comes_with_the_gradients(env, monkeypatch):
    w = torch.randn(3, dtype=F64, device="sql", requires_grad=True)
    expected = (w.detach().cpu() ** 2).sum().item()
    loss = (w * w).sum()
    loss.backward()
    queries = []
    orig = env.engine.query
    monkeypatch.setattr(env.engine, "query", lambda q: queries.append(q) or orig(q))
    torch.testing.assert_close(loss.item(), expected)
    assert len(queries) == 1 and "__ddx_" in queries[0]


def test_gradients_accumulate(env):
    w = torch.randn(3, dtype=F64, device="sql", requires_grad=True)
    (w * w).sum().backward()
    (3 * w).sum().backward()
    close(w.grad, 2 * w.detach().cpu() + 3)


def test_backward_with_a_gradient(env):
    wc = torch.randn(3, dtype=F64, requires_grad=True)
    v = torch.randn(3, dtype=F64)
    (wc * wc).backward(v)
    w = wc.detach().to("sql").requires_grad_()
    (w * w).backward(v)
    close(w.grad, wc.grad)


def test_inputs_restricts_the_leaves(env):
    a = torch.randn(2, dtype=F64, device="sql", requires_grad=True)
    b = torch.randn(2, dtype=F64, device="sql", requires_grad=True)
    (a * b).sum().backward(inputs=[a])
    close(a.grad, b.detach().cpu())
    assert b.grad is None


def test_detach_stops_the_gradient(env):
    wc = torch.randn(3, dtype=F64, requires_grad=True)
    (wc * wc.detach()).sum().backward()
    w = wc.detach().to("sql").requires_grad_()
    (w * w.detach()).sum().backward()
    close(w.grad, wc.grad)


def test_no_grad_stops_the_gradient(env):
    wc = torch.randn(3, dtype=F64, requires_grad=True)
    with torch.no_grad():
        cc = wc * 2
    (wc * cc).sum().backward()
    w = wc.detach().to("sql").requires_grad_()
    with torch.no_grad():
        c = w * 2
    (w * c).sum().backward()
    close(w.grad, wc.grad)


def test_frozen_parameters_get_no_gradient(env):
    model = mlp().to("sql")
    model[0].weight.requires_grad_(False)
    model(torch.randn(2, 3, dtype=F64, device="sql")).sum().backward()
    assert model[0].weight.grad is None
    assert model[2].weight.grad is not None


def test_refusals(env):
    w = torch.randn(3, dtype=F64, device="sql", requires_grad=True)
    with pytest.raises(NotImplementedError, match="create_graph"):
        (w * w).sum().backward(create_graph=True)
    with pytest.raises(RuntimeError, match="does not require grad"):
        torch.ones(2, dtype=F64, device="sql").sum().backward()
    with pytest.raises(RuntimeError, match="scalar outputs"):
        (w * w).backward()


# -- Each op's gradient, on OpInfo's samples --------------------------------

#: The ops whose gradients ddx computes, by OpInfo name.
DIFFERENTIABLE = [
    "add",
    "sub",
    "mul",
    "div",
    "neg",
    "exp",
    "log",
    "sqrt",
    "rsqrt",
    "tanh",
    "sigmoid",
    "reciprocal",
    "pow",
    "nn.functional.relu",
    "clamp",
    "where",
    "sum",
    "mean",
    "amax",
    "mm",
    "bmm",
    "addmm",
    "t",
    "permute",
    "unsqueeze",
    "squeeze",
    "expand",
    "cat",
    "transpose",
    "select",
    "narrow",
    "reshape",
    "view",
    "nn.functional.linear",
    "nn.functional.mse_loss",
    "abs",
    "sin",
    "cos",
    "maximum",
    "minimum",
    "amin",
]


@pytest.mark.parametrize(
    "info",
    [o for o in op_db if o.name in DIFFERENTIABLE],
    ids=lambda o: f"{o.name}.{o.variant_test_name}".strip("."),
)
def test_op_gradient(engine, info):
    """For each sample: the gradient of sum(op(x) * c), for a fixed random c,
    with respect to every float input, matches PyTorch's autograd."""
    if F64 not in info.dtypes:
        pytest.skip("no float64 variant")
    env = torch_sql.Environment(engine)
    ran = refused = 0
    for sample in info.sample_inputs("cpu", F64, requires_grad=True):
        tensors = [
            t
            for t in tree_flatten((sample.input, sample.args, sample.kwargs))[0]
            if isinstance(t, torch.Tensor)
        ]
        if any(t.dtype not in (F64, torch.int64, torch.bool) for t in tensors):
            continue
        leaves = [
            t
            for t in tree_flatten((sample.input, sample.args, sample.kwargs))[0]
            if isinstance(t, torch.Tensor) and t.requires_grad
        ]
        out = info(sample.input, *sample.args, **sample.kwargs)
        outs = [o for o in tree_flatten(out)[0] if isinstance(o, torch.Tensor)]
        if not leaves or not outs or any(o.numel() == 0 or not o.requires_grad for o in outs):
            continue
        # Values within 1e-6 of a kink (relu, clamp, abs, maximum) or a tie
        # differ between subgradients; leave those samples out.
        if info.name in {
            "abs",
            "nn.functional.relu",
            "clamp",
            "maximum",
            "minimum",
            "amax",
            "amin",
        } and _near_kink(sample):
            continue
        torch.manual_seed(0)
        cotangents = [torch.randn(o.shape, dtype=o.dtype) for o in outs]
        expected = torch.autograd.grad(
            sum((o * c).sum() for o, c in zip(outs, cotangents, strict=True)),
            leaves,
            allow_unused=True,
        )
        sql_of = {id(t): env.tensor(t.detach()).requires_grad_() for t in leaves}

        def to_sql(x, sql_of=sql_of):
            if isinstance(x, torch.Tensor):
                return sql_of[id(x)] if id(x) in sql_of else env.tensor(x.detach())
            return x

        args = tree_map(to_sql, (sample.input, sample.args, sample.kwargs))
        try:
            with env:
                got_outs = [
                    o
                    for o in tree_flatten(info(args[0], *args[1], **args[2]))[0]
                    if isinstance(o, torch.Tensor)
                ]
                objective = sum(
                    (o * c.to("sql")).sum() for o, c in zip(got_outs, cotangents, strict=True)
                )
                objective.backward()
        except (NotImplementedError, ddxdb.DdxError) as e:
            assert info.name in GRAD_REFUSED, f"{sample}: {e}"
            refused += 1
            continue
        for t, e in zip(leaves, expected, strict=True):
            got = sql_of[id(t)].grad
            if e is None:
                assert got is None or not got.cpu().any()
            else:
                torch.testing.assert_close(
                    got.cpu(), e, rtol=1e-10, atol=1e-10, msg=lambda m, s=sample: f"{s}\n{m}"
                )
        ran += 1
    if not ran:
        pytest.skip(
            f"every sample refused: {GRAD_REFUSED[info.name]}"
            if refused
            else "no sample with a gradient"
        )


@pytest.mark.parametrize("p", [2, 3, 0.5, -1.5])
def test_power_of_a_constant(env, p):
    xc = torch.tensor([0.5, 1.5, 2.0], dtype=F64, requires_grad=True)
    (xc**p).sum().backward()
    x = xc.detach().to("sql").requires_grad_()
    (x**p).sum().backward()
    close(x.grad, xc.grad)


def _near_kink(sample) -> bool:
    values = [
        t.detach().flatten()
        for t in tree_flatten((sample.input, sample.args, sample.kwargs))[0]
        if isinstance(t, torch.Tensor) and t.is_floating_point()
    ]
    if not values:
        return False
    v = torch.cat(values)
    if (v.abs() < 1e-6).any():
        return True
    s = v.sort().values
    return bool((s[1:] - s[:-1]).abs().min() < 1e-6) if s.numel() > 1 else False
