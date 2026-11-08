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
    "nn.functional.gelu": "exact GELU needs erf (xqlsystems/ddx#256)",
    "nn.functional.embedding": "padding_idx, max_norm or scale_grad_by_freq",
    "nn.functional.cross_entropy": "class weights or label smoothing",
    "nn.functional.scaled_dot_product_attention": "enable_gqa",
}

#: (op, engine, a phrase of the error): refusals of one engine only, each a
#: ddx issue.
ENGINE_REFUSED = {
    # A 0-d input's gather is a CTE reading only a one-row table's value.
    ("gather", "duckdb", "does not depend on any wrt column"): "xqlsystems/ddx#264",
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
    # AMSGrad's foreach path calls _foreach_maximum_, and there's no
    # aten.maximum_.
    "amsgrad-foreach": lambda p: torch.optim.Adam(p, lr=0.01, amsgrad=True, foreach=True),
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


def test_ema_of_weights(env):
    # AveragedModel deep-copies the model, then lerps p.detach() in place,
    # which must reach the averaged parameters (§8.6).
    from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

    def averaged(device):
        model = mlp()
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.5))
        if device == "sql":
            model, ema = model.to("sql"), ema.to("sql")
        for _ in range(2):
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(1.0)
            ema.update_parameters(model)
        return list(ema.module.parameters())

    for p, r in zip(averaged("sql"), averaged("cpu"), strict=True):
        close(p, r)


def test_gradient_of_a_sum_of_an_expand(env):
    # On DuckDB, ddx mistook arange's unread v for the other input's
    # (xqlsystems/ddx#266); arange tables now have no v.
    w = torch.tensor([-1.0], dtype=F64, device="sql").requires_grad_()
    w.expand(2).sum().backward()
    close(w.grad, torch.tensor([2.0], dtype=F64))


def _leaf_and_loss(program):
    """``program(device)`` returns a leaf and a loss: the leaf's gradient
    on the device, and on the CPU."""
    got = []
    for device in ("sql", "cpu"):
        leaf, loss = program(device)
        loss.backward()
        got.append(leaf.grad)
    return got


def _leaf(device):
    return torch.tensor([1.0, 2.0, 3.0], dtype=F64, device=device).requires_grad_()


def test_view_of_a_parameter_after_a_write(env):
    # A view of a parameter derives its node from the parameter's, which
    # backward() treats as the leaf (§8.6).
    def program(device):
        w = torch.nn.Parameter(_leaf(device).detach())
        v = w[:2]
        with torch.no_grad():
            w.mul_(2)  # as an optimizer step does
        return w, (v * v).sum()

    close(*_leaf_and_loss(program))


@pytest.mark.parametrize("how", ["no_grad", "detach", "data"])
def test_untracked_write_to_a_tensor_with_history_is_refused(env, how):
    # PyTorch would keep y's history, MulBackward; torch-sql can't yet.
    w = _leaf("sql")
    y = w * 2
    with pytest.raises(NotImplementedError, match="autograd doesn't record"):
        if how == "no_grad":
            with torch.no_grad():
                y.add_(1)
        elif how == "detach":
            y.detach().add_(1)
        else:
            y.data.add_(1)


def test_tracked_write_to_a_tensor_with_history(env):
    def program(device):
        w = _leaf(device)
        y = w * 2
        y.mul_(3).add_(w).exp_()  # exp(7w)
        return w, y.sum()

    close(*_leaf_and_loss(program))


def test_leaf_written_after_use(env):
    # Each use of a leaf adds to its gradient, whichever value it read, as
    # PyTorch's AccumulateGrad does: 1 + 2 + 3.
    def program(device):
        w = _leaf(device)
        c = w * 2
        with torch.no_grad():
            w.add_(0.5)
        d = w * 3
        with torch.no_grad():
            w.mul_(2)
        return w, w.sum() + c.sum() + d.sum()

    close(*_leaf_and_loss(program))


def test_leaf_written_after_a_use_that_saved_it(env):
    w = _leaf("sql")
    c = w * w
    with torch.no_grad():
        w.add_(0.5)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        c.sum().backward()


def test_leaf_read_between_use_and_backward(env):
    # Reading a leaf syncs it, replacing its node with a table; the loss
    # reads the node from before.
    w = (torch.ones(3, dtype=F64, device="sql") * 2).requires_grad_()
    c = w * 2
    w.cpu()
    c.sum().backward()
    close(w.grad, torch.full((3,), 2.0, dtype=F64))


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


def test_reading_the_loss_first_keeps_its_gradient(env):
    # Reading a value saves it as a table; a tensor with a history keeps the
    # history too, or backward() would see a table, a constant.
    wc = torch.randn(3, dtype=F64, requires_grad=True)
    (wc * wc).sum().backward()
    w = wc.detach().to("sql").requires_grad_()
    hidden = w * w
    hidden.cpu()
    loss = hidden.sum()
    loss.item()
    loss.backward()
    close(w.grad, wc.grad)


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
    # nanoGPT's ops
    "nn.functional.embedding",
    "nn.functional.layer_norm",
    "softmax",
    "log_softmax",
    "nn.functional.cross_entropy",
    "gather",
    "masked_fill",
    "split",
    "split_with_sizes",
    "nn.functional.scaled_dot_product_attention",
    "nn.functional.gelu",
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
        if sample.kwargs.get("dropout_p"):  # random, so it can't match PyTorch
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
            known = [
                k for k in ENGINE_REFUSED if k[:2] == (info.name, engine.name) and k[2] in str(e)
            ]
            assert info.name in GRAD_REFUSED or known, f"{sample}: {e}"
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


# -- Found in review ----------------------------------------------------------


def test_copy_under_no_grad_is_a_constant(env):
    p = torch.tensor([1.0, 2.0], dtype=F64, device="sql", requires_grad=True)
    q = torch.zeros(2, dtype=F64, device="sql", requires_grad=True)
    with torch.no_grad():
        q.copy_(p)
    (p * q).sum().backward()
    close(p.grad, torch.tensor([1.0, 2.0], dtype=F64))
    close(q.grad, torch.tensor([1.0, 2.0], dtype=F64))


def test_load_state_dict_between_models_keeps_their_gradients_apart(env):
    # A target network loaded from another, as in EMA or DQN.
    a, b = mlp().to("sql"), mlp().to("sql")
    b.load_state_dict(a.state_dict())
    x = torch.randn(4, 3, dtype=F64)
    ref = mlp()
    ref_loss = (ref(x) * ref(x).detach()).sum()
    ref_loss.backward()
    (a(x.to("sql")) * b(x.to("sql"))).sum().backward()
    for p, q, r in zip(a.parameters(), b.parameters(), ref.parameters(), strict=True):
        close(p.grad, r.grad)
        close(q.grad, r.grad)


def test_clones_are_separate_leaves(env):
    x = torch.randn(3, dtype=F64, device="sql")
    a, b = x.clone().requires_grad_(), x.clone().requires_grad_()
    (a * 2 + b * 3).sum().backward()
    close(a.grad, torch.full((3,), 2.0, dtype=F64))
    close(b.grad, torch.full((3,), 3.0, dtype=F64))


def test_backward_under_no_grad(env):
    p = torch.tensor([1.0, 2.0], dtype=F64, device="sql", requires_grad=True)
    loss = (p * p).sum()
    with torch.no_grad():
        loss.backward()
    close(p.grad, torch.tensor([2.0, 4.0], dtype=F64))


@pytest.mark.parametrize(
    "other",
    [
        lambda s: s.detach(),
        lambda s: torch.tensor(2.0, dtype=F64, device="sql"),
        lambda s: torch.full_like(s, 2.0),
    ],
    ids=["detached", "constant", "full_like"],
)
def test_scalar_losses_with_scalar_operands(env, other):
    # d(s * o)/ds = o, with o a constant.
    s = torch.tensor(3.0, dtype=F64, device="sql", requires_grad=True)
    o = other(s)
    expected = o.detach().cpu().item()
    (s * o).backward()
    torch.testing.assert_close(s.grad.cpu().item(), expected)
