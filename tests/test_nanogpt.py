"""nanoGPT on device="sql", trained step by step against PyTorch (M1).

The model is nanoGPT's ``model.py``, unchanged (``tests/nanogpt``). Its
exact GELU needs ``erf``, which torch-sql registers on each engine. On
DuckDB, ddxdb can't yet differentiate through it (xqlsystems/ddx#275), so
training there substitutes GELU's tanh form, as xarray-sql's
``benchmarks/ml/02_nanogpt.py`` does.
"""

import pytest
import torch
from nanogpt.model import GPT, GPTConfig

import torch_sql

F64 = torch.float64


def gpt(layers=1, dropout=0.0, dtype=F64, exact=True):
    config = GPTConfig(
        block_size=8, vocab_size=11, n_layer=layers, n_head=2, n_embd=8, dropout=dropout, bias=True
    )
    torch.manual_seed(0)
    model = GPT(config).to(dtype)
    if exact:
        return model
    for module in model.modules():
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate="tanh"))
    return model


def batch(step):
    torch.manual_seed(100 + step)
    return torch.randint(0, 11, (2, 8)), torch.randint(0, 11, (2, 8))


def close(got, expected, tol=1e-11):
    torch.testing.assert_close(got.cpu(), expected.detach(), rtol=tol, atol=tol)


def train(engine, layers, steps, dtype=F64, tol=1e-11):
    """Train nanoGPT as its train.py does (AdamW from configure_optimizers,
    gradients clipped to norm 1) on the CPU and on device="sql", and check
    the loss, every gradient and every parameter after each step."""
    # Exact GELU on DataFusion; on DuckDB, its tanh form (xqlsystems/ddx#275).
    exact = engine.name != "duckdb"
    ref = gpt(layers, dtype=dtype, exact=exact)
    ref_opt = ref.configure_optimizers(0.1, 1e-3, (0.9, 0.99), "cpu")
    with torch_sql.Environment(engine):
        model = gpt(layers, dtype=dtype, exact=exact).to("sql")
        assert model.lm_head.weight is model.transformer.wte.weight  # still tied
        opt = model.configure_optimizers(0.1, 1e-3, (0.9, 0.99), "cpu")
        for step in range(steps):
            idx, targets = batch(step)
            _, ref_loss = ref(idx, targets)
            ref_loss.backward()
            torch.nn.utils.clip_grad_norm_(ref.parameters(), 1.0)
            ref_opt.step()
            ref_opt.zero_grad(set_to_none=True)

            _, loss = model(idx.to("sql"), targets.to("sql"))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grads = [p.grad.cpu() for p in model.parameters()]
            opt.step()
            opt.zero_grad(set_to_none=True)

            close(loss, ref_loss, tol)
            for g, p, r in zip(grads, model.parameters(), ref.parameters(), strict=True):
                assert g.shape == p.shape and g.dtype == dtype
                close(p, r, tol)


def test_forward_and_gradients(engine):
    exact = engine.name != "duckdb"  # xqlsystems/ddx#275
    ref = gpt(exact=exact)
    idx, targets = batch(0)
    logits_ref, loss_ref = ref(idx, targets)
    loss_ref.backward()
    with torch_sql.Environment(engine):
        model = gpt(exact=exact).to("sql")
        logits, loss = model(idx.to("sql"), targets.to("sql"))
        close(logits, logits_ref)
        loss.backward()
        names = [n for n, _ in model.named_parameters()]
        assert names == [n for n, _ in ref.named_parameters()]
        for (name, p), r in zip(model.named_parameters(), ref.parameters(), strict=True):
            torch.testing.assert_close(p.grad.cpu(), r.grad, rtol=1e-11, atol=1e-11, msg=name)


def test_trains_one_layer(engine):
    train(engine, layers=1, steps=3)


def test_trains_two_layers(engine):
    # Without cut points, DataFusion runs out of memory planning two layers'
    # loss, since it plans a CTE again at every read (xarray-sql#267).
    train(engine, layers=2, steps=2)


def test_trains_in_float32(engine):
    # PyTorch's default dtype, as nanoGPT's train.py runs on the CPU.
    train(engine, layers=1, steps=3, dtype=torch.float32, tol=1e-5)


def test_inference_reads_the_last_position(engine):
    ref = gpt()
    idx, _ = batch(0)
    expected, _ = ref(idx)
    with torch_sql.Environment(engine):
        logits, loss = gpt().to("sql")(idx.to("sql"))
        assert loss is None
        close(logits, expected)


def test_exact_gelu_on_duckdb_is_refused_clearly(engine):
    if engine.name != "duckdb":
        pytest.skip("only DuckDB refuses a gradient through erf")
    model = gpt()
    idx, targets = batch(0)
    with torch_sql.Environment(engine):
        _, loss = model.to("sql")(idx.to("sql"), targets.to("sql"))
        with pytest.raises(NotImplementedError, match="ddx#275"):
            loss.backward()
