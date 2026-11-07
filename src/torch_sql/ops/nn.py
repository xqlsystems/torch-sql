"""Neural-network ops (design §9.4): embeddings, gathers, norms, softmax and
dropout.

An embedding, or a gather, is a join on an index *value*: the token id in
row ``(b, t)`` of ``idx`` meets the row of ``wte`` it names. Norms and
softmax are ``GROUP BY``s of the dimensions they keep, joined back. Each is
written out here as plain SQL; none needs a function the engines lack.
"""

from __future__ import annotations

import torch

from .. import graph as g
from . import op
from ._build import elementwise, normalize_dims, scalar

aten = torch.ops.aten
F64 = torch.float64


def grouped(x: g.Node, dims, agg: str, value=None) -> g.Select:
    """``agg`` of ``value`` (``x``'s value by default) over ``dims``,
    keeping them as dimensions of size 1."""
    shape = tuple(1 if d in dims else n for d, n in enumerate(x.shape))
    index = tuple(g.Lit(0, torch.int64) if d in dims else g.Col(0, f"i{d}") for d in range(x.rank))
    v = g.Col(0, "v") if value is None else value
    return g.Select(shape, x.dtype, (x,), index=index, value=g.Agg(agg, v), group=True)


@op(aten.embedding.default)
def embedding(ctx, weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
    if scale_grad_by_freq or sparse:
        raise NotImplementedError("embedding with scale_grad_by_freq or sparse gradients")
    # padding_idx only changes the gradient, which PyTorch then leaves at 0
    # for that row; ddx would give it one.
    if padding_idx is not None and padding_idx >= 0:
        raise NotImplementedError("embedding with a padding_idx")
    out = ctx.out
    index = tuple(g.Col(0, c) for c in indices.columns) + (g.Col(1, "i1"),)
    return g.Select(
        tuple(out.shape),
        out.dtype,
        (indices, weight),
        on=((g.Col(0, "v"), g.Col(1, "i0")),),
        index=index,
        value=g.Col(1, "v"),
    )


@op(aten.gather.default)
def gather(ctx, x, dim, index, *, sparse_grad=False):
    """``out[i][j] = x[i][index[i][j]]`` for ``dim = 1``: the index's value
    names the position along ``dim``, and every other dimension matches."""
    dim = dim % x.rank if x.rank else 0
    # A scalar is gathered at index 0, which every index must be.
    on = (
        ()
        if x.rank == 0
        else tuple(
            (g.Col(0, "v") if d == dim else g.Col(0, f"i{d}"), g.Col(1, f"i{d}"))
            for d in range(index.rank)
        )
    )
    out = ctx.out
    return g.Select(
        tuple(out.shape),
        out.dtype,
        (index, x),
        on=on,
        index=tuple(g.Col(0, c) for c in index.columns),
        value=g.Col(1, "v"),
    )


@op(aten.native_layer_norm.default)
def native_layer_norm(ctx, x, normalized_shape, weight, bias, eps):
    """Returns the output, the mean and ``1 / sqrt(var + eps)``, as PyTorch
    does. The variance is a second pass over the centered values, not
    ``AVG(x²) - AVG(x)²``, which loses precision (§9.4)."""
    dims = set(range(x.rank - len(normalized_shape), x.rank))
    ctx.env.marked_cuts.add(x)  # a norm reads its input three times (D15)
    mean = grouped(x, dims, "AVG")
    centered = elementwise(x.shape, F64, [x, mean], lambda v, m: v - m)
    var = grouped(centered, dims, "AVG", g.Col(0, "v") * g.Col(0, "v"))
    rstd = elementwise(
        var.shape, F64, [var], lambda v: scalar(1.0, F64) / g.call("sqrt", v + scalar(eps, F64))
    )
    operands = [centered, rstd] + [t for t in (weight, bias) if t is not None]

    def fn(c, r, *affine):
        y = c * r
        affine = list(affine)
        if weight is not None:
            y = y * affine.pop(0)
        if bias is not None:
            y = y + affine.pop(0)
        return y

    return elementwise(x.shape, F64, operands, fn), mean, rstd


@op(aten._softmax.default)
def softmax(ctx, x, dim, half_to_float):
    """``exp(x - max) / sum(exp(x - max))`` along ``dim``. The max is SQL's
    plain ``MAX``: a NaN makes the whole slice NaN either way."""
    dims = set(normalize_dims(dim, x.rank))
    m = grouped(x, dims, "MAX")
    e = elementwise(x.shape, F64, [x, m], lambda v, mx: g.call("exp", v - mx))
    s = grouped(e, dims, "SUM")
    return elementwise(x.shape, F64, [e, s], lambda v, t: v / t)


@op(aten._log_softmax.default)
def log_softmax(ctx, x, dim, half_to_float):
    """``x - max - ln(sum(exp(x - max)))`` along ``dim``. The sum is at
    least 1, so ``ln`` needs no guard."""
    dims = set(normalize_dims(dim, x.rank))
    m = grouped(x, dims, "MAX")
    shifted = elementwise(x.shape, F64, [x, m], lambda v, mx: v - mx)
    s = grouped(shifted, dims, "SUM", g.call("exp", g.Col(0, "v")))
    return elementwise(x.shape, F64, [shifted, s], lambda v, t: v - g.call("ln", t))


@op(aten.native_dropout.default)
def native_dropout(ctx, x, p, train):
    """Dropout's mask is drawn on the CPU, as PyTorch's CPU kernel draws it,
    and registered; the output is ``x * mask / (1 - p)``."""
    out, mask_meta = ctx.out
    if not train or p == 0:
        from .creation import filled

        return x, filled(ctx.env, mask_meta.shape, torch.bool, True)
    noise = torch.empty(tuple(x.shape), dtype=x.dtype).bernoulli_(1 - p)
    mask = ctx.env.upload(noise.to(torch.bool))
    scale = 1.0 / (1.0 - p)
    y = elementwise(
        x.shape,
        F64,
        [x, mask],
        lambda v, k: g.call("case", k, v * scalar(scale, F64), scalar(0.0, F64)),
        compute=[F64, torch.bool],
    )
    return y, mask


@op(aten.bernoulli.p)
def bernoulli(ctx, x, p=0.5, *, generator=None):
    cpu = torch.empty(tuple(ctx.out.shape), dtype=ctx.out.dtype).bernoulli_(p, generator=generator)
    return ctx.env.upload(cpu)


@op(aten.index.Tensor)
def index(ctx, x, indices):
    """``x[..., idx, ...]`` with one integer index tensor: a join of the
    index's value (made non-negative) with ``x``'s position along that
    dimension. Several index tensors, and boolean masks, whose result's
    shape depends on values, aren't supported."""
    given = [(d, t) for d, t in enumerate(indices) if t is not None]
    if len(given) != 1 or given[0][1].dtype != torch.int64:
        raise NotImplementedError("indexing with more than one tensor, or with a boolean mask")
    dim, idx = given[0]
    n = x.shape[dim]
    v = g.Col(1, "v")
    position = g.call("case", g.call("<", v, g.Lit(0, torch.int64)), v + n, v)
    on = ((g.Col(0, f"i{dim}"), position),)
    out_index = (
        [g.Col(0, f"i{d}") for d in range(dim)]
        + [g.Col(1, c) for c in idx.columns]
        + [g.Col(0, f"i{d}") for d in range(dim + 1, x.rank)]
    )
    out = ctx.out
    return g.Select(
        tuple(out.shape), out.dtype, (x, idx), on=on, index=tuple(out_index), value=g.Col(0, "v")
    )
