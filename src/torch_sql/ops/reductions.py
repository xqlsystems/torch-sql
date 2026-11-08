"""Reductions (design §9.4): a ``GROUP BY`` of the dimensions they keep.

A reduced dimension that ``keepdim`` keeps has the index 0 in every row,
which is a constant, so it isn't grouped by.

PyTorch's ``amax`` and ``amin`` return NaN for a group holding one. SQL's
don't, reliably: DataFusion's ``MAX`` returns NaN without a ``GROUP BY`` and
skips it with one, and ``MIN`` skips it on both engines, since SQL orders
NaN above every number. So a float ``amax`` checks for NaN itself, and
``amin`` of floats is ``-amax(-x)``.

DataFusion 55's grouped ``MAX`` also gives ``f64::MIN`` (-1.797e308) for a
group whose values are all ``-inf``, where PyTorch gives ``-inf``, so a
float ``amax`` with a ``GROUP BY`` checks for that too. Its ``MAX`` without
one is right. A row of attention scores that a mask fills with ``-inf`` is
such a group.
"""

from __future__ import annotations

import torch

from .. import graph as g
from ..sql import cast_to
from . import Context, op
from ._build import as_dtype, normalize_dims

aten = torch.ops.aten


def reduce(ctx, x: g.Node, dim, keepdim: bool, agg: str, *, dtype=None) -> g.Select:
    out = ctx.out
    dims = normalize_dims(dim, x.rank)
    if any(x.shape[d] == 0 for d in dims):
        # PyTorch's sum over nothing is 0, its mean NaN; it refuses amax and
        # amin before this, on the meta device.
        from .creation import filled

        return filled(ctx.env, out.shape, out.dtype, {"SUM": 0, "AVG": float("nan")}[agg])
    index = []
    for d in range(x.rank):
        if d not in dims:
            index.append(g.Col(0, f"i{d}"))
        elif keepdim:
            index.append(g.Lit(0, torch.int64))
    compute = out.dtype
    v = as_dtype(g.Col(0, "v"), x.dtype, compute)
    value = g.Agg(agg, v)
    if agg == "MAX" and compute == torch.float64:
        flag = g.call("case", g.call("isnan", v), g.Lit(1, torch.int64), g.Lit(0, torch.int64))
        nan = g.call("=", g.Agg("MAX", flag), g.Lit(1, torch.int64))
        if any(d not in dims for d in range(x.rank)):
            above = g.call(
                "case",
                g.call(">", v, g.Lit(float("-inf"), torch.float64)),
                g.Lit(1, torch.int64),
                g.Lit(0, torch.int64),
            )
            none_above = g.call("=", g.Agg("MAX", above), g.Lit(0, torch.int64))
            value = g.call("case", none_above, g.Lit(float("-inf"), torch.float64), value)
        value = g.call("case", nan, g.Lit(float("nan"), torch.float64), value)
    if compute == torch.int64:
        # DuckDB's SUM of BIGINT is a 128-bit DECIMAL.
        value = cast_to(value, torch.int64)
    return g.Select(tuple(out.shape), out.dtype, (x,), index=tuple(index), value=value, group=True)


@op(aten.sum.dim_IntList, aten.sum.default)
def sum_(ctx, x, dim=None, keepdim=False, *, dtype=None):
    return reduce(ctx, x, dim, keepdim, "SUM")


@op(aten.mean.dim, aten.mean.default)
def mean(ctx, x, dim=None, keepdim=False, *, dtype=None):
    return reduce(ctx, x, dim, keepdim, "AVG")


@op(aten.amax.default, aten.max.default)
def amax(ctx, x, dim=(), keepdim=False):
    return reduce(ctx, x, dim, keepdim, "MAX")


@op(aten.amin.default, aten.min.default)
def amin(ctx, x, dim=(), keepdim=False):
    if x.dtype != torch.float64:
        return reduce(ctx, x, dim, keepdim, "MIN")
    minus = g.Lit(-1.0, torch.float64)
    negated = g.Select(
        x.shape,
        x.dtype,
        (x,),
        index=tuple(g.Col(0, c) for c in x.columns),
        value=minus * g.Col(0, "v"),
    )
    m = reduce(ctx, negated, dim, keepdim, "MAX")
    return g.Select(
        m.shape,
        m.dtype,
        (m,),
        index=tuple(g.Col(0, c) for c in m.columns),
        value=minus * g.Col(0, "v"),
    )


def _arg_extremum(ctx, x, dim, keepdim, extremum, shape):
    """``argmax`` or ``argmin`` along ``dim`` (all of ``x`` when None,
    giving the index into its flattened form): the extremum, joined back to
    ``x`` on the kept dimensions, and the least index whose value equals
    it, since PyTorch gives the first. A NaN is the extremum, as in
    ``amax``. The result has ``shape``."""
    dims = tuple(range(x.rank)) if dim is None else (dim % max(x.rank, 1),)
    dims = tuple(d for d in dims if d < x.rank)
    kept = [d for d in range(x.rank) if d not in dims]
    meta = torch.empty([x.shape[d] for d in kept], dtype=x.dtype, device="meta")
    m = extremum(Context(ctx.func, meta, ctx.env), x, dims, False)
    v, best = g.Col(0, "v"), g.Col(1, "v")
    hit = g.call("=", v, best)
    if x.dtype == torch.float64:
        hit = g.call("or", hit, g.call("and", g.call("isnan", v), g.call("isnan", best)))
    flat, stride = g.Lit(0, torch.int64), 1
    for d in reversed(dims):
        flat = g.call("+", flat, g.call("*", g.Col(0, f"i{d}"), g.Lit(stride, torch.int64)))
        stride *= x.shape[d]
    index = []
    for d in range(x.rank):
        if d in kept:
            index.append(g.Col(0, f"i{d}"))
        elif keepdim and dim is not None:
            index.append(g.Lit(0, torch.int64))
    if keepdim and dim is None:
        index = [g.Lit(0, torch.int64)] * x.rank
    return g.Select(
        tuple(shape),
        torch.int64,
        (x, m),
        on=tuple((g.Col(0, f"i{d}"), g.Col(1, f"i{k}")) for k, d in enumerate(kept)),
        where=hit,
        index=tuple(index),
        value=g.Agg("MIN", flat),
        group=True,
    )


@op(aten.argmax.default)
def argmax(ctx, x, dim=None, keepdim=False):
    return _arg_extremum(ctx, x, dim, keepdim, amax, ctx.out.shape)


@op(aten.argmin.default)
def argmin(ctx, x, dim=None, keepdim=False):
    return _arg_extremum(ctx, x, dim, keepdim, amin, ctx.out.shape)


@op(aten.max.dim)
def max_dim(ctx, x, dim, keepdim=False):
    values, indices = ctx.out
    meta = Context(ctx.func, values, ctx.env)
    return amax(meta, x, (dim,), keepdim), _arg_extremum(ctx, x, dim, keepdim, amax, indices.shape)


@op(aten.min.dim)
def min_dim(ctx, x, dim, keepdim=False):
    values, indices = ctx.out
    meta = Context(ctx.func, values, ctx.env)
    return amin(meta, x, (dim,), keepdim), _arg_extremum(ctx, x, dim, keepdim, amin, indices.shape)


def _truth_reduce(ctx, x, dim, keepdim, agg):
    """``any`` and ``all``: the ``MAX`` or ``MIN`` of each element's truth
    as 0 or 1."""
    truth = g.call(
        "case",
        g.call("<>", g.Col(0, "v"), g.Lit(0, x.dtype)) if x.dtype != torch.bool else g.Col(0, "v"),
        g.Lit(1, torch.int64),
        g.Lit(0, torch.int64),
    )
    ints = g.Select(
        x.shape, torch.int64, (x,), index=tuple(g.Col(0, c) for c in x.columns), value=truth
    )
    out = ctx.out
    dims = normalize_dims(dim, x.rank)
    if any(x.shape[d] == 0 for d in dims):
        from .creation import filled

        return filled(ctx.env, out.shape, out.dtype, agg == "MIN")
    index = [
        g.Col(0, f"i{d}") if d not in dims else g.Lit(0, torch.int64)
        for d in range(x.rank)
        if d not in dims or keepdim
    ]
    return g.Select(
        tuple(out.shape),
        torch.bool,
        (ints,),
        index=tuple(index),
        value=g.call("=", g.Agg(agg, g.Col(0, "v")), g.Lit(1, torch.int64)),
        group=True,
    )


@op(aten.any.default, aten.any.dim, aten.any.dims)
def any_(ctx, x, dim=None, keepdim=False):
    return _truth_reduce(ctx, x, dim, keepdim, "MAX")


@op(aten.all.default, aten.all.dim, aten.all.dims)
def all_(ctx, x, dim=None, keepdim=False):
    return _truth_reduce(ctx, x, dim, keepdim, "MIN")
