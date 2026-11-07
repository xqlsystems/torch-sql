"""Reductions (design §9.4): a ``GROUP BY`` of the dimensions they keep.

A reduced dimension that ``keepdim`` keeps has the index 0 in every row,
which is a constant, so it isn't grouped by.

PyTorch's ``amax`` and ``amin`` return NaN for a group holding one. SQL's
don't, reliably: DataFusion's ``MAX`` returns NaN without a ``GROUP BY`` and
skips it with one, and ``MIN`` skips it on both engines, since SQL orders
NaN above every number. So a float ``amax`` checks for NaN itself, and
``amin`` of floats is ``-amax(-x)``.
"""

from __future__ import annotations

import torch

from .. import graph as g
from ..sql import cast_to
from . import op
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
