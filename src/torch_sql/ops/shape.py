"""Views and shape ops (design §8.5). None moves data: each is a different
query over the same table. ``reshape`` beyond adding and removing
dimensions of size 1 isn't here yet."""

from __future__ import annotations

import torch

from .. import graph as g
from . import Context, op
from ._build import as_dtype

aten = torch.ops.aten


def same(ctx, x, *args, **kwargs):
    """Ops that change only how PyTorch refers to a tensor keep its node."""
    return x


op(aten.alias.default, aten.detach.default, aten.lift_fresh.default, aten.clone.default)(same)


def reindex(ctx, x: g.Node, index) -> g.Select:
    out = ctx.out
    return g.Select(tuple(out.shape), out.dtype, (x,), index=tuple(index), value=g.Col(0, "v"))


@op(aten.permute.default)
def permute(ctx, x, dims):
    dims = [d % x.rank for d in dims]
    return reindex(ctx, x, [g.Col(0, f"i{d}") for d in dims])


@op(aten.unsqueeze.default)
def unsqueeze(ctx, x, dim):
    dim = dim % (x.rank + 1)
    index = [g.Col(0, c) for c in x.columns]
    index.insert(dim, g.Lit(0, torch.int64))
    return reindex(ctx, x, index)


@op(aten.squeeze.dims, aten.squeeze.dim, aten.squeeze.default)
def squeeze(ctx, x, dim=None):
    dims = range(x.rank) if dim is None else ([dim] if isinstance(dim, int) else dim)
    dropped = {d % x.rank for d in dims if x.shape[d % x.rank] == 1} if x.rank else set()
    return reindex(ctx, x, [g.Col(0, f"i{d}") for d in range(x.rank) if d not in dropped])


@op(aten.view.default, aten._unsafe_view.default, aten.reshape.default)
def view(ctx, x, size):
    """A join with an index table for each run of dimensions the reshape
    merges or splits (§8.5, §9.5), so no index is computed."""
    from ..index_tables import reshape_groups, reshape_table

    out = tuple(ctx.out.shape)
    if out == x.shape:
        return x
    inputs, on = [x], []
    index: list = [g.Lit(0, torch.int64)] * len(out)
    for src, dst in reshape_groups(x.shape, out):
        if len(src) == 1 and len(dst) == 1:
            index[dst[0]] = g.Col(0, f"i{src[0]}")
            continue
        table = reshape_table(ctx.env, tuple(x.shape[d] for d in src), tuple(out[d] for d in dst))
        inputs.append(table)
        k = len(inputs) - 1
        on.extend((g.Col(0, f"i{d}"), g.Col(k, f"a{n}")) for n, d in enumerate(src))
        for n, d in enumerate(dst):
            index[d] = g.Col(k, f"b{n}")
    return g.Select(
        out, ctx.out.dtype, tuple(inputs), on=tuple(on), index=tuple(index), value=g.Col(0, "v")
    )


@op(aten.expand.default)
def expand(ctx, x, size, *, implicit=False):
    out = tuple(ctx.out.shape)
    lead = len(out) - x.rank
    inputs, index = [x], []
    for d, n in enumerate(out):
        k = d - lead
        if k >= 0 and x.shape[k] == n:
            index.append(g.Col(0, f"i{k}"))
        elif n == 1:
            index.append(g.Lit(0, torch.int64))
        else:
            # A new or size-1 dimension: one row per position, from a range.
            inputs.append(ctx.env.arange(n))
            index.append(g.Col(len(inputs) - 1, "i0"))
    return g.Select(out, ctx.out.dtype, tuple(inputs), index=tuple(index), value=g.Col(0, "v"))


@op(aten.select.int)
def select(ctx, x, dim, index):
    """A join with a one-row table holding ``index``, rather than a filter
    on a constant. DataFusion 55 pushes a filter like ``i0 = 0`` below an
    aggregate with no ``GROUP BY``, which then returns a row of NULLs, so
    ``torch.stack([a.sum(), b.sum()])[0]`` would have two rows. A join is
    also the form einfold and ddx want (§9.5)."""
    dim = dim % x.rank
    index = index % x.shape[dim]
    out = ctx.out
    return g.Select(
        tuple(out.shape),
        out.dtype,
        (x, ctx.env.point(index)),
        on=((g.Col(0, f"i{dim}"), g.Col(1, "v")),),
        index=tuple(g.Col(0, f"i{d}") for d in range(x.rank) if d != dim),
        value=g.Col(0, "v"),
    )


@op(aten.unbind.int)
def unbind(ctx, x, dim=0):
    """``select`` at each index of ``dim``: ``q, k, v = qkv.unbind(0)``."""
    dim = dim % x.rank
    return [select(Context(ctx.func, out, ctx.env), x, dim, k) for k, out in enumerate(ctx.out)]


@op(aten.slice.Tensor)
def slice_(ctx, x, dim=0, start=None, end=None, step=1):
    dim = dim % x.rank
    n = x.shape[dim]
    start, end, _ = slice(start, end, step).indices(n)
    if step != 1:
        raise NotImplementedError("a slice with a step")
    if (start, end) == (0, n):
        return x
    i = g.Col(0, f"i{dim}")
    out = ctx.out
    index = [g.Col(0, c) for c in x.columns]
    index[dim] = i - start if start else i
    return g.Select(
        tuple(out.shape),
        out.dtype,
        (x,),
        where=g.call("and", g.call(">=", i, start), g.call("<", i, end)),
        index=tuple(index),
        value=g.Col(0, "v"),
    )


@op(aten.cat.default)
def cat(ctx, tensors, dim=0):
    """A ``UNION ALL`` of the inputs, each with its offset along ``dim``.
    Each part is declared with the output's shape, since it holds a slab of
    the output's rows."""
    out = ctx.out
    shape = tuple(out.shape)
    dim = dim % len(shape)
    parts, offset = [], 0
    for t in tensors:
        if t.numel == 0:  # PyTorch skips empty inputs, even of another shape
            continue
        index = [g.Col(0, c) for c in t.columns]
        if offset:
            index[dim] = index[dim] + offset
        parts.append(
            g.Select(
                shape,
                out.dtype,
                (t,),
                index=tuple(index),
                value=as_dtype(g.Col(0, "v"), t.dtype, out.dtype),
            )
        )
        offset += t.shape[dim]
    return parts[0] if len(parts) == 1 else g.UnionAll(shape, out.dtype, tuple(parts))
