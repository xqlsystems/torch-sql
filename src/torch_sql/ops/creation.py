"""Ops that create tensors (design §8.7), and casts.

A constant tensor is a cross join of ranges, one per dimension, with the
value as a literal, so its size never reaches the SQL text. Random tensors
are drawn on the CPU, with PyTorch's CPU generator, and registered: the
same seed gives the same numbers as PyTorch on the CPU.
"""

from __future__ import annotations

import torch

from .. import graph as g
from . import op
from ._build import as_dtype, scalar

aten = torch.ops.aten


def filled(env, shape, dtype, value) -> g.Node:
    """A tensor of ``shape`` whose every element is ``value``."""
    shape = tuple(shape)
    if not shape:
        return g.Values((), dtype, rows=((scalar(value, dtype).value,),))
    ranges = tuple(env.arange(n) for n in shape)
    return g.Select(
        shape,
        dtype,
        ranges,
        index=tuple(g.Col(k, "i0") for k in range(len(shape))),
        value=scalar(value, dtype),
    )


@op(aten.full_like.default)
def full_like(ctx, x, fill_value, **kwargs):
    return filled(ctx.env, ctx.out.shape, ctx.out.dtype, fill_value)


@op(aten.empty_like.default, aten.zeros_like.default)
def zeros_like(ctx, x, **kwargs):
    # An empty tensor's values are undefined; zeros are a safe choice.
    return filled(ctx.env, ctx.out.shape, ctx.out.dtype, 0)


@op(aten.ones_like.default)
def ones_like(ctx, x, **kwargs):
    return filled(ctx.env, ctx.out.shape, ctx.out.dtype, 1)


@op(aten.new_full.default)
def new_full(ctx, x, size, fill_value, **kwargs):
    return filled(ctx.env, ctx.out.shape, ctx.out.dtype, fill_value)


@op(aten.new_zeros.default, aten.new_empty.default)
def new_zeros(ctx, x, size, **kwargs):
    return filled(ctx.env, ctx.out.shape, ctx.out.dtype, 0)


@op(aten.new_ones.default)
def new_ones(ctx, x, size, **kwargs):
    return filled(ctx.env, ctx.out.shape, ctx.out.dtype, 1)


@op(aten.fill.Scalar)
def fill(ctx, x, value):
    return filled(ctx.env, ctx.out.shape, ctx.out.dtype, value)


@op(aten._to_copy.default)
def to_copy(ctx, x, **kwargs):
    """A cast. A copy to the CPU never reaches here: the device reads the
    table instead (:meth:`torch_sql.device.environment.Environment.read`)."""
    dtype = ctx.out.dtype
    if dtype == x.dtype:
        return x
    return g.Select(
        x.shape,
        dtype,
        (x,),
        index=tuple(g.Col(0, c) for c in x.columns),
        value=as_dtype(g.Col(0, "v"), x.dtype, dtype),
    )


@op(aten.to.dtype)
def to_dtype(ctx, x, dtype, non_blocking=False, copy=False, memory_format=None):
    return to_copy(ctx, x)


def _drawn(fill):
    """A random op: the same draw on the CPU, registered as a table."""

    def lower(ctx, x, *args, **kwargs):
        cpu = torch.empty(tuple(ctx.out.shape), dtype=ctx.out.dtype)
        fill(cpu, *args, **kwargs)
        return ctx.env.upload(cpu)

    return lower


op(aten.uniform.default)(
    _drawn(lambda t, a=0.0, b=1.0, *, generator=None: t.uniform_(a, b, generator=generator))
)
op(aten.normal_functional.default)(
    _drawn(
        lambda t, mean=0.0, std=1.0, *, generator=None: t.normal_(mean, std, generator=generator)
    )
)
