"""Matrix products (design §9.4): a join on the contracted index, then
``SUM(a.v * b.v)`` grouped by the rest."""

from __future__ import annotations

import torch

from .. import graph as g
from . import op
from ._build import elementwise, scalar

aten = torch.ops.aten


def contraction(ctx, a, b, on, index) -> g.Node:
    out = ctx.out
    if any(a.shape[int(c.name[1:])] == 0 for c, _ in on):
        # A sum over nothing is 0, as in PyTorch; the join would have no rows.
        from .creation import filled

        return filled(ctx.env, out.shape, out.dtype, 0)
    return g.Select(
        tuple(out.shape),
        out.dtype,
        (a, b),
        on=tuple(on),
        index=tuple(index),
        value=g.Agg("SUM", g.Col(0, "v") * g.Col(1, "v")),
        group=True,
    )


@op(aten.mm.default)
def mm(ctx, a, b):
    # a(i, j) · b(j, k). The two inputs stay two aliases even when they are
    # one node, as in x @ x.
    return contraction(
        ctx, a, b, [(g.Col(0, "i1"), g.Col(1, "i0"))], [g.Col(0, "i0"), g.Col(1, "i1")]
    )


@op(aten.bmm.default)
def bmm(ctx, a, b):
    return contraction(
        ctx,
        a,
        b,
        [(g.Col(0, "i0"), g.Col(1, "i0")), (g.Col(0, "i2"), g.Col(1, "i1"))],
        [g.Col(0, "i0"), g.Col(0, "i1"), g.Col(1, "i2")],
    )


@op(aten.addmm.default)
def addmm(ctx, bias, a, b, *, beta=1, alpha=1):
    out = ctx.out
    product = mm(ctx, a, b)
    if beta == 0:
        # PyTorch ignores the bias, so a NaN or an infinity in it doesn't
        # reach the result.
        return elementwise(
            out.shape,
            out.dtype,
            [product],
            lambda p: p if alpha == 1 else p * scalar(alpha, out.dtype),
        )

    def fn(c, p):
        c = c if beta == 1 else c * scalar(beta, out.dtype)
        p = p if alpha == 1 else p * scalar(alpha, out.dtype)
        return c + p

    return elementwise(out.shape, out.dtype, [bias, product], fn)
