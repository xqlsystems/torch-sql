"""Helpers that build the common shapes of :class:`~torch_sql.graph.Select`."""

from __future__ import annotations

import torch

from .. import graph as g
from ..sql import cast_to


def scalar(value, dtype: torch.dtype) -> g.Lit:
    """A Python number as a literal of ``dtype``."""
    if dtype == torch.bool:
        return g.Lit(bool(value), dtype)
    if dtype == torch.int64:
        if isinstance(value, float) and not value.is_integer():
            raise TypeError(f"{value} isn't an integer")
        return g.Lit(int(value), dtype)
    return g.Lit(float(value), dtype)


def python_dtype(x) -> torch.dtype:
    if isinstance(x, bool):
        return torch.bool
    if isinstance(x, int):
        return torch.int64
    return torch.float64


def as_dtype(e: g.Expr, have: torch.dtype, want: torch.dtype) -> g.Expr:
    if have == want:
        return e
    if isinstance(e, g.Lit):
        return scalar(e.value, want)
    return cast_to(e, want)


def elementwise(shape, dtype, operands, fn, *, compute=None) -> g.Select:
    """``fn`` applied to ``operands`` position by position, with PyTorch's
    broadcasting (§8.4), as a node of ``shape`` and ``dtype``.

    ``operands`` are nodes and Python numbers. Each node is joined to the
    others on the index columns where it has the output's full size; where
    it has size 1, or lacks the dimension, it repeats. ``fn`` takes one
    expression per operand and returns the value. Each operand is first
    converted to ``compute``: one dtype for all (``dtype`` if not given), or
    a list with one per operand, in which None keeps the operand's own.
    """
    compute = compute or dtype
    if not isinstance(compute, (list, tuple)):
        compute = [compute] * len(operands)
    shape = tuple(shape)
    rank = len(shape)
    nodes: list[g.Node] = []
    for o in operands:
        if isinstance(o, g.Node) and all(o is not n for n in nodes):
            nodes.append(o)
    position = {n.id: k for k, n in enumerate(nodes)}
    index, on = [], []
    for d in range(rank):
        if shape[d] == 1:
            index.append(g.Lit(0, torch.int64))
            continue
        sources = []
        for n in nodes:
            k = d - (rank - n.rank)
            if k >= 0 and n.shape[k] == shape[d]:
                sources.append(g.Col(position[n.id], f"i{k}"))
            elif k >= 0 and n.shape[k] != 1:
                raise ValueError(f"can't broadcast {n.shape} to {shape}")
        if not sources:
            raise ValueError(f"no operand has dimension {d} of {shape}; expand it first")
        index.append(sources[0])
        on.extend((sources[0], s) for s in sources[1:])
    exprs = []
    for o, want in zip(operands, compute, strict=True):
        if isinstance(o, g.Node):
            exprs.append(as_dtype(g.Col(position[o.id], "v"), o.dtype, want or o.dtype))
        else:
            exprs.append(scalar(o, want or python_dtype(o)))
    return g.Select(shape, dtype, tuple(nodes), on=tuple(on), index=tuple(index), value=fn(*exprs))


def project(x: g.Node, shape, dtype, index, value) -> g.Select:
    """A projection of one node: ``index`` and ``value`` read its columns
    as input 0."""
    return g.Select(tuple(shape), dtype, (x,), index=tuple(index), value=value)


def col(name: str, input: int = 0) -> g.Col:
    return g.Col(input, name)


def dims(x: g.Node, input: int = 0) -> list[g.Col]:
    return [g.Col(input, c) for c in x.columns]


def normalize_dims(dim, rank: int) -> list[int]:
    """PyTorch's ``dim`` argument (an int, a list, or None for all) as a
    sorted list of non-negative dimensions."""
    if rank == 0:
        return []
    if dim is None:
        return list(range(rank))
    if isinstance(dim, int):
        dim = [dim]
    if len(dim) == 0:
        return list(range(rank))
    return sorted({d % rank for d in dim})
