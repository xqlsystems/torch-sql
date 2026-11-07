"""Cut points: which values a program saves as tables (design §9.6).

DataFusion plans a CTE again at every place a query reads it, so a value
read twice in each of many layers multiplies the plan: two nanoGPT layers'
loss doesn't plan in 8 GB (xqlsystems/xarray-sql#267, spike S8). torch-sql
therefore chooses *cut points*, nodes whose values are saved as tables, so
that no query it hands an engine grows past a budget.

**Choosing them** (:func:`cut_points`). Walking the graph from its inputs,
each node's *inlined size* is the number of ``SELECT`` blocks the engine
would plan for it if every node it reads were written into it: one, plus
the inlined size of each input, once per read. A table counts once. A node
is a cut point when its inlined size passes the engine's budget, or when it
feeds a norm, which reads its input three times (decision D15). Only
DataFusion has a budget; DuckDB plans a CTE read many times once, so it
gets no cut points and every query stays whole.

**Differentiating across them** (:func:`backward`). A loss computed from
cut points is a chain of *segments*: each cut point's value is computed from
earlier cut points and leaves, and the loss from the last ones. The chain
rule runs over the segments in reverse, as autograd runs over ops:

1. ddx differentiates the loss's own segment with respect to the leaves and
   the cut points it reads, which gives each cut point a *cotangent* (the
   gradient of the loss with respect to that table).
2. For each cut point, latest first, ddx differentiates
   ``SUM(value * cotangent)`` over its segment, with the cotangent held
   constant. That's the vector-Jacobian product of the segment, and it
   gives the leaves and earlier cut points their share.

Each segment is one ddx program, and each leaf's gradient is the sum of its
shares.
"""

from __future__ import annotations

import weakref
from collections import defaultdict

import torch

from . import graph as g
from . import sql

F64 = torch.float64


def cut_points(root: g.Node, read, budget: int | None, marked) -> list[g.Node]:
    """The nodes of ``root``'s graph to save as tables, in the order they're
    computed. ``read`` maps a node to the table that already holds it, if
    one does; ``marked`` holds nodes to cut whatever their size (the inputs
    of norms)."""
    if budget is None:
        return []
    size, cuts = {}, []
    for node in g.topological([root], read):
        if isinstance(node, g.Table):
            size[node.id] = 1
            continue
        s = 1 + sum(size[read(i).id] for i in node.inputs)
        if node is not root and (node in marked or s > budget):
            cuts.append(node)
            s = 1
        size[node.id] = s
    return cuts


def materialize(env, root: g.Node, resolve: dict) -> g.Table:
    """Save ``root``'s value as a table, saving its cut points first. Each
    query reads the tables before it, through ``resolve``, which this
    extends."""
    held = []
    for node in cut_points(
        root, lambda n: resolve.get(n.id, n), env.engine.cut_budget, env.marked_cuts
    ):
        table = _save(env, node, resolve)
        resolve[node.id] = table
        held.append(table)
    return _save(env, root, resolve)


def _save(env, node: g.Node, resolve: dict) -> g.Table:
    table = g.Table(node.shape, node.dtype)
    env.engine.materialize(table.name, sql.query(node, resolve, key=True))
    _adopt(env, table)
    return table


def backward(env, objective: g.Node, resolve: dict, wrt: list[g.Table]):
    """The gradient of ``objective``, a scalar, with respect to each table
    in ``wrt``. ``resolve`` maps the ids of nodes whose values are constants
    or leaves to the tables holding them.

    Returns the loss's value as a table, and a node for each ``wrt``
    table's gradient (None for a table the loss doesn't read)."""
    resolve = dict(resolve)
    read = lambda n: resolve.get(n.id, n)  # noqa: E731
    cuts = [c for c in cut_points(objective, read, env.engine.cut_budget, env.marked_cuts)]
    held = []  # the cut points' tables, kept until the gradients exist
    cut_of = {}  # a cut point's table's id -> the cut point
    for node in cuts:
        table = _save(env, node, resolve)
        resolve[node.id] = table
        held.append(table)
        cut_of[table.id] = node
    leaf_ids = {t.id for t in wrt}
    shares = defaultdict(list)  # a wrt table's id -> its gradient tables
    cotangents = defaultdict(list)  # a cut point's id -> its cotangent tables

    def differentiate(node, segment_resolve):
        """ddx's gradient of ``node`` over its segment, shared out."""
        reads = [
            n
            for n in g.topological([node], lambda n: segment_resolve.get(n.id, n))
            if isinstance(n, g.Table)
        ]
        targets = [t for t in reads if t.id in leaf_ids or (t.id in cut_of and t.dtype == F64)]
        if not targets:
            return None
        out = env.engine.grad(sql.query(node, segment_resolve), [f"{t.name}.v" for t in targets])
        for t in targets:
            share = _adopt(env, g.Table(t.shape, t.dtype, name=out.gradients[t.name]))
            if t.id in leaf_ids:
                shares[t.id].append(share)
            else:
                cotangents[cut_of[t.id].id].append(share)
        return _adopt(env, g.Table((), F64, name=out.value))

    value = differentiate(objective, resolve)
    for node in reversed(cuts):
        if not cotangents[node.id]:
            continue
        cotangent = total(cotangents.pop(node.id))
        on = tuple((g.Col(0, c), g.Col(1, c)) for c in node.columns)
        vjp = g.Select(
            (),
            F64,
            (node, cotangent),
            on=on,
            value=g.Agg("SUM", g.Col(0, "v") * g.Col(1, "v")),
            group=True,
        )
        segment = {k: v for k, v in resolve.items() if k != node.id}
        differentiate(vjp, segment)
    del held
    return value, {t.id: total(shares[t.id]) if shares[t.id] else None for t in wrt}


def total(tables: list[g.Node]) -> g.Node:
    """The sum of tensors of one shape, as one node."""
    if len(tables) == 1:
        return tables[0]
    first = tables[0]
    union = g.UnionAll(first.shape, first.dtype, tuple(tables))
    return g.Select(
        first.shape,
        first.dtype,
        (union,),
        index=tuple(g.Col(0, c) for c in first.columns),
        value=g.Agg("SUM", g.Col(0, "v")),
        group=True,
    )


def _adopt(env, table: g.Table) -> g.Table:
    """The engine table lives as long as the node."""
    weakref.finalize(table, _drop, env.engine, table.name)
    return table


def _drop(engine, name):
    try:
        engine.drop(name)
    except Exception:  # pragma: no cover - the engine may be gone at exit
        pass
