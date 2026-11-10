"""Cut points: which values a program saves as tables (design §9.6).

DataFusion plans a CTE again at every place a query reads it, so a value
read twice in each of many layers multiplies the plan: two nanoGPT layers'
loss doesn't plan in 8 GB (xqlsystems/xarray-sql#267, spike S8). torch-sql
therefore chooses *cut points*, nodes whose values are saved as tables, so
that no query it hands an engine grows past a budget.

**Choosing them** (:func:`cut_points`). Walking the graph from its inputs,
each node's *inlined size* is the number of ``SELECT`` blocks the engine
would plan for it if every node it reads were written into it: one, plus
the inlined size of each input, once per read. A table counts once. When a
node's inlined size would pass the engine's budget, its largest inputs
become cut points, until it fits, so no query passes the budget unless a
node reads more tables than the budget allows. A node that feeds a norm,
which reads its input three times, is a cut point too (decision D15). Only
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
   gives the leaves and earlier cut points their share. A cut point that
   several segments read gets a cotangent from each, and their sum is saved
   as a table first.

Each segment is one ddx program, and each leaf's gradient is the sum of its
shares.
"""

from __future__ import annotations

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
    size, order, cuts = {}, {}, {}
    for k, node in enumerate(g.topological([root], read)):
        order[node.id] = k
        if isinstance(node, g.Table):
            size[node.id] = 1
            continue
        reads = defaultdict(int)  # an input's id -> how often the node reads it
        inputs = {}
        for i in node.inputs:
            i = read(i)
            reads[i.id] += 1
            inputs[i.id] = i
        s = 1 + sum(size[i] * n for i, n in reads.items())
        # Cutting the inputs that save the most first, so the node's own
        # query fits, rather than cutting the node after it's too big.
        for i in sorted(reads, key=lambda i: size[i] * reads[i], reverse=True):
            if s <= budget or size[i] == 1:
                break
            s -= (size[i] - 1) * reads[i]
            size[i] = 1
            cuts[i] = inputs[i]
        if node is not root and node in marked:
            cuts[node.id] = node
            s = 1
        size[node.id] = s
    # In the order they're computed, so each reads the ones before it.
    return sorted(cuts.values(), key=lambda n: order[n.id])


def materialize(env, root: g.Node, resolve: dict) -> g.Table:
    """Save ``root``'s value as a table, saving its cut points first. Each
    query reads the tables before it, through ``resolve``, which this
    extends, and which holds them until ``root`` is saved."""
    for node in cut_points(
        root, lambda n: resolve.get(n.id, n), env.engine.cut_budget, env.marked_cuts
    ):
        resolve[node.id] = _save(env, node, resolve)
    return _save(env, root, resolve)


def _save(env, node: g.Node, resolve: dict) -> g.Table:
    table = g.Table(node.shape, node.dtype)
    env.engine.materialize(table.name, sql.query(node, resolve, key=True))
    env._drop_with(table)
    return table


def backward(env, objective: g.Node, resolve: dict, wrt: list[g.Table]):
    """The gradient of ``objective``, a scalar, with respect to each table
    in ``wrt``. ``resolve`` maps the ids of nodes whose values are constants
    or leaves to the tables holding them.

    Returns the loss's value as a table, and a node for each ``wrt``
    table's gradient (None for a table the loss doesn't read)."""
    # Our own copy, which also holds the cut points' tables until the
    # gradients exist.
    resolve = dict(resolve)
    read = lambda n: resolve.get(n.id, n)  # noqa: E731
    cuts = cut_points(objective, read, env.engine.cut_budget, env.marked_cuts)
    cut_of = {}  # a cut point's table's id -> the cut point
    for node in cuts:
        table = _save(env, node, resolve)
        resolve[node.id] = table
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

    # SUM over the loss's one row: ddx can then show the loss is one row,
    # which it can't for a scalar cross joined with another (a detached
    # scalar). Each cut point's objective below is a SUM already.
    summed = g.Select((), F64, (objective,), value=g.Agg("SUM", g.Col(0, "v")), group=True)
    value = differentiate(summed, resolve)
    for node in reversed(cuts):
        if not cotangents[node.id]:
            continue
        parts = cotangents.pop(node.id)
        # Saved once, rather than summed inside the VJP's query, which would
        # then grow with the number of readers, past the budget (issue #23).
        cotangent = parts[0] if len(parts) == 1 else _save(env, total(parts), resolve)
        on = tuple((g.Col(0, c), g.Col(1, c)) for c in node.columns)
        vjp = g.Select(
            (),
            F64,
            (node, cotangent),
            on=on,
            value=g.Agg("SUM", g.Col(0, "v") * g.Col(1, "v")),
            group=True,
        )
        # The segment computes this cut point from the ones before it, so
        # it must not read the cut point's own table.
        table = resolve.pop(node.id)
        differentiate(vjp, resolve)
        resolve[node.id] = table
    grads = {}
    for t in wrt:
        if not shares[t.id]:
            grads[t.id] = None
        elif len(shares[t.id]) == 1:
            grads[t.id] = shares[t.id][0]
        else:  # saved once, rather than summed again by every reader
            grads[t.id] = _save(env, total(shares[t.id]), resolve)
    return value, grads


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
    env._drop_with(table)
    return table
