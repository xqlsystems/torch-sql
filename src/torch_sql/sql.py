"""Writes SQL from the recorded graph (design §9.2).

The SQL is portable: the same text runs on DataFusion and DuckDB and means
the same there (P1). Where the two engines read a construct differently,
the writer picks a spelling both read alike:

- a float literal is written with an exponent (``0.5e0``), which both read
  as DOUBLE; DuckDB reads ``0.5`` as a DECIMAL;
- infinities and NaN are read from a one-row table every engine registers,
  :data:`IEEE`, since ddx refuses to differentiate a query that holds a
  non-finite literal (xqlsystems/ddx#261). Outside a ``SELECT`` that a
  gradient could reach (in ``VALUES``), they're casts from strings;
- ``/`` only ever divides floats, since the engines divide integers
  differently.

:func:`query` writes the SQL for one node: each node it depends on becomes
a CTE (a named subquery, ``WITH name AS (...)``), except tables, which are
read by name.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from .graph import (
    PREFIX,
    SQL_TYPES,
    Agg,
    Call,
    Col,
    Expr,
    Lit,
    Node,
    Select,
    Table,
    UnionAll,
    Values,
    topological,
    walk,
)

#: Infix operators.
BINARY = {
    "+": "+",
    "-": "-",
    "*": "*",
    "/": "/",
    "=": "=",
    "<>": "<>",
    "<": "<",
    "<=": "<=",
    ">": ">",
    ">=": ">=",
    "and": "AND",
    "or": "OR",
}

#: Functions, by the name a :class:`~torch_sql.graph.Call` uses, and their SQL
#: name. Each is in both engines under the same name. ddx differentiates the
#: unary ones it lists in ``ddxdb.supported_functions()``, plus ``power``.
FUNCTIONS = {
    "exp": "exp",
    "ln": "ln",
    "sqrt": "sqrt",
    "tanh": "tanh",
    "abs": "abs",
    "sin": "sin",
    "cos": "cos",
    "power": "power",
    "greatest": "greatest",
    "least": "least",
    "isnan": "isnan",
}

AGGREGATES = {"SUM", "AVG", "MAX", "MIN", "COUNT"}

#: The one-row table of non-finite values, and its columns.
IEEE = PREFIX + "ieee"
IEEE_COLUMNS = {"nan": math.nan, "pinf": math.inf, "ninf": -math.inf}


def _nonfinite_column(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return "pinf" if value > 0 else "ninf"


def _nonfinite(e: Expr) -> bool:
    return isinstance(e, Lit) and isinstance(e.value, float) and not math.isfinite(e.value)


def cte_name(node: Node) -> str:
    return f"{PREFIX}n{node.id}"


def relation(node: Node) -> str:
    """The name a query reads ``node`` by."""
    return node.name if isinstance(node, Table) else cte_name(node)


def literal(value, dtype: torch.dtype) -> str:
    if dtype == torch.bool:
        return "TRUE" if value else "FALSE"
    if dtype == torch.int64:
        return str(int(value))
    value = float(value)
    if math.isnan(value):
        return "CAST('NaN' AS DOUBLE)"
    if math.isinf(value):
        return f"CAST('{'-' if value < 0 else ''}inf' AS DOUBLE)"
    s = repr(value)
    return s if "e" in s else f"{s}e0"


def expr(e: Expr, aliases: list[str], ieee: str | None = None, grouped: bool = False) -> str:
    """``e`` in SQL, reading input ``k`` under the alias ``aliases[k]``, and
    a non-finite constant from :data:`IEEE` under the alias ``ieee``. In a
    grouping ``SELECT`` (``grouped``), a constant read outside an aggregate
    is read through ``MAX``, which every row's group allows."""
    if isinstance(e, Col):
        return f"{aliases[e.input]}.{e.name}"
    if isinstance(e, Lit):
        if ieee is not None and _nonfinite(e):
            col = f"{ieee}.{_nonfinite_column(e.value)}"
            return f"MAX({col})" if grouped else col
        return literal(e.value, e.dtype)
    if isinstance(e, Agg):
        if e.fn not in AGGREGATES:
            raise ValueError(f"unknown aggregate {e.fn}")
        return f"{e.fn}({expr(e.arg, aliases, ieee, False)})"
    if isinstance(e, Call):
        args = [expr(a, aliases, ieee, grouped) for a in e.args]
        if e.fn in BINARY:
            a, b = args
            return f"({a} {BINARY[e.fn]} {b})"
        if e.fn == "not":
            return f"(NOT {args[0]})"
        if e.fn == "case":  # case(cond, then, else)
            return f"CASE WHEN {args[0]} THEN {args[1]} ELSE {args[2]} END"
        if e.fn.startswith("cast:"):
            return f"CAST({args[0]} AS {e.fn[5:]})"
        if e.fn == "stop_gradient":
            return f"ddx_stop_gradient({args[0]})"
        if e.fn in FUNCTIONS:
            return f"{FUNCTIONS[e.fn]}({', '.join(args)})"
        raise ValueError(f"unknown function {e.fn}")
    raise TypeError(f"not an expression: {e!r}")


def cast_to(e: Expr, dtype: torch.dtype) -> Call:
    return Call(f"cast:{SQL_TYPES[dtype]}", (e,))


def select(node: Select, resolve=lambda n: n) -> str:
    aliases = [f"t{k}" for k in range(len(node.inputs))]
    nonfinite = any(_nonfinite(sub) for e in node.expressions() for sub in walk(e))
    ieee = "tieee" if nonfinite else None
    # Attach each equality to the latest input it reads, so that every input
    # joins on the ones before it; an input no equality reaches is a cross
    # join.
    conditions: list[list[str]] = [[] for _ in node.inputs]
    for a, b in node.on:
        reads = [c.input for e in (a, b) for c in _cols(e)]
        if not reads:
            raise ValueError("a join condition must read a column")
        conditions[max(reads)].append(f"{expr(a, aliases, ieee)} = {expr(b, aliases, ieee)}")
    if node.inputs and conditions[0]:
        raise ValueError("a join condition reads only the first input; write it as a filter")
    parts = [f"{relation(resolve(node.inputs[0]))} {aliases[0]}"] if node.inputs else []
    for k in range(1, len(node.inputs)):
        source = f"{relation(resolve(node.inputs[k]))} {aliases[k]}"
        if conditions[k]:
            parts.append(f"JOIN {source} ON {' AND '.join(conditions[k])}")
        else:
            parts.append(f"CROSS JOIN {source}")
    if ieee is not None:
        parts.append(f"CROSS JOIN {IEEE} {ieee}" if parts else f"{IEEE} {ieee}")
    grouped = node.group
    cols = [
        f"{expr(e, aliases, ieee, grouped)} AS {c}"
        for e, c in zip(node.index, node.columns, strict=True)
    ]
    cols.append(f"{expr(node.value, aliases, ieee, grouped)} AS v")
    sql = f"SELECT {', '.join(cols)}"
    if parts:
        sql += f" FROM {' '.join(parts)}"
    if node.where is not None:
        sql += f" WHERE {expr(node.where, aliases, ieee)}"
    # A constant index (a dimension keepdim keeps) isn't grouped by: an
    # integer literal in GROUP BY means a column position.
    keys = [expr(e, aliases) for e in node.index if _cols(e)]
    if node.group and keys:
        sql += f" GROUP BY {', '.join(keys)}"
    return sql


def _cols(e: Expr):
    return [s for s in walk(e) if isinstance(s, Col)]


def values(node: Values) -> str:
    cols = (*node.columns, "v")
    if not node.rows:
        raise ValueError("an empty VALUES list")
    types = [torch.int64] * node.rank + [node.dtype]
    rows = []
    for row in node.rows:
        items = [literal(x, t) for x, t in zip(row, types, strict=True)]
        rows.append(f"({', '.join(items)})")
    # A derived table with column aliases reads the same on both engines.
    return f"SELECT * FROM (VALUES {', '.join(rows)}) AS t({', '.join(cols)})"


def union_all(node: UnionAll, resolve=lambda n: n) -> str:
    cols = ", ".join((*node.columns, "v"))
    return " UNION ALL ".join(f"SELECT {cols} FROM {relation(resolve(n))}" for n in node.inputs)


def body(node: Node, resolve=lambda n: n) -> str:
    """The ``SELECT`` that computes ``node`` from its inputs, by name.
    ``resolve`` maps each input to the node to read in its place."""
    if isinstance(node, Select):
        return select(node, resolve)
    if isinstance(node, Values):
        return values(node)
    if isinstance(node, UnionAll):
        return union_all(node, resolve)
    if isinstance(node, Table):
        return f"SELECT {', '.join((*node.columns, 'v'))} FROM {node.name}"
    raise TypeError(f"no SQL for {type(node).__name__}")


def query(root: Node, saved: Mapping[int, Table] | None = None, *, key: bool = False) -> str:
    """The SQL that computes ``root``: every node it depends on as a CTE,
    then ``SELECT`` of ``root``'s columns.

    ``saved`` maps node ids to tables that already hold those nodes' values,
    which the query reads instead of computing them again. With ``key``, a
    scalar's query also returns the key column ``i`` its table needs (see
    :class:`~torch_sql.graph.Table`).
    """
    saved = saved or {}

    def resolve(node: Node) -> Node:
        return saved.get(node.id, node)

    root = resolve(root)
    ctes = []
    for node in topological([root], resolve):
        if isinstance(node, Table):
            continue
        ctes.append(f"{cte_name(node)} AS ({body(node, resolve)})")
    cols = ["0 AS i"] if key and root.rank == 0 else []
    cols += [*root.columns, "v"]
    final = f"SELECT {', '.join(cols)} FROM {relation(root)}"
    return f"WITH {', '.join(ctes)} {final}" if ctes else final
