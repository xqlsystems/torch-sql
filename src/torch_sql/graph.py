"""The recorded graph: torch-sql's small relational IR (design §9.2).

Every node is a relation that holds one tensor as a *long table* (§8.1): one
integer index column per dimension, named ``i0``, ``i1``, ..., and one value
column ``v``. A scalar (a tensor with no dimensions) has only ``v``.

Nodes form a DAG, since a value is often read more than once. There are four
kinds of node:

- :class:`Table`, a table registered on the engine: a parameter, an input,
  or a value saved by an earlier program;
- :class:`Values`, a small constant written into the SQL as a ``VALUES``
  list (§8.7);
- :class:`Select`, one ``SELECT`` block: a scan of one input, or an inner
  join of several on equalities, then an optional filter, then either a
  projection or a ``GROUP BY`` with aggregates;
- :class:`UnionAll`, the rows of several inputs together.

A :class:`Select` covers the relational primitives ddx differentiates
(projection, filter, join, aggregate) in the form SQL writes them, so the
SQL writer (:mod:`torch_sql.sql`) is a direct printer, and the IR stays
independent of the output format. Expressions are small immutable trees
(:class:`Expr`), which refer to their node's inputs by position.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import torch

#: The dtypes a value column may have, and their SQL types. float64 is the
#: default for verification (§8.2). float32 is planned; until it lands,
#: torch-sql refuses it rather than computing in a different precision.
SQL_TYPES = {
    torch.float64: "DOUBLE",
    torch.int64: "BIGINT",
    torch.bool: "BOOLEAN",
}

#: The prefix of every name torch-sql generates. A generated name must never
#: equal a registered table's, since in SQL a subquery's name hides a table
#: of the same name (§9.2). ddx reserves ``__ddx_`` the same way.
PREFIX = "_tsql_"

_ids = itertools.count(1)


def index_columns(rank: int) -> tuple[str, ...]:
    return tuple(f"i{k}" for k in range(rank))


def check_dtype(dtype: torch.dtype) -> None:
    if dtype not in SQL_TYPES:
        raise NotImplementedError(
            f"torch-sql doesn't support {dtype} yet; it supports "
            f"{', '.join(str(d) for d in SQL_TYPES)} (design §8.2)"
        )


# ----------------------------------------------------------------------------
# Expressions


class Expr:
    """An expression over the columns of a :class:`Select`'s inputs."""

    __slots__ = ()

    # Python operators build expressions, so lowerings read like the SQL.
    def __add__(self, o):
        return Call("+", (self, lit(o)))

    def __radd__(self, o):
        return Call("+", (lit(o), self))

    def __sub__(self, o):
        return Call("-", (self, lit(o)))

    def __rsub__(self, o):
        return Call("-", (lit(o), self))

    def __mul__(self, o):
        return Call("*", (self, lit(o)))

    def __rmul__(self, o):
        return Call("*", (lit(o), self))

    def __truediv__(self, o):
        return Call("/", (self, lit(o)))

    def __rtruediv__(self, o):
        return Call("/", (lit(o), self))


@dataclass(frozen=True, slots=True)
class Col(Expr):
    """Column ``name`` of the node's input number ``input``."""

    input: int
    name: str


@dataclass(frozen=True, slots=True)
class Lit(Expr):
    """A constant. Floats are written so that every engine reads them as
    DOUBLE, with the same bits."""

    value: Any
    dtype: torch.dtype


@dataclass(frozen=True, slots=True)
class Call(Expr):
    """A function or operator applied to arguments. ``fn`` is a name from
    :data:`torch_sql.sql.FUNCTIONS`."""

    fn: str
    args: tuple[Expr, ...]


@dataclass(frozen=True, slots=True)
class Agg(Expr):
    """An aggregate (``SUM``, ``AVG``, ``MAX``, ``MIN``, ``COUNT``) of an
    expression, in a :class:`Select` that groups."""

    fn: str
    arg: Expr


def lit(x) -> Expr:
    """``x`` as an expression: an expression stays one, and a Python number
    becomes a :class:`Lit` of the dtype PyTorch would give it."""
    if isinstance(x, Expr):
        return x
    if isinstance(x, bool):
        return Lit(x, torch.bool)
    if isinstance(x, int):
        return Lit(x, torch.int64)
    if isinstance(x, float):
        return Lit(x, torch.float64)
    raise TypeError(f"not an expression or a number: {x!r}")


def call(fn: str, *args) -> Call:
    return Call(fn, tuple(lit(a) for a in args))


def walk(e: Expr):
    """Every expression in ``e``, ``e`` first."""
    yield e
    if isinstance(e, Call):
        for a in e.args:
            yield from walk(a)
    elif isinstance(e, Agg):
        yield from walk(e.arg)


# ----------------------------------------------------------------------------
# Nodes


@dataclass(eq=False)
class Node:
    """A relation holding one tensor. Nodes compare by identity; ``id`` is
    stable for the life of the process, so facts and hints (§10) can refer
    to a node."""

    shape: tuple[int, ...]
    dtype: torch.dtype
    inputs: tuple[Node, ...] = ()
    #: Provenance (§10): the ATen op that made the node, and its arguments.
    #: Hints never change the meaning of the SQL.
    hint: dict = field(default_factory=dict, repr=False)
    id: int = field(default_factory=lambda: next(_ids), init=False)

    def __post_init__(self):
        self.shape = tuple(int(s) for s in self.shape)
        check_dtype(self.dtype)

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def columns(self) -> tuple[str, ...]:
        return index_columns(self.rank)

    @property
    def numel(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n


@dataclass(eq=False)
class Table(Node):
    """A table registered on the engine under ``name``. A table holding a
    scalar has a key column ``i`` (always 0) besides ``v``, since ddx needs a
    column that identifies a table's rows to tell their gradients apart."""

    name: str = ""

    def __post_init__(self):
        super().__post_init__()
        if not self.name:
            self.name = f"{PREFIX}t{self.id}"


@dataclass(eq=False)
class Values(Node):
    """A small constant (§8.7): ``rows`` holds ``(index..., value)`` tuples."""

    rows: tuple[tuple, ...] = ()


@dataclass(eq=False)
class Select(Node):
    """One ``SELECT`` block over ``inputs``.

    - ``on``: equalities ``(left, right)`` joining the inputs. Inputs that no
      equality joins to the ones before them are cross joined.
    - ``where``: a filter, applied after the join.
    - ``index``: the expression for each output index column.
    - ``value``: the expression for the value column.
    - ``group``: whether the block groups by its index expressions; its
      ``value`` then contains an :class:`Agg`. A grouping block with no index
      expressions aggregates every row into one.
    - ``stop_gradient``: no gradient flows through this node, as with
      ``x.detach()``. It doesn't change the SQL; before ddx differentiates a
      loss, torch-sql saves every such node as a table, which ddx treats as
      a constant (:mod:`torch_sql.gradients`).
    """

    on: tuple[tuple[Expr, Expr], ...] = ()
    where: Expr | None = None
    index: tuple[Expr, ...] = ()
    value: Expr | None = None
    group: bool = False
    stop_gradient: bool = False

    def __post_init__(self):
        super().__post_init__()
        if len(self.index) != len(self.shape):
            raise ValueError(f"{len(self.index)} index expressions for shape {self.shape}")
        if self.value is None:
            raise ValueError("a Select needs a value expression")
        for e in self.expressions():
            for sub in walk(e):
                if isinstance(sub, Col) and not 0 <= sub.input < len(self.inputs):
                    raise ValueError(f"{sub} refers to input {sub.input} of {len(self.inputs)}")
        aggregated = any(isinstance(sub, Agg) for sub in walk(self.value))
        if aggregated != self.group:
            raise ValueError("a grouping Select's value must aggregate, and only then")

    def expressions(self):
        for a, b in self.on:
            yield a
            yield b
        if self.where is not None:
            yield self.where
        yield from self.index
        yield self.value


@dataclass(eq=False)
class UnionAll(Node):
    """The rows of every input, which all have this node's columns."""

    def __post_init__(self):
        super().__post_init__()
        for n in self.inputs:
            if n.rank != self.rank or n.dtype != self.dtype:
                raise ValueError("a UNION ALL's inputs must have its rank and dtype")


def identity(x: Node, *, stop_gradient: bool = False) -> Select:
    """A projection of ``x`` that changes nothing."""
    return Select(
        x.shape,
        x.dtype,
        (x,),
        index=tuple(Col(0, c) for c in x.columns),
        value=Col(0, "v"),
        stop_gradient=stop_gradient,
    )


def topological(roots, resolve=lambda n: n) -> list[Node]:
    """Every node the ``roots`` depend on, each after its inputs. Iterative,
    since a deep model's graph is deeper than Python's recursion limit.
    ``resolve`` maps each node to the one to visit in its place (a table that
    already holds its value, say)."""
    order, seen = [], set()
    for root in roots:
        root = resolve(root)
        stack = [(root, False)]
        while stack:
            node, done = stack.pop()
            if done:
                order.append(node)
                continue
            if node.id in seen:
                continue
            seen.add(node.id)
            stack.append((node, True))
            inputs = [resolve(i) for i in node.inputs]
            stack.extend((i, False) for i in reversed(inputs) if i.id not in seen)
    return order
