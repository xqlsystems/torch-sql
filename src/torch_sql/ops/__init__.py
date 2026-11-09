"""The op table (design §9.3): a lowering for each supported ATen op.

ATen is PyTorch's library of tensor operators; every PyTorch program
reaches it in the end. A *lowering* turns one ATen op on recorded tensors
into a node of the recorded graph (:mod:`torch_sql.graph`). It's a Python
function registered with :func:`op`::

    @op(aten.mm.default)
    def mm(ctx, a, b):
        ...  # a join on the contracted index, SUM(a.v * b.v), grouped by the rest

Its arguments are the op's, with each torch-sql tensor replaced by its
:class:`~torch_sql.graph.Node`, and a 0-d CPU tensor by its Python value, as
PyTorch treats one beside a tensor on another device. ``ctx``
(:class:`Context`) holds what PyTorch says the result is: the device runs
every op on PyTorch's ``meta`` device first, which computes shapes, dtypes
and errors without data. A lowering returns a node for each tensor the op
returns, of the shape and dtype ``ctx.out`` gives; the device checks.

An op with no lowering is decomposed into simpler ones with PyTorch's
``core_aten_decompositions()``. An op with neither raises
:class:`OperatorNotFound` (P3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: The lowerings, by ATen op overload.
LOWERINGS: dict = {}


class OperatorNotFound(NotImplementedError):
    """torch-sql can't lower this op, and PyTorch can't decompose it."""


@dataclass
class Context:
    """What a lowering needs besides the op's arguments."""

    #: The ATen op.
    func: Any
    #: PyTorch's result on the ``meta`` device: a tensor, or a tuple of them.
    out: Any
    #: The environment, for what lowerings register on the engine, such as
    #: index tables (:mod:`torch_sql.index_tables`).
    env: Any


def op(*funcs):
    """Register the decorated function as the lowering of each of ``funcs``."""

    def register(fn):
        for f in funcs:
            if f in LOWERINGS:
                raise ValueError(f"{f} has two lowerings")
            LOWERINGS[f] = fn
        return fn

    return register


# Import the families, which register their lowerings.
from . import creation, elementwise, linalg, reductions, shape  # noqa: E402,F401
