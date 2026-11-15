"""Decompositions of ATen ops that PyTorch doesn't decompose, written with
ops torch-sql lowers. The environment runs them as it runs PyTorch's own
(``core_aten_decompositions()``, §9.3): their ops come back to it one by
one.
"""

from __future__ import annotations

import math

import torch

aten = torch.ops.aten


def var(x, dim=None, *, correction=None, keepdim=False):
    """``aten.var.correction``, which ``torch.var`` and ``torch.std`` call:
    the mean of the squared distances from the mean, divided by ``n -
    correction`` rather than ``n``. Two passes, as ``layer_norm``'s
    variance is (§9.4), rather than ``AVG(x²) - AVG(x)²``, which loses
    precision."""
    if dim is None or (not isinstance(dim, int) and len(dim) == 0):
        dims = list(range(x.dim()))
    else:
        dims = [dim] if isinstance(dim, int) else list(dim)
    n = math.prod(x.shape[d] for d in dims)
    correction = 1 if correction is None else correction
    deviation = x - x.mean(dims, keepdim=True)
    return (deviation * deviation).sum(dims, keepdim=keepdim) / max(n - correction, 0)


#: The decompositions, by ATen op overload.
DECOMPOSITIONS = {
    aten.var.correction: var,
}
