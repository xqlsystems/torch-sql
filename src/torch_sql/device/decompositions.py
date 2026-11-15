"""Decompositions of ATen ops that PyTorch doesn't decompose, written with
ops torch-sql lowers. The environment runs them as it runs PyTorch's own
(``core_aten_decompositions()``, §9.3): their ops come back to it one by
one.
"""

from __future__ import annotations

import math

import torch

aten = torch.ops.aten

_CORE = torch._decomp.core_aten_decompositions()


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
    # A 0-d tensor reduces along dimension 0 or -1, of size 1.
    n = math.prod(x.shape[d] for d in dims) if x.dim() else 1
    correction = 1 if correction is None else correction
    deviation = x - x.mean(dims, keepdim=True)
    return (deviation * deviation).sum(dims, keepdim=keepdim) / max(n - correction, 0)


def vector_norm(x, ord=2, dim=None, keepdim=False, *, dtype=None):
    """``aten.linalg_vector_norm``. For the 2-norm, ``sqrt`` of the sum of
    squares with a gradient of 0 where the norm is 0, as PyTorch's backward
    gives, rather than ``sqrt``'s 0/0: ``F.normalize`` and
    ``cosine_similarity`` of an all-zero row, such as padding, read it. The
    other norms are PyTorch's decomposition."""
    if ord != 2 or not x.is_floating_point():
        return _CORE[aten.linalg_vector_norm.default](x, ord, dim, keepdim, dtype=dtype)
    if dtype is not None:
        x = x.to(dtype)
    squares = (x * x).sum(dim if dim is not None else list(range(x.dim())), keepdim=keepdim)
    zero = squares == 0
    # sqrt only of what's positive, so neither branch's gradient is 0/0.
    return squares.masked_fill(zero, 1.0).sqrt().masked_fill(zero, 0.0)


#: The decompositions, by ATen op overload.
DECOMPOSITIONS = {
    aten.var.correction: var,
    aten.linalg_vector_norm.default: vector_norm,
}
