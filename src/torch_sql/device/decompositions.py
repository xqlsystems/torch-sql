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
    precision. A float32 variance is computed in float64 and rounded once,
    as PyTorch's CPU kernel accumulates it."""
    return _variance(x, dim, correction, keepdim).to(x.dtype)


def _variance(x, dim, correction, keepdim):
    """The variance, in float64 for a float32 ``x``."""
    if x.dtype == torch.float32:
        x = x.double()
    if dim is None or (not isinstance(dim, int) and len(dim) == 0):
        dims = list(range(x.dim()))
    else:
        dims = [dim] if isinstance(dim, int) else list(dim)
    # A 0-d tensor reduces along dimension 0 or -1, of size 1.
    n = math.prod(x.shape[d] for d in dims) if x.dim() else 1
    correction = 1 if correction is None else correction
    deviation = x - x.mean(dims, keepdim=True)
    return (deviation * deviation).sum(dims, keepdim=keepdim) / max(n - correction, 0)


def std(x, dim=None, *, correction=None, keepdim=False):
    """``aten.std.correction``: ``sqrt(var)``, with a gradient of 0 where
    the standard deviation is 0, as PyTorch's backward gives, rather than
    ``sqrt``'s 0/0. A constant feature in ``(x - x.mean(0)) / x.std(0)``
    would otherwise make every gradient NaN."""
    v = _variance(x, dim, correction, keepdim)
    zero = v == 0
    # In float64 for a float32 x, as PyTorch: the variance of [-1e20, 1e20]
    # passes float32's range, its square root doesn't.
    return v.masked_fill(zero, 1.0).sqrt().masked_fill(zero, 0.0).to(x.dtype)


def var_mean(x, dim=None, *, correction=None, keepdim=False):
    """``torch.var_mean``: the variance, and the mean it's taken about."""
    dims = list(range(x.dim())) if dim is None else dim
    return var(x, dim, correction=correction, keepdim=keepdim), x.mean(dims, keepdim=keepdim)


def std_mean(x, dim=None, *, correction=None, keepdim=False):
    """``torch.std_mean``."""
    dims = list(range(x.dim())) if dim is None else dim
    return std(x, dim, correction=correction, keepdim=keepdim), x.mean(dims, keepdim=keepdim)


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


def log_sigmoid_forward(x):
    """``aten.log_sigmoid_forward``, which ``F.logsigmoid`` and
    ``binary_cross_entropy_with_logits`` call: ``min(x, 0) - log1p(exp(-|x|))``,
    as PyTorch's decomposition, with both kinks taken on the same side of 0.
    ``minimum`` and ``abs`` each have a kink at 0 whose derivatives cancel,
    and ddx takes 0 for each, so the gradient at 0 came out 0 rather than
    0.5. ``exp`` reads at most 0, so it never overflows. The buffer is
    PyTorch's CPU one, ``exp(-|x|)``."""
    positive = x >= 0
    zero = torch.zeros_like(x)
    z = torch.exp(torch.where(positive, -x, x))
    return torch.where(positive, zero, x) - torch.log1p(z), z


def softplus(x, beta=1, threshold=20):
    """``aten.softplus``: ``x`` past the threshold, ``log1p(exp(x * beta)) /
    beta`` below it, as PyTorch's decomposition. ``exp`` reads at most the
    threshold, so the branch not taken never overflows: its gradient was
    inf / inf, which ``where`` turned into NaN, past exp's range. ``mish``
    calls it too."""
    scaled = x * beta
    above = scaled > threshold
    inner = torch.where(above, torch.full_like(scaled, threshold), scaled)
    return torch.where(above, x, torch.log1p(torch.exp(inner)) / beta)


#: The least ``p * (1 - p)`` that PyTorch's backward of
#: ``binary_cross_entropy`` divides by.
_BCE_EPSILON = 1e-12


def binary_cross_entropy(p, target, weight=None, reduction=1):
    """``aten.binary_cross_entropy``, with PyTorch's gradient rather than the
    derivative of its forward. PyTorch's backward is ``(p - y) / max(p * (1 -
    p), 1e-12)``, which flattens once ``p`` is within about 1e-12 of 0 or 1,
    where the forward's clamped logs keep a slope; ``sigmoid`` saturates
    there, so the two trainings would part. The forward is PyTorch's,
    computed from a detached ``p``, plus ``(p - p.detach())`` times the
    detached gradient: that is 0, and its gradient is PyTorch's. ``target``
    and ``weight`` get the forward's gradient, as in PyTorch."""
    q = p.detach()
    loss = _CORE[aten.binary_cross_entropy.default](q, target, weight, 0)
    y = target.detach()
    slope = (q - y) / ((1 - q) * q).clamp_min(_BCE_EPSILON)
    if weight is not None:
        slope = slope * weight.detach()
    loss = loss + (p - q) * slope
    if reduction == 1:
        return loss.mean()
    return loss.sum() if reduction == 2 else loss


#: The decompositions, by ATen op overload.
DECOMPOSITIONS = {
    aten.var.correction: var,
    aten.std.correction: std,
    aten.var_mean.correction: var_mean,
    aten.std_mean.correction: std_mean,
    aten.linalg_vector_norm.default: vector_norm,
    aten.log_sigmoid_forward.default: log_sigmoid_forward,
    aten.softplus.default: softplus,
    aten.binary_cross_entropy.default: binary_cross_entropy,
}
