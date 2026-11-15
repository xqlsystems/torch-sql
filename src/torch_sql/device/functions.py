"""PyTorch functions torch-sql lowers before PyTorch decomposes them.

Some PyTorch functions are decomposed in C++, above the dispatcher, into ops
that create tensors on the input's device: ``scaled_dot_product_attention``
builds its causal mask with ``at::ones(query.options())``. A torch-sql
tensor's device is ``meta``, so that mask would be a plain meta tensor,
which holds no data. The environment's function mode sees the Python call
first, and runs the functions here instead, which are written with ordinary
tensor ops that torch-sql records one by one.

They run inside the function mode's handler, where the mode is off, so
they create tensors through the environment (``env.tensor``), not with
``device="sql"``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
):
    """PyTorch's math reference for attention
    (``torch.nn.functional.scaled_dot_product_attention``'s documentation)."""
    if enable_gqa:
        raise NotImplementedError("scaled_dot_product_attention with enable_gqa")
    L, S = query.size(-2), key.size(-2)
    scale = 1.0 / math.sqrt(query.size(-1)) if scale is None else scale
    scores = query @ key.transpose(-2, -1) * scale
    if is_causal:
        if attn_mask is not None:
            raise RuntimeError(
                "_scaled_dot_product_attention: Explicit attn_mask should not be "
                "set when is_causal=True"
            )
        causal = query._env.tensor(torch.ones(L, S, dtype=torch.bool).tril())
        scores = scores.masked_fill(causal.logical_not(), float("-inf"))
    elif attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(attn_mask.logical_not(), float("-inf"))
        else:
            scores = scores + attn_mask
    weights = torch.softmax(scores, dim=-1)
    if dropout_p > 0.0:
        weights = F.dropout(weights, dropout_p, training=True)
    return weights @ value


def getitem(x, index):
    """``x[index]``. PyTorch turns a list in an index (``x[:, [-1]]``), or a
    CPU tensor, into an index tensor on ``x``'s device, in C++; torch-sql
    makes it a torch-sql tensor first."""

    def convert(i):
        if isinstance(i, torch.Tensor) and i.dtype in (torch.bool, torch.uint8) and i.dim() > 0:
            raise NotImplementedError(
                "x[mask], with a boolean mask, has a shape that depends on mask's values, which "
                "torch-sql can't plan; torch.where(mask, x, 0.0) or x.masked_fill(~mask, 0.0) "
                "keep x's shape"
            )
        if isinstance(i, list) or (
            isinstance(i, torch.Tensor) and i.device.type == "cpu" and i.dim() > 0
        ):
            return x._env.tensor(torch.as_tensor(i))
        return i

    index = tuple(convert(i) for i in index) if isinstance(index, tuple) else convert(index)
    return torch.Tensor.__getitem__(x, index)


def where(condition, input=None, other=None, **kwargs):
    """``torch.where(condition, x, 0.0)``. PyTorch wraps a number branch in a
    tensor on ``x``'s device, in C++, which would be a plain meta tensor;
    torch-sql makes it a torch-sql tensor first, in the dtype PyTorch
    gives it."""
    if input is None or other is None or kwargs:
        return torch.where(condition, *(a for a in (input, other) if a is not None), **kwargs)
    env = next(a._env for a in (condition, input, other) if hasattr(a, "_env"))
    branches = [input, other]
    for k, branch in enumerate(branches):
        if not isinstance(branch, torch.Tensor):
            peer = branches[1 - k]
            dtype = torch.result_type(
                peer if isinstance(peer, torch.Tensor) else torch.tensor(peer), branch
            )
            branches[k] = env.tensor(torch.tensor(branch, dtype=dtype))
    return torch.where(condition, *branches)


def tensor_where(self, condition, other):
    """``x.where(condition, other)``, which is ``torch.where(condition, x,
    other)``."""
    return where(condition, self, other)


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """PyTorch's C++ form (``Distance.cpp``), which clamps the norms in place
    under ``NoGradGuard``: the clamp changes their values, not their
    gradients. torch-sql refuses such a write (issue #18), so the clamp here
    is a stop-gradient difference, with the same value and gradient."""
    x1, x2 = torch.broadcast_tensors(x1, x2)
    norms = []
    for x in (x1, x2):
        norm = torch.linalg.vector_norm(x, 2, dim=dim, keepdim=True)
        norms.append(norm + (norm.clamp_min(eps) - norm).detach())
    return ((x1 / norms[0]) * (x2 / norms[1])).sum(dim)


#: The functions, by the PyTorch function they replace.
FUNCTIONS = {
    F.scaled_dot_product_attention: scaled_dot_product_attention,
    torch.Tensor.__getitem__: getitem,
    torch.where: where,
    torch.Tensor.where: tensor_where,
    F.cosine_similarity: cosine_similarity,
    torch.cosine_similarity: cosine_similarity,
}
