"""``torch_sql.Tensor``, a tensor whose value is a table (design §9.1).

It's a *wrapper subclass*: a ``torch.Tensor`` subclass that holds no data of
its own, so PyTorch sends every ATen op on it to its ``__torch_dispatch__``.
torchax, a PyTorch backend that runs ops as JAX, makes the same kind of
tensor, and as it does, torch-sql allocates them on PyTorch's ``meta``
device, which tracks shapes and dtypes but holds no data. So ``t.device`` reads ``meta``, not
``sql``; giving it a ``sql`` device would need PyTorch's private
Python-backend setup (spike S1, open question 11 of the design).

A tensor holds a node of the recorded graph (:mod:`torch_sql.graph`) and the
environment that records it. Ops make new tensors with new nodes; nothing
runs until a value is read (:meth:`Tensor.numpy`, ``.item()``, ``.cpu()``,
``print``) or :func:`torch_sql.sync` is called.
"""

from __future__ import annotations

import torch

from .. import graph as g


class Tensor(torch.Tensor):
    _node: g.Node
    _env: object

    # Every function goes straight to __torch_dispatch__, below autograd.
    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(cls, node: g.Node, env, *, stride=None, requires_grad=False):
        t = torch.Tensor._make_wrapper_subclass(
            cls,
            node.shape,
            strides=stride,
            dtype=node.dtype,
            device="meta",
            requires_grad=requires_grad,
        )
        t._node, t._env = node, env
        return t

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        env = None
        for a in torch.utils._pytree.tree_leaves((args, kwargs)):
            if isinstance(a, Tensor):
                if env is not None and a._env is not env:
                    raise RuntimeError("an op on tensors from two torch-sql environments")
                env = a._env
        return env.dispatch(func, types, args, kwargs or {})

    @property
    def is_meta(self):
        """False: the tensor is allocated on ``meta``, but it holds data, and
        PyTorch code that asks (``load_state_dict``, for one) should treat it
        as a tensor with values."""
        return False

    # Reading a value. The CPU tensor is computed by the engine.

    def numpy(self, *, force=False):
        return self.cpu().numpy()

    def tolist(self):
        return self.cpu().tolist()

    def item(self):
        return self.cpu().item()

    def __repr__(self, *, tensor_contents=None):
        body = repr(self.detach().cpu())
        return f"{body[:-1]}, device='sql')" if body.endswith(")") else body

    def __reduce_ex__(self, protocol):
        raise TypeError("a torch-sql tensor can't be pickled yet; move it to the CPU first")
