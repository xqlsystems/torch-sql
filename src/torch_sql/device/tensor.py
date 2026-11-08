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

**Aliases** (design §8.6). In PyTorch a view (``x[:2]``, ``x.view(2, 2)``)
and ``x.detach()`` share ``x``'s storage, so an in-place op on one shows
through the others. Here they share a :class:`Storage`, which holds the
current node of the tensor that owns it, and each alias keeps the function
that derives its own node from that one. An in-place op gives the storage a
new node, and each alias derives its node again the next time it's used.
DataFusion Accelerated's table handles work the same way (its design §5.2):
a query reads a handle's current buffer, not the buffer it was planned
against.
"""

from __future__ import annotations

import copy

import torch

from .. import graph as g


class Storage:
    """What a tensor and its aliases share: the owner's current node, a
    count of the in-place writes to it, and the table that holds its value
    once ``torch_sql.sync()`` has run it."""

    __slots__ = ("node", "version", "table", "__weakref__")

    def __init__(self, node: g.Node):
        self.node, self.version, self.table = node, 0, None


class Tensor(torch.Tensor):
    _storage: Storage
    _env: object
    #: How this tensor's node follows from its storage's: None for the
    #: storage's owner, else a function of the owner's node.
    _derive: object
    #: Whether this tensor holds the same values as the owner, as
    #: ``.detach()`` does, so an in-place op on it writes the storage.
    _same_values: bool

    # Every function goes straight to __torch_dispatch__, below autograd.
    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(
        cls,
        node: g.Node,
        env,
        *,
        stride=None,
        requires_grad=False,
        alias_of=None,
        derive=None,
        same_values=False,
    ):
        t = torch.Tensor._make_wrapper_subclass(
            cls,
            node.shape,
            strides=stride,
            dtype=node.dtype,
            device="meta",
            requires_grad=requires_grad,
        )
        t._env = env
        #: The table holding the value of a tensor that keeps its graph
        #: (see Environment.sync).
        t._table = None
        if alias_of is None:
            t._storage, t._derive, t._same_values = Storage(node), None, True
            env._pending[id(t._storage)] = t._storage
        else:
            t._storage, t._derive = alias_of._storage, derive
            t._same_values = same_values and alias_of._same_values
            t._cached = (node, t._storage.version)
        return t

    @property
    def _node(self) -> g.Node:
        """The node holding this tensor's current value."""
        if self._derive is None:
            return self._storage.node
        node, version = self._cached
        if version != self._storage.version:
            node = self._derive(self._storage.node)
            self._cached = (node, self._storage.version)
            self._table = None
        return node

    @_node.setter
    def _node(self, node: g.Node):
        """A node with the same value, such as the table that holds it."""
        if self._derive is None:
            self._storage.node = node
        else:
            self._cached = (node, self._cached[1])

    def _write(self, node: g.Node):
        """An in-place op's result: a new value, which every alias sees."""
        if not self._same_values:
            raise NotImplementedError(
                "an in-place op writes through a view, which torch-sql doesn't support "
                "(design §8.6)"
            )
        self._storage.node = node
        self._storage.version += 1
        self._table = None
        self._storage.table = None
        self._env._pending[id(self._storage)] = self._storage

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

    def backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None):
        """ddx computes the gradients, not PyTorch's autograd (§11.1); see
        :func:`torch_sql.gradients.backward`."""
        from ..gradients import backward

        backward(self, gradient, retain_graph, create_graph, inputs)

    def __deepcopy__(self, memo):
        """A tensor with storage of its own, in the same environment, as a
        deep copy of a CUDA tensor stays on its GPU. ``AveragedModel``
        deep-copies its model this way."""
        if id(self) in memo:
            return memo[id(self)]
        with torch.no_grad():
            new = self.clone()
        if isinstance(self, torch.nn.Parameter):
            new = torch.nn.Parameter(new, requires_grad=self.requires_grad)
        else:
            new.requires_grad_(self.requires_grad)
        if self.grad is not None:
            new.grad = copy.deepcopy(self.grad, memo)
        memo[id(self)] = new
        return new

    def __reduce_ex__(self, protocol):
        raise TypeError("a torch-sql tensor can't be pickled yet; move it to the CPU first")
