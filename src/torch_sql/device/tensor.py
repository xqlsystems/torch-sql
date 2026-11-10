"""``torch_sql.Tensor``, a tensor whose value is a table (design §9.1).

It's a *wrapper subclass*: a ``torch.Tensor`` subclass that holds no data of
its own, so PyTorch sends every ATen op on it to its ``__torch_dispatch__``.
torchax, a PyTorch backend that runs ops as JAX, makes the same kind of
tensor, and as it does, torch-sql allocates them on PyTorch's ``meta``
device, which tracks shapes and dtypes but holds no data. A ``sql`` device
underneath would need PyTorch's private Python-backend setup (spike S1),
which the design decides against (D16). PyTorch's C++ code sees ``meta``;
the Python ``device`` property reads ``sql``.

A tensor holds a node of the recorded graph (:mod:`torch_sql.graph`) and the
environment that records it. Ops make new tensors with new nodes; nothing
runs until a value is read (:meth:`Tensor.numpy`, ``.item()``, ``.cpu()``,
``print``) or :func:`torch_sql.sync` is called.

**Aliases** (design §8.6). In PyTorch a view (``x[:2]``, ``x.view(2, 2)``)
and ``x.detach()`` share ``x``'s storage, so an in-place op on one shows
through the others. Here they share a :class:`Storage`, which holds the
current node of the tensor that owns it. Each alias keeps the *lineage* of
the tensor it was taken from (its *base*): how the base finds its node, and
the function that derives the alias's node from the base's. An in-place op
gives the storage a new node, and each alias derives its node again, from
its base's current node, the next time it's used. So a chain of aliases (a
view of an ``nn.Parameter``, which is itself an alias) shares one node per
value, and backward() finds the leaf's node wherever it's read.
DataFusion Accelerated's table handles work the same way (its design §5.2):
a query reads a handle's current buffer, not the buffer it was planned
against.
"""

from __future__ import annotations

import copy
import weakref

import torch

from .. import graph as g

#: The device a torch-sql tensor reports. Importing the device package names
#: PyTorch's spare device slot ``sql``, before this module is used.
_SQL = None


class Storage:
    """What a tensor and its aliases share: the owner's current node, and
    the table that holds its value once ``torch_sql.sync()`` has run it."""

    __slots__ = ("node", "table", "__weakref__")

    def __init__(self, node: g.Node):
        self.node, self.table = node, None


class Lineage:
    """How a tensor finds its node: the storage's, for the tensor that owns
    the storage, or for an alias, a function of its base's node.

    An alias keeps its base's lineage, never the base tensor. PyTorch's
    autograd saves a ``.detach()`` of an op's output (``exp``, ``tanh``) in
    the op's backward node, so an alias that held the output would close a
    cycle through autograd's C++ graph, which Python's garbage collector
    can't see and so never frees."""

    __slots__ = ("storage", "derive", "base", "cached", "same_values", "history")

    def __init__(self, storage: Storage, derive=None, base: Lineage | None = None, node=None):
        self.storage, self.derive, self.base = storage, derive, base
        #: Whether the tensor holds the same values as the owner, as
        #: ``.detach()`` does, so an in-place op on it writes the storage.
        self.same_values = base is None
        #: An alias's node, and the base's node it was derived from.
        self.cached = None if base is None else (node, base.node())
        #: Whether a tensor on the chain of bases had a history (a
        #: ``grad_fn``) when this alias was taken. Not the tensors
        #: themselves, as above, nor weak references to them, which
        #: PyTorch's ``swap_tensors`` refuses.
        self.history = False

    def node(self) -> g.Node:
        if self.base is None:
            return self.storage.node
        node, source = self.cached
        current = self.base.node()
        if current is not source:
            node = self.derive(current)
            self.cached = (node, current)
        return node

    def set(self, node: g.Node):
        if self.base is None:
            self.storage.node = node
        else:
            self.cached = (node, self.base.node())


class Tensor(torch.Tensor):
    _env: object
    _lineage: Lineage

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
        #: The node last returned as ``_node``, and the earlier ones still
        #: alive, by id: a leaf's uses may have read any of them, and
        #: backward() differentiates with respect to each.
        t._last, t._earlier = None, weakref.WeakValueDictionary()
        if alias_of is None:
            t._lineage = Lineage(Storage(node))
            env._pending[id(t._storage)] = t._storage
        else:
            base = alias_of._lineage
            t._lineage = Lineage(base.storage, derive, base, node)
            t._lineage.same_values = same_values and base.same_values
            # Only an alias with the owner's values can be written (§8.6).
            # A view never reads its base's grad_fn: autograd makes views
            # again while it rebuilds a view's grad_fn after a write, and
            # reading one then breaks the rebuild.
            t._lineage.history = t._lineage.same_values and (
                alias_of.grad_fn is not None or base.history
            )
        return t

    @property
    def _storage(self) -> Storage:
        return self._lineage.storage

    @property
    def _node(self) -> g.Node:
        """The node holding this tensor's current value."""
        node = self._lineage.node()
        if node is not self._last:
            if self._last is not None:
                self._earlier[self._last.id] = self._last
                # Derived again after a write: the table, if any, holds the
                # old value.
                self._table = None
            self._last = node
        return node

    @_node.setter
    def _node(self, node: g.Node):
        """A node with the same value, such as the table that holds it."""
        self._lineage.set(node)

    def _adopt(self, other: Tensor):
        """Share ``other``'s storage from now on, as ``p.data = other``
        makes ``p`` do in PyTorch."""
        self._lineage = other._lineage

    def _write(self, node: g.Node):
        """A new value, which every alias sees. Only a write's target
        (``Environment._target``) calls this, once the write's checks have
        passed."""
        self._storage.node = node
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
    def data(self):
        return torch.Tensor.data.__get__(self)

    @data.setter
    def data(self, value):
        """``p.data = t``: PyTorch swaps ``p``'s shape and dtype for ``t``'s
        (``set_data``); ``p`` takes ``t``'s storage as well.
        ``torch.nn.utils.vector_to_parameters`` assigns this way."""
        if not isinstance(value, Tensor) or value._env is not self._env:
            raise RuntimeError("p.data = t needs t on the same torch-sql environment as p")
        torch.Tensor.data.__set__(self, value)
        self._adopt(value)

    @property
    def device(self):
        """``sql``. PyTorch's C++ code sees ``meta``, where the tensor is
        allocated; Python code that makes a tensor on another's device
        (``torch.zeros(n, device=x.device)``, ``t.to(x.device)``) names
        ``sql``, so ``meta`` keeps meaning PyTorch's own meta device."""
        return _SQL

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
