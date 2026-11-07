"""The environment: records ops on torch-sql tensors, and runs them on an
engine (design §9.1).

An :class:`Environment` owns an engine (:mod:`torch_sql.engines`) and the
tables torch-sql registers on it. While it's active (``with env:`` or
:meth:`Environment.enable`), a ``TorchFunctionMode`` sees every PyTorch
function call, and turns tensors created with ``device="sql"``, and tensors
moved with ``.to("sql")``, into torch-sql tensors. Ops on those tensors need
no mode: PyTorch sends them to :meth:`Environment.dispatch` through
``Tensor.__torch_dispatch__``, which adds a node to the recorded graph.

Dispatch runs each op on PyTorch's ``meta`` device first. That gives the
result's shape, dtype and strides exactly as PyTorch computes them, and
raises PyTorch's own error for a call PyTorch would refuse. The lowering
(:mod:`torch_sql.ops`) then only writes the relational plan.
"""

from __future__ import annotations

import weakref

import torch
from torch.overrides import TorchFunctionMode
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from .. import graph as g
from .. import sql
from ..boundary.layout import to_dense, to_table
from ..engines import Engine
from ..engines import engine as make_engine
from ..ops import LOWERINGS, Context, OperatorNotFound
from .tensor import Tensor

aten = torch.ops.aten

#: PyTorch's decompositions into core ATen ops (§9.3). A plain dict: the
#: table PyTorch returns (``CustomDecompTable``) answers ``.get()`` with None
#: even for ops it holds.
_table = torch._decomp.core_aten_decompositions()
DECOMPOSITIONS = {op: _table[op] for op in _table}

#: In-place ops that change their target's shape or strides.
IN_PLACE_VIEWS = {
    aten.squeeze_.default,
    aten.squeeze_.dim,
    aten.squeeze_.dims,
    aten.unsqueeze_.default,
    aten.t_.default,
    aten.transpose_.default,
    aten.as_strided_.default,
    aten.resize_.default,
}

#: In-place ops whose functional form isn't named after them.
FUNCTIONAL = {
    aten.normal_.default: aten.normal_functional.default,
}


def is_sql(device) -> bool:
    """Whether ``device`` names torch-sql's device. Inside the environment,
    ``meta`` does too: torch-sql tensors live on ``meta``, so code that
    creates a tensor on another tensor's device names ``meta``."""
    if device is None:
        return False
    return torch.device(device).type in ("sql", "meta")


class Environment:
    """Records ops on torch-sql tensors and runs them on one engine."""

    def __init__(self, engine: str | Engine = "datafusion"):
        self.engine = make_engine(engine) if isinstance(engine, str) else engine
        #: Tables that hold nodes' values, by node id, for nodes already read.
        self._saved: weakref.WeakValueDictionary[int, g.Table] = weakref.WeakValueDictionary()
        self._ranges: dict[int, g.Table] = {}
        self._index_tables: dict = {}
        self._mode = _FunctionMode(self)
        self._depth = 0

    def __repr__(self):
        return f"Environment({self.engine.name})"

    # -- Activation -------------------------------------------------------

    def __enter__(self):
        self._mode.__enter__()
        self._depth += 1
        return self

    def __exit__(self, *exc):
        self._depth -= 1
        return self._mode.__exit__(*exc)

    def enable(self):
        """Activate the environment until :meth:`disable`."""
        return self.__enter__()

    def disable(self):
        self.__exit__(None, None, None)

    # -- Tables -----------------------------------------------------------

    def upload(self, cpu: torch.Tensor) -> g.Table:
        """Register a CPU tensor's values as a table."""
        table = g.Table(tuple(cpu.shape), cpu.dtype)
        self.engine.register(table.name, to_table(cpu))
        self._drop_with(table)
        return table

    def _drop_with(self, table: g.Table):
        # The engine table lives as long as the node: when no tensor or graph
        # refers to it any more, it's dropped.
        weakref.finalize(table, _drop, self.engine, table.name)

    def tensor(self, cpu: torch.Tensor) -> Tensor:
        """A torch-sql tensor holding a copy of a CPU tensor."""
        if cpu.device.type != "cpu":
            raise ValueError(f"expected a CPU tensor, got one on {cpu.device}")
        return Tensor(self.upload(cpu.detach()), self, requires_grad=cpu.requires_grad)

    def arange(self, n: int) -> g.Table:
        """The table of ``torch.arange(n)``, registered once."""
        if n not in self._ranges:
            self._ranges[n] = self.upload(torch.arange(n))
        return self._ranges[n]

    def saved(self, node: g.Node) -> g.Node:
        """The table holding ``node``'s value if one does, else ``node``."""
        return self._saved.get(node.id, node)

    def query(self, node: g.Node, *, key=False) -> str:
        return sql.query(node, self._saved, key=key)

    def materialize(self, node: g.Node) -> g.Table:
        """Run ``node``'s query and save its rows as a table."""
        node = self.saved(node)
        if isinstance(node, g.Table):
            return node
        table = g.Table(node.shape, node.dtype)
        self.engine.materialize(table.name, self.query(node, key=True))
        self._drop_with(table)
        self._saved[node.id] = table
        return table

    def sync(self, *tensors: Tensor) -> None:
        """Run what each tensor's value depends on, and keep it as a table
        (§9.6). Later ops on these tensors read the tables."""
        for t in tensors:
            t._node = self.materialize(t._node)

    def read(self, t: Tensor) -> torch.Tensor:
        """``t``'s value as a CPU tensor."""
        self.sync(t)
        node = t._node
        rows = self.engine.query(f"SELECT {', '.join((*node.columns, 'v'))} FROM {node.name}")
        return to_dense(rows, node.shape, node.dtype)

    # -- Dispatch ---------------------------------------------------------

    def dispatch(self, func, types, args, kwargs):
        if "out" in kwargs:
            raise NotImplementedError(f"{func} with out=")
        special = _SPECIAL.get(func)
        if special is not None:
            return special(self, *args, **kwargs)
        if func._schema.is_mutable:
            return self._in_place(func, args, kwargs)
        lowering = LOWERINGS.get(func)
        if lowering is None:
            decomposition = DECOMPOSITIONS.get(func)
            if decomposition is not None:
                # It calls simpler ops, which come back here.
                return decomposition(*args, **kwargs)
            raise OperatorNotFound(
                f"torch-sql has no lowering for {func}, and PyTorch has no decomposition of it"
            )
        out = self._meta(func, args, kwargs)
        flat, spec = tree_flatten(out)
        if flat and all(isinstance(t, torch.Tensor) and t.numel() == 0 for t in flat):
            # A result with no elements is a relation with no rows, whatever
            # the op.
            nodes = tree_unflatten([self.empty(t.shape, t.dtype) for t in flat], spec)
        else:
            nodes = lowering(Context(func, out, self), *self._nodes(args), **self._nodes(kwargs))
        return self._wrap(func, out, nodes)

    def empty(self, shape, dtype) -> g.Node:
        """A tensor with no elements: no rows."""
        shape = tuple(shape)
        return g.Select(
            shape,
            dtype,
            (self.arange(1),),
            where=g.Lit(False, torch.bool),
            index=tuple(g.Lit(0, torch.int64) for _ in shape),
            value=g.Lit(0 if dtype != torch.bool else False, dtype)
            if dtype != torch.float64
            else g.Lit(0.0, dtype),
        )

    def _meta(self, func, args, kwargs):
        """PyTorch's result of ``func`` on the ``meta`` device."""

        def meta(x):
            if isinstance(x, Tensor):
                return torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device="meta")
            if isinstance(x, torch.device) and x.type == "sql":
                return torch.device("meta")
            return x

        margs, mkwargs = tree_map(meta, (args, kwargs))
        if "device" in mkwargs and is_sql(mkwargs["device"]):
            mkwargs["device"] = torch.device("meta")
        with torch._C.DisableTorchFunction():
            return func(*margs, **mkwargs)

    def _nodes(self, tree):
        def node(x):
            if isinstance(x, Tensor):
                return x._node
            if isinstance(x, torch.Tensor):
                if x.device.type == "cpu" and x.dim() == 0:
                    # A 0-d CPU tensor beside tensors on another device acts
                    # as a number, as in PyTorch.
                    return x.item()
                if x.device.type == "meta":
                    raise RuntimeError(
                        "a plain meta tensor reached torch-sql; create tensors with "
                        "device='sql' while the environment is active"
                    )
                raise RuntimeError(f"expected torch-sql tensors, got one on {x.device}")
            return x

        return tree_map(node, tree)

    def _wrap(self, func, out, nodes):
        flat, spec = tree_flatten(out)
        produced, _ = tree_flatten(nodes)
        if len(flat) != len(produced):
            raise AssertionError(
                f"{func}'s lowering returned {len(produced)} nodes for {len(flat)} results"
            )
        wrapped = []
        for meta, node in zip(flat, produced, strict=True):
            if not isinstance(meta, torch.Tensor):
                wrapped.append(meta)
                continue
            if tuple(meta.shape) != node.shape or meta.dtype != node.dtype:
                raise AssertionError(
                    f"{func}'s lowering made {node.shape} {node.dtype}; PyTorch makes "
                    f"{tuple(meta.shape)} {meta.dtype}"
                )
            node.hint.setdefault("aten", str(func))
            wrapped.append(Tensor(node, self, stride=meta.stride()))
        return tree_unflatten(wrapped, spec)

    def _in_place(self, func, args, kwargs):
        """An in-place op replaces its target's node (§8.6)."""
        target = args[0]
        if (
            func._schema.returns
            and func._schema.returns[0].alias_info is not None
            and func in IN_PLACE_VIEWS
        ):
            raise NotImplementedError(
                f"{func} changes a tensor's shape in place, which a torch-sql tensor can't "
                "(PyTorch's matmul does this for a 1-d argument)"
            )
        if not isinstance(target, Tensor):
            raise RuntimeError(f"{func} on a tensor that isn't torch-sql's")
        # PyTorch marks a view's wrapper itself (``_is_view``), above the
        # dispatch that reaches torch-sql. ``.detach()`` and ``nn.Parameter``
        # aren't views.
        if target._is_view():
            raise NotImplementedError(
                f"{func} writes through a view, which torch-sql doesn't support (design §8.6)"
            )
        functional = FUNCTIONAL.get(func)
        if functional is None:
            name = func._schema.name.split("::")[1]
            packet = getattr(aten, name[:-1], None) if name.endswith("_") else None
            functional = getattr(packet, func._overloadname, None) if packet else None
        if functional is None:
            raise OperatorNotFound(f"torch-sql has no lowering for {func}")
        # PyTorch's checks of the in-place call itself: shapes that don't
        # broadcast to the target, a cast it refuses.
        self._meta(func, args, kwargs)
        result = self.dispatch(functional, (), args, kwargs)
        self._assign(target, result._node)
        return target

    def _assign(self, target: Tensor, node: g.Node):
        if node.shape != target._node.shape:
            raise AssertionError(f"in-place result {node.shape} for {target._node.shape}")
        if node.dtype != target.dtype:
            from ..ops._build import as_dtype

            node = g.Select(
                node.shape,
                target.dtype,
                (node,),
                index=tuple(g.Col(0, c) for c in node.columns),
                value=as_dtype(g.Col(0, "v"), node.dtype, target.dtype),
            )
        target._node = node


def _drop(engine: Engine, name: str):
    try:
        engine.drop(name)
    except Exception:  # pragma: no cover - the engine may be gone at exit
        pass


# -- Ops the environment handles itself -----------------------------------


def _to_copy(env: Environment, x, **kwargs):
    device = kwargs.get("device")
    if device is not None and torch.device(device).type == "cpu":
        cpu = env.read(x)
        dtype = kwargs.get("dtype")
        return cpu.to(dtype) if dtype is not None else cpu
    out = env._meta(aten._to_copy.default, (x,), kwargs)
    return env._wrap(
        aten._to_copy.default,
        out,
        LOWERINGS[aten._to_copy.default](
            Context(aten._to_copy.default, out, env), x._node, **kwargs
        ),
    )


def _copy_(env: Environment, dst, src, non_blocking=False):
    if not isinstance(dst, Tensor):
        raise RuntimeError("copying a torch-sql tensor into a CPU tensor: use .cpu()")
    if dst._is_view():
        raise NotImplementedError("copy_ writes through a view (design §8.6)")
    if not isinstance(src, Tensor):
        src = env.tensor(src.detach().to(dst.dtype))
    if tuple(src.shape) != tuple(dst.shape):
        src = src.expand(dst.shape)
    env._assign(dst, src._node)
    return dst


def _scalar(env: Environment, x):
    return env.read(x).item()


def _is_nonzero(env: Environment, x):
    return bool(env.read(x).item())


_SPECIAL = {
    aten._to_copy.default: _to_copy,
    aten.copy_.default: _copy_,
    aten._local_scalar_dense.default: _scalar,
    aten.is_nonzero.default: _is_nonzero,
}


# -- The function mode ----------------------------------------------------


class _FunctionMode(TorchFunctionMode):
    """Makes tensors created on, or moved to, ``device="sql"`` torch-sql
    tensors."""

    def __init__(self, env: Environment):
        super().__init__()
        self.env = env

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.Tensor.to:
            moved = self._to(args, kwargs)
            if moved is not None:
                return moved
        elif is_sql(kwargs.get("device")) and not _has_sql(args, kwargs):
            # A factory: run it on the CPU, with PyTorch's CPU generator, so
            # the same seed gives the same values, and register the result.
            kwargs = {**kwargs, "device": "cpu"}
            requires_grad = kwargs.pop("requires_grad", False)
            cpu = func(*args, **kwargs)
            if isinstance(cpu, torch.Tensor):
                t = self.env.tensor(cpu)
                return t.requires_grad_() if requires_grad else t
            return cpu
        return func(*args, **kwargs)

    def _to(self, args, kwargs):
        source = args[0]
        if isinstance(source, Tensor):
            return None  # dispatch handles it: a cast, or a read to the CPU
        device, dtype, _, _ = torch._C._nn._parse_to(*args[1:], **kwargs)
        if device is None or torch.device(device).type != "sql":
            return None
        cpu = source if dtype is None else source.to(dtype)
        t = self.env.tensor(cpu.detach())
        return t.requires_grad_() if source.requires_grad else t


def _has_sql(args, kwargs) -> bool:
    return any(isinstance(a, Tensor) for a in torch.utils._pytree.tree_leaves((args, kwargs)))
