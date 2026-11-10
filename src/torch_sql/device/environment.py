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
from collections import ChainMap

import pyarrow as pa
import torch
from torch.optim.optimizer import register_optimizer_step_post_hook
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from .. import graph as g
from .. import sql
from ..boundary.layout import to_dense, to_table
from ..engines import Engine
from ..engines import engine as make_engine
from ..ops import LOWERINGS, Context, OperatorNotFound
from .functions import FUNCTIONS
from .tensor import Storage, Tensor

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

#: View ops whose result holds the same values as their input, so that an
#: in-place op on the result writes the input's storage, as
#: ``p.detach().lerp_(q, w)`` does in PyTorch's EMA of weights.
SAME_VALUES = {aten.detach.default, aten.alias.default, aten.lift_fresh.default}

#: In-place ops whose functional form isn't named after them.
FUNCTIONAL = {
    aten.normal_.default: aten.normal_functional.default,
}


def is_sql(device) -> bool:
    """Whether ``device`` names torch-sql's device."""
    if device is None:
        return False
    return torch.device(device).type == "sql"


class Environment:
    """Records ops on torch-sql tensors and runs them on one engine."""

    def __init__(self, engine: str | Engine = "datafusion"):
        self.engine = make_engine(engine) if isinstance(engine, str) else engine
        #: Tables that hold nodes' values, by node id, for nodes already read.
        self._saved: weakref.WeakValueDictionary[int, g.Table] = weakref.WeakValueDictionary()
        self._ranges: dict[int, g.Table] = {}
        self._points: dict[int, g.Table] = {}
        self._index_tables: dict = {}
        #: Nodes to save as tables whenever a program reads them: the inputs
        #: of norms (decision D15). See torch_sql.program.
        self.marked_cuts: weakref.WeakSet[g.Node] = weakref.WeakSet()
        #: The storage of each tensor created or written since the last
        #: sync(), by id. Not the tensors: PyTorch can't swap a tensor that
        #: has a weak reference, as module.to() does.
        self._pending: weakref.WeakValueDictionary[int, Storage] = weakref.WeakValueDictionary()
        self._mode = _FunctionMode(self)
        self._depth = 0

    def __repr__(self):
        return f"Environment({self.engine.name})"

    # -- Activation -------------------------------------------------------

    @staticmethod
    def active() -> Environment | None:
        """The innermost active environment, if any."""
        return _active[-1] if _active else None

    def __enter__(self):
        self._mode.__enter__()
        _active.append(self)
        if self._depth == 0:
            self._hook = register_optimizer_step_post_hook(self._after_step)
            # module.to("sql") must keep each Parameter object, as it does
            # for CUDA: nanoGPT ties lm_head.weight to wte.weight, and a new
            # Parameter per module would untie them. For a tensor subclass,
            # PyTorch keeps the object only when it swaps tensors.
            self._swap = torch.__future__.get_swap_module_params_on_conversion()
            torch.__future__.set_swap_module_params_on_conversion(True)
        self._depth += 1
        return self

    def __exit__(self, *exc):
        if not _active or _active[-1] is not self:
            # Each environment's modes stack on PyTorch's, so they can only
            # come off innermost first.
            raise RuntimeError(
                "an environment exited while an environment entered after it was still "
                "active; exit environments innermost first"
            )
        _active.pop()
        self._depth -= 1
        if self._depth == 0:
            self._hook.remove()
            torch.__future__.set_swap_module_params_on_conversion(self._swap)
        return self._mode.__exit__(*exc)

    def _after_step(self, optimizer, args, kwargs):
        """An optimizer step ends a training step (P2): save every parameter
        and every piece of optimizer state as a table, so that the next
        step's plans start from tables rather than from a graph that grows
        with every step. State goes first, since a parameter's update reads
        it."""
        state = [
            v
            for s in optimizer.state.values()
            for v in s.values()
            if isinstance(v, Tensor) and v._env is self
        ]
        params = [
            p
            for group in optimizer.param_groups
            for p in group["params"]
            if isinstance(p, Tensor) and p._env is self
        ]
        self.sync(*state, *params)

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
        """A table of the indices ``0`` to ``n - 1``, in its column ``i0``,
        registered once. It has no value column: a join with it (``expand``)
        reads only the index, and on DuckDB ddx mistakes an unread ``v``
        across a cross join for the other input's (xqlsystems/ddx#266)."""
        if n not in self._ranges:
            table = g.Table((n,), torch.int64)
            self.engine.register(table.name, pa.table({"i0": pa.array(range(n), pa.int64())}))
            self._drop_with(table)
            self._ranges[n] = table
        return self._ranges[n]

    def point(self, k: int) -> g.Table:
        """A one-row table whose value is ``k``, registered once: a join
        with it selects index ``k`` (``select``)."""
        if k not in self._points:
            self._points[k] = self.upload(torch.tensor([k]))
        return self._points[k]

    def saved(self, node: g.Node) -> g.Node:
        """The table holding ``node``'s value if one does, else ``node``."""
        return self._saved.get(node.id, node)

    def query(self, node: g.Node, *, key=False) -> str:
        return sql.query(node, self._saved, key=key)

    def materialize(self, node: g.Node) -> g.Table:
        """Run ``node``'s query and save its rows as a table."""
        from .. import program

        node = self.saved(node)
        if isinstance(node, g.Table):
            return node
        # A view of the saved tables that keeps the cut points' tables
        # without copying the rest.
        table = program.materialize(self, node, ChainMap({}, self._saved))
        self._saved[node.id] = table
        return table

    def sync(self, *tensors: Tensor) -> None:
        """Run what each tensor's value depends on, and keep it as a table
        (§9.6). Later queries read the table. With no tensors, every tensor
        still alive that was created or written since the last such call
        (§7.1).

        A leaf (a tensor with no ``grad_fn``: a parameter, optimizer state,
        an input) then *is* its table: its history is no one's business.
        Any other tensor keeps its graph as well, since a later
        ``backward()`` must differentiate through it."""
        if not tensors:
            # Each value's table is saved under its node, so later queries
            # read it, and kept alive by the storage.
            for storage in tuple(self._pending.values()):
                storage.table = self.materialize(storage.node)
            self._pending.clear()
            return
        for t in tensors:
            table = self.materialize(t._node)
            if t.grad_fn is None:
                t._node = table
            else:
                t._table = table

    def table_of(self, t: Tensor) -> g.Table:
        """The table holding ``t``'s value, once synced."""
        return t._node if isinstance(t._node, g.Table) else t._table

    def read(self, t: Tensor) -> torch.Tensor:
        """``t``'s value as a CPU tensor."""
        self.sync(t)
        node = self.table_of(t)
        rows = self.engine.query(f"SELECT {', '.join((*node.columns, 'v'))} FROM {node.name}")
        return to_dense(rows, node.shape, node.dtype)

    # -- Dispatch ---------------------------------------------------------

    def dispatch(self, func, types, args, kwargs):
        if "out" in kwargs:
            return self._out(func, args, kwargs)
        special = _SPECIAL.get(func)
        if special is not None:
            return special(self, *args, **kwargs)
        if func._schema.name.startswith("aten::_foreach_"):
            return _foreach(self, func, args, kwargs)
        if func._schema.is_mutable:
            return self._in_place(func, args, kwargs)
        lowering = LOWERINGS.get(func)
        if lowering is None:
            decomposition = DECOMPOSITIONS.get(func)
            if decomposition is not None:
                # It calls simpler ops, which come back here. Some create
                # tensors on the input's device, sql to Python code and meta
                # to C++; _Factories makes those torch-sql tensors.
                with _Factories(self):
                    return decomposition(*args, **kwargs)
            raise OperatorNotFound(
                f"torch-sql has no lowering for {func}, and PyTorch has no decomposition of it"
            )
        out = self._meta(func, args, kwargs)
        flat, spec = tree_flatten(out)
        for t in flat:
            if isinstance(t, torch.Tensor):
                # A result PyTorch gives in a dtype torch-sql lacks, such as
                # int / int in float32, is refused before it's lowered.
                g.check_dtype(t.dtype)
        if flat and all(isinstance(t, torch.Tensor) and t.numel() == 0 for t in flat):
            # A result with no elements is a relation with no rows, whatever
            # the op.
            nodes = tree_unflatten([self.empty(t.shape, t.dtype) for t in flat], spec)
            return self._wrap(func, out, nodes)
        stop = not torch.is_grad_enabled() and _requires_grad(args, kwargs)
        nodes = lowering(Context(func, out, self), *self._nodes(args), **self._nodes(kwargs))
        aliasing = self._aliasing(func, out, lowering, args, kwargs, stop)
        return self._wrap(func, out, _stopped(nodes) if stop else nodes, aliasing)

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
                        "a plain meta tensor reached torch-sql. Create tensors with "
                        "device='sql' while the environment is active. If this came from "
                        "inside a PyTorch function, PyTorch decomposed it in C++ into ops that "
                        "create tensors, and torch-sql needs to lower that function "
                        "(torch_sql.device.functions)"
                    )
                raise RuntimeError(f"expected torch-sql tensors, got one on {x.device}")
            return x

        return tree_map(node, tree)

    def _aliasing(self, func, out, lowering, args, kwargs, stop=False):
        """For a view op (one whose result aliases its input, §8.6): the
        input, and a function that lowers the op again from the input's
        current node, giving every result's node."""
        returns = func._schema.returns
        if (
            not returns
            or returns[0].alias_info is None
            or not args
            or not isinstance(args[0], Tensor)
        ):
            return None
        rest, kw = self._nodes(args[1:]), self._nodes(kwargs)

        def relower(node):
            nodes = lowering(Context(func, out, self), node, *rest, **kw)
            return tree_flatten(_stopped(nodes) if stop else nodes)[0]

        return args[0], _once_per_node(relower)

    def _wrap(self, func, out, nodes, aliasing=None):
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
            if aliasing is None:
                wrapped.append(Tensor(node, self, stride=meta.stride()))
                continue
            base, relower = aliasing
            wrapped.append(
                Tensor(
                    node,
                    self,
                    stride=meta.stride(),
                    alias_of=base,
                    derive=_result(relower, len(wrapped)),
                    same_values=func in SAME_VALUES,
                )
            )
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
        target = self._target(func, target, args, kwargs)
        functional = FUNCTIONAL.get(func)
        if functional is None:
            name = func._schema.name.split("::")[1]
            packet = getattr(aten, name[:-1], None) if name.endswith("_") else None
            functional = getattr(packet, func._overloadname, None) if packet else None
        if functional is None:
            raise OperatorNotFound(f"torch-sql has no lowering for {func}")
        result = self.dispatch(functional, (), args, kwargs)
        return target.write(result._node)

    def _out(self, func, args, kwargs):
        """An op with ``out=``, such as ``torch.maximum(a, b, out=a)`` in
        ``Adam(amsgrad=True)``: the functional op, assigned to ``out`` as an
        in-place op assigns its result (§8.6)."""
        kwargs = dict(kwargs)
        target = kwargs.pop("out")
        functional = _functional_of(func)
        if functional is None or not isinstance(target, Tensor):
            raise NotImplementedError(f"{func} with out=")
        out = self._meta(functional, args, kwargs)
        if not isinstance(out, torch.Tensor) or out.shape != target.shape:
            # PyTorch resizes out=; torch-sql's tensors keep their shape.
            raise NotImplementedError(f"{func} with an out= of another shape")
        target = self._target(func, target, args, {**kwargs, "out": target})
        result = self.dispatch(functional, (), args, kwargs)
        return target.write(result._node)

    def _target(self, func, tensor: Tensor, args=None, kwargs=None, *, reads=None) -> _Target:
        """``tensor``, as the target of a write by ``func``, once every check
        of the write has passed (§8.6). Each way of writing a tensor (an
        in-place op, ``out=``, ``copy_``) makes its target here before it
        computes anything, and writes only through the target.

        With ``args``, PyTorch's own checks of the call run as well: shapes
        that don't broadcast to the target, a cast it refuses. ``reads``
        holds the write's other inputs, ``(args, kwargs)`` by default."""
        # PyTorch marks a view's wrapper itself (``_is_view``), above the
        # dispatch that reaches torch-sql. ``.detach()`` and ``nn.Parameter``
        # aren't views, and hold their owner's values.
        if tensor._is_view() or not tensor._lineage.same_values:
            raise NotImplementedError(
                f"{func} writes through a view, which torch-sql doesn't support (design §8.6)"
            )
        if _untracked(tensor):
            raise NotImplementedError(
                "an in-place op that autograd doesn't record (under torch.no_grad(), or "
                "through .detach() or .data) on a tensor with a grad_fn: PyTorch keeps the "
                "tensor's gradient, which torch-sql can't yet (design §8.6, issue #18)"
            )
        if args is not None:
            self._meta(func, args, kwargs or {})
        target = _Target(tensor)
        target.reads(*((args, kwargs) if reads is None else reads))
        return target


def _functional_of(func):
    """The overload of ``func`` without ``out=``: ``maximum.out`` is
    ``maximum.default``, ``add.out`` is ``add.Tensor``."""
    packet = getattr(aten, func._schema.name.split("::")[1])
    params = [a.name for a in func._schema.arguments if a.name != "out"]
    for name in packet.overloads():
        candidate = getattr(packet, name)
        schema = candidate._schema
        if not schema.is_mutable and [a.name for a in schema.arguments] == params:
            return candidate
    return None


def _once_per_node(relower):
    """``relower``, run once for each base node: after a write, the results
    of a view op with many of them (``unbind``, ``split``) share one
    lowering instead of each lowering the op again."""
    last = [None, None]

    def relowered(node):
        if last[0] is not node:
            last[:] = [node, relower(node)]
        return last[1]

    return relowered


def _result(relower, k):
    """The function giving result ``k`` of a view op from its base's node."""
    return lambda node: relower(node)[k]


#: The active environments, innermost last.
_active: list[Environment] = []


def _stopped(nodes):
    """Nodes computed under ``torch.no_grad()`` from a tensor that requires
    a gradient: constants to the loss, as in PyTorch (§11.3)."""
    return tree_map(
        lambda n: g.identity(n, stop_gradient=True)
        if isinstance(n, g.Node) and n.dtype == torch.float64
        else n,
        nodes,
    )


class _Target:
    """A tensor that a write may change, made by
    :meth:`Environment._target` once the write's checks have passed."""

    __slots__ = ("tensor", "recorded")

    def __init__(self, tensor: Tensor):
        self.tensor, self.recorded = tensor, False

    def reads(self, args, kwargs):
        """The write's other inputs: whether autograd records the write, and
        so gives the tensor a history, depends on them."""
        self.recorded = torch.is_grad_enabled() and (
            self.tensor.requires_grad or _requires_grad(args, kwargs)
        )

    def write(self, node: g.Node) -> Tensor:
        """Give the tensor ``node``'s value, cast to its dtype as PyTorch
        casts an in-place op's result, and return the tensor."""
        t = self.tensor
        if node.shape != t._node.shape:
            raise AssertionError(f"in-place result {node.shape} for {t._node.shape}")
        if node.dtype != t.dtype:
            from ..ops._build import as_dtype

            node = g.Select(
                node.shape,
                t.dtype,
                (node,),
                index=tuple(g.Col(0, c) for c in node.columns),
                value=as_dtype(g.Col(0, "v"), node.dtype, t.dtype),
            )
        t._write(node)
        if self.recorded:
            # The tensor has a history from now on, which an alias taken
            # before must see (issue #21).
            t._storage.history = True
        return t


def _untracked(target: Tensor) -> bool:
    """Whether writing ``target`` would change the value of a tensor with a
    history (a ``grad_fn``) without autograd recording the write. PyTorch
    then keeps the history, so the gradient is the old value's; torch-sql
    gives the tensor the write's node, which a stop-gradient cuts from it.
    Autograd records the write only if gradients are enabled and the target
    is the tensor with the history, not a ``.detach()`` alias of it. An
    alias's base counts if it had its history when the alias was taken, or
    gained it since through a recorded write."""
    if torch.is_grad_enabled() and target.grad_fn is not None:
        return False
    return target.grad_fn is not None or target._lineage.history or target._storage.history


def _requires_grad(args, kwargs) -> bool:
    return any(
        isinstance(a, Tensor) and a.requires_grad
        for a in torch.utils._pytree.tree_leaves((args, kwargs))
    )


#: A _foreach_ op's overload, and the plain op's overload it applies.
_FOREACH_OVERLOADS = {
    "List": "Tensor",
    "ScalarList": "Scalar",
    "Tensor": "Tensor",
    "Scalar": "Scalar",
    "default": "default",
}


def _foreach(env: Environment, func, args, kwargs):
    """A ``_foreach_*`` op applies a plain op to each tensor of a list
    (spike S5). One rule lowers them: call the plain op once per tensor, with
    each list argument's element and every other argument as it is. The
    plain op's positional parameters take the foreach op's arguments in
    order, and its keyword-only ones take the rest (``value`` of
    ``addcmul_``, say). ``_foreach_norm`` is ``linalg_vector_norm``."""
    schema = func._schema
    name = schema.name.split("::")[1][len("_foreach_") :]
    provided = []
    for k, a in enumerate(schema.arguments):
        if k < len(args):
            provided.append((a, args[k]))
        elif a.name in kwargs:
            provided.append((a, kwargs[a.name]))
    lists = [v for a, v in provided if isinstance(a.type, torch._C.ListType)]
    n = len(lists[0])

    def element(a, v, i):
        if isinstance(a.type, torch._C.ListType):
            return v[i]
        if a.name == "scalars" and isinstance(v, torch.Tensor):
            return v[i].item()  # a CPU tensor of one scalar per tensor
        return v

    if name == "norm":
        ord_ = next((v for a, v in provided if a.name == "ord"), 2)
        return [aten.linalg_vector_norm(t, ord_) for t in provided[0][1]]
    # An in-place foreach op whose plain op has no in-place form, such as
    # _foreach_maximum_ (AMSGrad), applies the functional op and assigns.
    assign = name.endswith("_") and not hasattr(aten, name)
    if assign:
        name = name[:-1]
    packet = getattr(aten, name)
    plain = getattr(packet, _FOREACH_OVERLOADS.get(func._overloadname, func._overloadname), None)
    plain = plain or packet.default
    positional = [a for a in plain._schema.arguments if not a.kwarg_only]
    keyword = [a for a in plain._schema.arguments if a.kwarg_only]
    if assign:
        targets = [env._target(func, t, reads=((), {})) for t in provided[0][1]]
        env._meta(func, args, kwargs)
    results = []
    for i in range(n):
        values = [element(a, v, i) for a, v in provided]
        call_args = values[: len(positional)]
        call_kwargs = {a.name: v for a, v in zip(keyword, values[len(positional) :], strict=False)}
        results.append(plain(*call_args, **call_kwargs))
        if assign:  # autograd records each element's write apart
            targets[i].reads(call_args, call_kwargs)
    if assign:
        for target, result in zip(targets, results, strict=True):
            target.write(result._node)
    return None if not schema.returns else results


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
    g.check_dtype(out.dtype)
    return env._wrap(
        aten._to_copy.default,
        out,
        LOWERINGS[aten._to_copy.default](
            Context(aten._to_copy.default, out, env), x._node, **kwargs
        ),
    )


def _to_device(
    env: Environment, x, device, dtype=None, non_blocking=False, copy=False, memory_format=None
):
    """``aten.to.device``, which decompositions call: a read to the CPU, or a
    cast."""
    kwargs = {"device": device} if device is not None else {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    return _to_copy(env, x, **kwargs)


def _copy_(env: Environment, dst, src, non_blocking=False):
    if not isinstance(dst, Tensor):
        raise RuntimeError("copying a torch-sql tensor into a CPU tensor: use .cpu()")
    target = env._target(aten.copy_.default, dst, reads=((src,), {}))
    if not isinstance(src, Tensor):
        src = env.tensor(src.detach().to(dst.dtype))
    if tuple(src.shape) != tuple(dst.shape):
        src = src.expand(dst.shape)
    # A node of dst's own, so that two tensors never share one: as leaves,
    # they'd share a table, and their gradients would mix. Under no_grad
    # (load_state_dict), the copy is a constant, as in PyTorch.
    stop = not torch.is_grad_enabled() and src.requires_grad
    return target.write(g.identity(src._node, stop_gradient=stop))


def _bernoulli_(env: Environment, x, p=0.5, *, generator=None):
    """In place, with ``p``'s default, which the functional form lacks."""
    target = env._target(aten.bernoulli_.float, x)
    cpu = torch.empty(tuple(x.shape), dtype=x.dtype).bernoulli_(p, generator=generator)
    return target.write(env.upload(cpu))


def _scalar(env: Environment, x):
    return env.read(x).item()


def _is_nonzero(env: Environment, x):
    return bool(env.read(x).item())


_SPECIAL = {
    aten._to_copy.default: _to_copy,
    aten.to.device: _to_device,
    aten.bernoulli_.float: _bernoulli_,
    aten.copy_.default: _copy_,
    aten._local_scalar_dense.default: _scalar,
    aten.is_nonzero.default: _is_nonzero,
}


class _Factories(TorchDispatchMode):
    """While a decomposition runs: an op that creates a tensor on ``sql``
    or ``meta`` from no tensor (``scalar_tensor``, ``full``, ``arange``), as
    decompositions do for the device of their inputs, runs on the CPU and
    becomes a torch-sql tensor. Every other op passes through."""

    def __init__(self, env: Environment):
        super().__init__()
        self.env = env

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        device = kwargs.get("device")
        if (
            device is not None
            and torch.device(device).type in ("sql", "meta")
            and not any(isinstance(a, torch.Tensor) for a in torch.utils._pytree.tree_leaves(args))
        ):
            cpu = func(*args, **{**kwargs, "device": torch.device("cpu")})
            return self.env.tensor(cpu)
        return func(*args, **kwargs)


# -- The function mode ----------------------------------------------------


class _FunctionMode(TorchFunctionMode):
    """Makes tensors created on, or moved to, ``device="sql"`` torch-sql
    tensors."""

    def __init__(self, env: Environment):
        super().__init__()
        self.env = env

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        replacement = FUNCTIONS.get(func)
        if replacement is not None and _has_sql(args, kwargs):
            return replacement(*args, **kwargs)
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
        # _parse_to doesn't take copy=.
        parse = {k: v for k, v in kwargs.items() if k != "copy"}
        device, dtype, _, _ = torch._C._nn._parse_to(*args[1:], **parse)
        if device is None or not is_sql(device):
            return None  # for a torch-sql tensor, dispatch: a cast, or a read
        if isinstance(source, Tensor):
            # Already on sql, where PyTorch's C++ code can't copy to: what's
            # left is the cast, or the copy, which dispatch handles.
            copy = kwargs.get("copy", False)
            if dtype is None and not copy:
                return source
            return source.to(dtype=dtype or source.dtype, copy=copy)
        cpu = source if dtype is None else source.to(dtype)
        t = self.env.tensor(cpu.detach())
        return t.requires_grad_() if source.requires_grad else t


def _has_sql(args, kwargs) -> bool:
    return any(isinstance(a, Tensor) for a in torch.utils._pytree.tree_leaves((args, kwargs)))
