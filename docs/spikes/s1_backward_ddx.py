"""Spike S1: can `backward()` hand the gradients to ddx?

Question (design §16, S1): "Can a wrapper subclass override `backward()` so
that ddx computes the gradients and `torch.optim` works unchanged?"

Design refs: §9.1 (capturing ops), §9.2 (the recorded graph), §9.3 (the op
table), §11.1 (`backward()` calls ddx), §11.5 (the optimizer).

What this builds. A *wrapper tensor subclass* (§2) is a `torch.Tensor`
subclass that holds no storage of its own; PyTorch calls its
`__torch_dispatch__` for every ATen op that touches it. `SqlTensor` below is
one, on a `"sql"` device (PyTorch's spare `PrivateUse1` slot, finding 3). For
each op it does two things:

1. records a node of a small relational graph, if the op is in a tiny op
   table (`mm`, `addmm`, `t`, `add`, `sub`, `mul`, `pow`, `tanh`, `sum`,
   `mean`), decomposing other ops with PyTorch's decompositions first, as
   §9.3 describes;
2. runs the op on a CPU copy it carries (`_eager`).

The CPU copy is a shortcut of this spike, not of the design: it lets the
optimizer's ops run without lowering them (that is spike S5), and it gives a
reference to check against. The gradients never come from it.

`SqlTensor.backward()` writes the recorded graph as one SQL loss query over
long tables (one row per index, columns `i0, i1, ..., val`, §8.1), asks ddx
for `grad(loss, val)` of each parameter's table, and assigns each parameter's
`.grad`. PyTorch's autograd records its own graph during the forward pass
too, but nothing runs it.

Run:
    . <ldenv.sh> && uv run --project docs/spikes python docs/spikes/s1_backward_ddx.py

Findings (torch 2.15.0+cpu, ddxdb 0.4.6, DataFusion 54):

1. Yes. `loss.backward()` resolves to the subclass's override, and ddx's
   gradients match `torch.autograd` to 1e-16 (float64). `torch.optim.SGD`
   (momentum 0.9), `Adam()` and `Adam(foreach=True)` give the same
   parameters as plain PyTorch after 5 steps, to 1e-16 and 3e-16.
2. Autograd records its graph (the loss has an `MseLossBackward0`), but
   never runs it: a hook on every parameter never fires, and no backward
   ATen op reaches `__torch_dispatch__`.
3. A real `"sql"` device needs more than renaming `PrivateUse1`. With only
   `rename_privateuse1_backend`, the first autograd-recorded op aborts the
   whole process (a C++ `c10::Error`, "PyTorch is not linked with support
   for sql devices", not a Python exception), because autograd asks for the
   device's guard. The experimental
   `torch.utils.backend_registration._setup_privateuseone_for_python_backend`
   registers a guard, hooks and a device module from Python, and then
   everything works. torchax avoids the question by putting its wrapper on
   the `meta` device, which design §9.1 also describes; either works, but a
   `"sql"` device shows up in `p.device` and in error messages.
4. `p.grad = g` checks device, dtype and size, not the subclass: a CPU
   tensor, a float32 tensor and a transposed tensor are all refused with
   clear errors; a `SqlTensor` of the right dtype and size is accepted.
5. `nn.Parameter(SqlTensor(...))` keeps the subclass (`type(p) is
   SqlTensor`, `isinstance(p, nn.Parameter)`), by calling `detach()`
   through `__torch_dispatch__`, so `detach` must keep the graph node.
   `parameters()` and `zero_grad()` (which sets `.grad = None`) work
   unchanged.
6. For a subclass, `foreach=None` (the default) picks the per-tensor loop.
   Its in-place ops: SGD `add_`, `mul_`; Adam `add_`, `lerp_`, `mul_`,
   `addcmul_`, `addcdiv_` (plus out-of-place `sqrt` and `div`). With
   `foreach=True`, Adam calls `_foreach_add_`, `_foreach_lerp_`,
   `_foreach_mul_`, `_foreach_addcmul_`, `_foreach_addcdiv_`,
   `_foreach_div_` and `_foreach_sqrt`, which reach `__torch_dispatch__`
   whole. Spike S5 covers lowering them.
7. Decompositions: `mse_loss` decomposes into `sub`, `pow` and `mean`, with
   `aten.to.dtype` casts to the same dtype around them, which the op table
   must accept. `torch._decomp.decomposition_table` also holds
   decompositions into `prims` ops (`sigmoid` becomes `prims.div`), below
   core ATen; torch-sql should use `core_aten_decompositions()` (as S6
   does) so its op table stays at core ATen (design §9.3).
8. How ddx picks the target of `grad(loss, val)` when every table has a
   `val` column: it differentiates with respect to the column of the table
   the outer `SELECT` reads (`SELECT i0, i1, grad(loss, val) FROM
   lin1_weight`); `grad(loss, w.val)` works too. So the form is one query
   per parameter, each reading its own table. Each such query is its own
   ddx program, which repeats the forward work per parameter. ddx's plan API
   doesn't have that limit: `ddxdb.ad.grad(ctx, loss, ["w1.val", "w2.val"])`
   is one program with one backward pass and a gradient per table, built
   once and run every step (its docstring says so). torch-sql should call
   that rather than SQL `grad()` (open question 6).
9. `vjp(f, val, ct)` works when `f` ends in an aggregate or is row-local
   over the target table, and is refused ("vjp of an output joined to
   constant data: ddx cannot tell which columns identify its rows") when
   `f` is a join with a constant table and no aggregate after it. `grad` of
   `SUM(f.val * ct.val)` gives the vector-Jacobian product in every case;
   spike S6 uses it.
"""

import sys
import warnings

import numpy as np
import pyarrow as pa
import torch
from torch.utils._pytree import tree_leaves, tree_map

from ddxdb import Context

# Renaming PrivateUse1 alone is not enough: autograd asks for the device's
# guard and aborts the process ("PyTorch is not linked with support for sql
# devices"). This experimental helper also registers a device guard, hooks and
# a device module, all from Python.
torch.utils.backend_registration._setup_privateuseone_for_python_backend("sql")
SQL = torch.device("sql")
aten = torch.ops.aten

# ----------------------------------------------------------------------------
# The recorded graph: each node is a CTE with index columns i0..ik-1 and val.


class Node:
    count = 0

    def __init__(self, shape, sql=None, inputs=(), table=None):
        Node.count += 1
        self.name = table or f"n{Node.count}"
        self.shape = tuple(shape)
        self.sql = sql  # None for a leaf, which is a registered table
        self.inputs = tuple(inputs)

    def cols(self, alias=None):
        p = f"{alias}." if alias else ""
        return [f"{p}i{k}" for k in range(len(self.shape))]


class Unlowered(Node):
    """Stands for the output of an op the op table doesn't cover. Fine for
    the optimizer's ops, which this spike runs eagerly; an error if a loss
    depends on it (P3)."""

    def __init__(self, func, shape):
        super().__init__(shape)
        self.op = func


def topo(root):
    seen, order = set(), []

    def visit(n):
        if n.name in seen:
            return
        seen.add(n.name)
        for i in n.inputs:
            visit(i)
        order.append(n)

    visit(root)
    return order


def broadcast(a, b, op):
    """Elementwise op with PyTorch's broadcasting: a join on the index
    columns both sides really have (size > 1 on both, after right-alignment).
    `a` or `b` may be a Python number."""
    na = a if isinstance(a, Node) else None
    nb = b if isinstance(b, Node) else None
    shape = torch.broadcast_shapes(*(n.shape for n in (na, nb) if n))
    r = len(shape)
    sel, on = [], []
    for d in range(r):
        src = []
        for n, al in ((na, "a"), (nb, "b")):
            if n is None:
                continue
            k = d - (r - len(n.shape))
            if k >= 0 and n.shape[k] == shape[d] and shape[d] != 1:
                src.append(f"{al}.i{k}")
            elif k >= 0 and n.shape[k] == shape[d] == 1:
                src.append(f"{al}.i{k}")
        sel.append(f"{src[0]} AS i{d}")
        on += [f"{src[0]} = {s}" for s in src[1:]]
    ea = "a.val" if na else repr(float(a))
    eb = "b.val" if nb else repr(float(b))
    frm = []
    if na:
        frm.append(f"{na.name} a")
    if nb:
        frm.append(f"{nb.name} b")
    join = frm[0] if len(frm) == 1 else (
        f"{frm[0]} JOIN {frm[1]} ON {' AND '.join(on)}" if on else f"{frm[0]} CROSS JOIN {frm[1]}"
    )
    sql = f"SELECT {', '.join(sel + [f'({ea}) {op} ({eb}) AS val'])} FROM {join}"
    return Node(shape, sql, [n for n in (na, nb) if n])


def unary(a, expr):
    return Node(a.shape, f"SELECT {', '.join(a.cols() + [expr.format('val') + ' AS val'])} FROM {a.name}", [a])


def mm(a, b):
    sql = (f"SELECT a.i0 AS i0, b.i1 AS i1, SUM(a.val * b.val) AS val "
           f"FROM {a.name} a JOIN {b.name} b ON a.i1 = b.i0 GROUP BY a.i0, b.i1")
    return Node((a.shape[0], b.shape[1]), sql, [a, b])


def transpose(a):
    return Node(a.shape[::-1], f"SELECT i1 AS i0, i0 AS i1, val FROM {a.name}", [a])


def reduce(a, dims, mean):
    dims = sorted(d % len(a.shape) for d in dims) if dims else list(range(len(a.shape)))
    keep = [k for k in range(len(a.shape)) if k not in dims]
    n = int(np.prod([a.shape[d] for d in dims]))
    agg = f"SUM(val) / {float(n)!r}" if mean else "SUM(val)"
    sel = [f"i{k} AS i{j}" for j, k in enumerate(keep)] + [f"{agg} AS val"]
    group = f" GROUP BY {', '.join(f'i{k}' for k in keep)}" if keep else ""
    return Node([a.shape[k] for k in keep], f"SELECT {', '.join(sel)} FROM {a.name}{group}", [a])


def _sum(x, dims=None, keepdim=False, dtype=None):
    assert not keepdim and dtype is None
    return reduce(x, dims, mean=False)


def _mean(x, dims=None, keepdim=False, dtype=None):
    assert not keepdim and dtype is None
    return reduce(x, dims, mean=True)


# The op table (§9.3): ATen op -> a function of nodes and Python values.
LOWER = {
    aten.mm.default: mm,
    aten.addmm.default: lambda bias, a, b: broadcast(mm(a, b), bias, "+"),
    aten.t.default: transpose,
    aten.add.Tensor: lambda a, b: broadcast(a, b, "+"),
    aten.sub.Tensor: lambda a, b: broadcast(a, b, "-"),
    aten.mul.Tensor: lambda a, b: broadcast(a, b, "*"),
    aten.tanh.default: lambda a: unary(a, "tanh({})"),
    aten.pow.Tensor_Scalar: lambda a, p: unary(a, f"power({{}}, {float(p)!r})"),
    aten.sum.default: _sum,
    aten.sum.dim_IntList: _sum,
    aten.mean.default: _mean,
    aten.mean.dim: _mean,
}
# Ops that only change how a tensor is referred to keep its node.
SAME_NODE = {aten.detach.default, aten.alias.default, aten.clone.default}

DISPATCHED = []  # every ATen op that reached __torch_dispatch__


# ----------------------------------------------------------------------------
# The tensor subclass.


class SqlTensor(torch.Tensor):
    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(cls, eager, node=None, requires_grad=False):
        t = torch.Tensor._make_wrapper_subclass(
            cls, eager.shape, dtype=eager.dtype, device=SQL, requires_grad=requires_grad)
        t._eager, t._node, t._param = eager, node, None
        return t

    def __repr__(self):
        return f"SqlTensor({self._node.name if self._node else 'eager'}, {tuple(self.shape)})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        DISPATCHED.append(func)
        has_node = any(isinstance(a, SqlTensor) and a._node is not None for a in tree_leaves(args))
        if func not in LOWER and func not in SAME_NODE and has_node and not func._schema.is_mutable:
            decomp = torch._decomp.decomposition_table.get(func)
            if decomp is not None:
                return decomp(*args, **kwargs)  # re-enters dispatch with simpler ops
        unwrap = lambda x: x._eager if isinstance(x, SqlTensor) else x
        out = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))
        node = None
        if func is aten.to.dtype and has_node:  # mse_loss's decomposition casts to the same dtype
            assert args[1] == args[0].dtype, "a cast that changes dtype is not lowered here"
            node = args[0]._node
        elif func in LOWER and has_node:
            node = LOWER[func](*tree_map(lambda x: x._node if isinstance(x, SqlTensor) else x, args),
                               **kwargs)
        elif func in SAME_NODE and isinstance(args[0], SqlTensor):
            node = args[0]._node
        elif has_node and isinstance(out, torch.Tensor):
            node = Unlowered(func, out.shape)
        if func._schema.is_mutable:  # in place: only leaves (parameters) may be updated
            for a in args:
                if isinstance(a, SqlTensor) and a._node is not None and a._node.sql is not None:
                    raise NotImplementedError(f"in-place {func} on a computed tensor")
            return args[0] if isinstance(out, torch.Tensor) else out
        wrap = lambda x: SqlTensor(x, node) if isinstance(x, torch.Tensor) else x
        res = tree_map(wrap, out)
        if func in SAME_NODE and isinstance(args[0], SqlTensor):
            res._param = args[0]._param
        return res

    def backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None):
        """ddx computes every parameter's gradient (§11.1)."""
        assert gradient is None and self.dim() == 0, "a scalar loss"
        nodes = topo(self._node)
        bad = [n.op for n in nodes if isinstance(n, Unlowered)]
        if bad:
            raise NotImplementedError(f"OperatorNotFound: the loss depends on {bad[0]}")
        leaves = [n for n in nodes if n.sql is None]
        for n in leaves:  # (re)register each table from its current values
            register(n.name, n.value())
        ctes = ", ".join(f"{n.name} AS ({n.sql})" for n in nodes if n.sql is not None)
        loss = f"loss AS (SELECT val AS l FROM {self._node.name})"
        for n in leaves:
            p = n.param
            if p is None or not p.requires_grad:
                continue
            cols = ", ".join(n.cols())
            q = f"WITH {ctes}, {loss} SELECT {cols}, grad(loss, val) AS g FROM {n.name} ORDER BY {cols}"
            g = torch.from_numpy(CTX.sql(q).to_pandas()["g"].to_numpy().copy()).reshape(p.shape)
            g = g.to(p.dtype)
            if p.grad is None:
                p.grad = SqlTensor(g)
            else:  # PyTorch accumulates into an existing .grad
                p.grad._eager.add_(g)


CTX = Context()


def register(name, value):
    idx = np.indices(value.shape).reshape(value.dim(), -1) if value.dim() else np.zeros((0, 1), int)
    cols = {f"i{k}": pa.array(idx[k], pa.int64()) for k in range(value.dim())}
    cols["val"] = pa.array(value.detach().reshape(-1).numpy())
    CTX.deregister_table(name)
    CTX.register_record_batches(name, [[pa.record_batch(cols)]])


def leaf(value, name, param=None):
    """A tensor that is a table. Parameters keep a reference back, so
    backward() can find them, and their values are read at each backward."""
    n = Node(value.shape, table=name)
    t = SqlTensor(value, n)
    n.value = lambda: t._eager
    return t


def sql_parameter(value, name):
    t = leaf(value.clone(), name)
    p = torch.nn.Parameter(t)
    p._node.value = lambda: p._eager
    p._node.param = p
    return p


Node.param = None

# ----------------------------------------------------------------------------
# The checks.

FAILED = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        FAILED.append(name)


class Model(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.lin1 = torch.nn.Linear(4, 3).double()
        self.lin2 = torch.nn.Linear(3, 2).double()
        if params is not None:  # replace each parameter with a SqlTensor one
            for mod_name, mod in (("lin1", self.lin1), ("lin2", self.lin2)):
                for pn in ("weight", "bias"):
                    setattr(mod, pn, sql_parameter(params[f"{mod_name}.{pn}"], f"{mod_name}_{pn}"))

    def forward(self, x, y):
        return torch.nn.functional.mse_loss(self.lin2(torch.tanh(self.lin1(x))), y)


def main():
    torch.manual_seed(0)
    ref = Model(None)
    init = {k: v.detach().clone() for k, v in ref.named_parameters()}
    x, y = torch.randn(5, 4, dtype=torch.float64), torch.randn(5, 2, dtype=torch.float64)

    model = Model(init)
    p0 = model.lin1.weight
    check("nn.Parameter keeps the subclass", type(p0) is SqlTensor and isinstance(p0, torch.nn.Parameter),
          f"type={type(p0).__name__}, device={p0.device}")
    sx, sy = leaf(x, "x"), leaf(y, "y")

    # One step: gradients.
    fired = []
    for p in model.parameters():
        p.register_hook(lambda g: fired.append(g))
    with warnings.catch_warnings(record=True) as warned:
        warnings.simplefilter("always")
        loss = model(sx, sy)
        n_fwd = len(DISPATCHED)
        loss.backward()
    ref_loss = ref(x, y)
    ref_loss.backward()
    check("forward loss matches", abs(loss._eager.item() - ref_loss.item()) < 1e-14,
          f"{loss._eager.item():.6f}")
    errs = {k: (dict(model.named_parameters())[k].grad._eager - v.grad).abs().max().item()
            for k, v in ref.named_parameters()}
    check("ddx gradients == torch.autograd", max(errs.values()) < 1e-12,
          " ".join(f"{k}={e:.1e}" for k, e in errs.items()))
    check("autograd still recorded a graph", loss.grad_fn is not None, type(loss.grad_fn).__name__)
    check("autograd's graph never ran (no hook fired)", not fired, f"{len(fired)} hooks")
    bw_ops = [str(f) for f in DISPATCHED[n_fwd:] if "backward" in str(f)]
    check("no backward ATen op reached dispatch", not bw_ops, str(bw_ops[:3]))
    check("grad is a SqlTensor on 'sql'", type(p0.grad) is SqlTensor and p0.grad.device == SQL)
    check("no warnings", not warned, "; ".join(str(w.message)[:80] for w in warned))

    # What PyTorch requires of an assigned .grad.
    def assign(g):
        try:
            p0.grad = g
            return "accepted"
        except Exception as e:  # noqa: BLE001
            return f"{type(e).__name__}: {str(e)[:70]}"
    saved = p0.grad
    r_cpu = assign(torch.zeros_like(x[:3, :4]))
    r_f32 = assign(SqlTensor(torch.zeros(3, 4)))
    r_shape = assign(SqlTensor(torch.zeros(4, 3, dtype=torch.float64)))
    r_ok = assign(SqlTensor(torch.zeros(3, 4, dtype=torch.float64)))
    p0.grad = saved
    print(f"      .grad = cpu tensor -> {r_cpu}\n      .grad = float32 SqlTensor -> {r_f32}"
          f"\n      .grad = wrong shape -> {r_shape}\n      .grad = matching SqlTensor -> {r_ok}")
    check(".grad checks device, dtype and size",
          r_ok == "accepted" and "accepted" not in (r_cpu, r_f32, r_shape))

    try:  # an op outside the op table, with no decomposition, fails loud (P3)
        torch.sigmoid(model.lin1(sx)).sum().backward()
        r = "no error"
    except NotImplementedError as e:
        r = str(e)
    check("a loss through an unlowered op raises", r.startswith("OperatorNotFound"), r)

    model.zero_grad()
    check("zero_grad() sets .grad to None", all(p.grad is None for p in model.parameters()))

    # Several optimizer steps against plain PyTorch.
    for name, make in [("SGD(momentum=0.9)", lambda ps: torch.optim.SGD(ps, lr=0.1, momentum=0.9)),
                       ("Adam()", lambda ps: torch.optim.Adam(ps, lr=0.01)),
                       ("Adam(foreach=True)", lambda ps: torch.optim.Adam(ps, lr=0.01, foreach=True))]:
        torch.manual_seed(0)
        r, m = Model(None), Model(init)
        r.load_state_dict({k: v for k, v in init.items()})
        o_r, o_m = make(r.parameters()), make(m.parameters())
        start = len(DISPATCHED)
        for _ in range(5):
            o_r.zero_grad(); r(x, y).backward(); o_r.step()
            o_m.zero_grad(); m(sx, sy).backward(); o_m.step()
        ops = sorted({f.__name__.split(".")[0] for f in DISPATCHED[start:]} - {f.__name__.split(".")[0] for f in LOWER})
        err = max((dict(m.named_parameters())[k]._eager - v).abs().max().item()
                  for k, v in r.named_parameters())
        check(f"{name}: 5 steps match plain PyTorch", err < 1e-12, f"max|diff|={err:.1e}")
        print(f"      optimizer ops seen: {', '.join(o for o in ops if o.startswith('_foreach') or o.endswith('_'))[:200]}")

    if FAILED:
        print(f"\n{len(FAILED)} check(s) failed: {FAILED}")
        sys.exit(1)
    print("\nAll S1 checks pass.")


if __name__ == "__main__":
    main()
