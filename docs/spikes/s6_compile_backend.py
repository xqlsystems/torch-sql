"""Spike S6: a `torch.compile` backend whose backward pass is ddx's.

Question (design §16, S6): "Can a `torch.compile` backend take Dynamo's
forward graph, run it as SQL, and supply ddx's backward pass through an
autograd function, with DDP still averaging the gradients? How do partial
tensors pass through Dynamo?"

Design refs: §2 (`torch.compile`, custom ops and autograd functions), §6.5
(the compile backend), §6.6 (the boundary), §8.3 (partial tensors), §11.2
(checkpointing as ddx regions), §17 open question 7 (cache keys).

What this builds. `torch.compile(model, backend=sql_backend)` hands
`sql_backend` an *FX graph* (PyTorch's graph format) for each piece of the
program Dynamo captured. The backend:

1. traces that graph to ATen ops with `make_fx` and PyTorch's core ATen
   decompositions, under `no_grad`, so it gets the forward pass only;
2. lowers the ATen graph to SQL CTEs over long tables (one row per index,
   columns `i0, i1, ..., val`, design §8.1), with a small op table;
3. returns a function that calls `SqlGraph.apply`, a
   `torch.autograd.Function`. Its forward registers the inputs as tables and
   runs the output queries on DataFusion (through `ddxdb.Context`). Its
   backward registers the incoming cotangents as tables and asks ddx for
   `grad(loss, val)` of each input that needs a gradient, where `loss` is
   `SUM(output.val * cotangent.val)`, which is the vector-Jacobian product.

A *partial tensor* (design §8.3) enters as a custom op,
`torch.ops.torch_sql.restrict(x, mask)`, registered with `torch.library`.
In PyTorch it is `x * mask`; the backend lowers it to a filter that drops the
masked rows.

DDP runs in two CPU processes over gloo (PyTorch's CPU collective backend),
launched by this script itself.

Run:
    . <ldenv.sh> && uv run --project docs/spikes python docs/spikes/s6_compile_backend.py

Findings (torch 2.15.0+cpu, ddxdb 0.4.6, DataFusion 54):

1. Yes. Gradients through ddx match eager autograd to 1e-16 (float64) for a
   2-layer MLP with tanh and an MSE loss.
2. Parameters arrive as graph inputs (Dynamo inlines `nn.Module`s by
   default), as real `nn.Parameter`s, not fake tensors. So each parameter is
   an input table, and the autograd function returns its gradient like any
   other input's. A consequence: guards don't pin the module, so a second
   `MLP` instance reuses the first one's compiled graph.
3. After `make_fx` with `core_aten_decompositions()`, the MLP is
   `permute`, `addmm`, `tanh`, `sub`, `pow` and `mean`, all core ATen.
4. A graph break (`.item()` inside `forward`) gives two graphs, each its own
   `SqlGraph`. PyTorch's autograd chains them: the second graph's ddx
   backward produces the cotangent of `h`, which the first graph's ddx
   backward consumes. Gradients still match to 1e-16.
5. `torch.utils.checkpoint` reaches the backend as one node,
   `torch.ops.higher_order.tag_activation_checkpoint(wrap_body_0, ...)`,
   whose body (`permute`, `addmm`, `tanh` here) is a submodule of the FX
   graph. `make_fx` inlines it, so the marker is gone after decomposition.
   The backend must read the regions from Dynamo's graph before tracing, as
   this spike does, and pass them to ddx as regions (design §11.2, ddx#251).
6. The custom op `torch_sql::restrict` survives Dynamo and `make_fx` as one
   node, `torch.ops.torch_sql.restrict.default`, since it has no
   decomposition. Lowered to a filter, its output is partial, and the loss
   and gradients agree with the dense PyTorch definition (`x * mask`) to
   1e-15. This is how partial tensors can pass through Dynamo. A `mean` over
   a partial tensor is refused here, because SQL would count only the rows
   that exist (design §8.3); the lowering must know which it means.
7. DDP: Dynamo's DDP optimizer splits the forward graph at gradient-bucket
   boundaries, so the backend received 2 graphs on each rank. DDP's hooks
   fire on the gradients the autograd function returns, and the averaged
   gradients match eager DDP to 3e-17 and are identical on both ranks. (To
   get several buckets from a tiny model, the spike shrinks DDP's first
   bucket, which the DDP optimizer requires to be no larger than the rest.)
8. Guards: with `dynamic=False`, a new batch size compiles once more and a
   size seen before reuses its graph. With the default, the second size is
   compiled with symbolic sizes: a `SymInt` arrives as an extra graph input
   (`s17`), and tensor shapes are symbolic. A lowering that writes literal
   sizes into SQL must refuse it (as here) or treat sizes as parameters.
9. Cost (12-core CPU shared with other jobs, so rough): compiling one graph
   takes 50 to 140 ms, mostly `make_fx`. A step takes about 160 to 220 ms:
   one forward query and one ddx program per parameter (5 in all), against
   0.2 ms eager. Dynamo itself costs nothing measurable: with a
   pass-through backend a step is 0.23 ms, the same as eager. All of the
   cost is planning and running SQL, which is what one program per step
   (`ddxdb.prepare`) and the runtime should attack.
"""

import os
import sys
import time

import numpy as np
import pyarrow as pa
import torch
import torch.distributed as dist
from torch._decomp import core_aten_decompositions
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils.checkpoint import checkpoint

from ddxdb import Context

aten = torch.ops.aten

# ----------------------------------------------------------------------------
# A partial tensor, as a custom op. In PyTorch: x * mask.


@torch.library.custom_op("torch_sql::restrict", mutates_args=())
def restrict(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return x * mask


@restrict.register_fake
def _(x, mask):
    return torch.empty_like(x)


def _restrict_bwd(ctx, g):
    return g * ctx.mask, None


def _restrict_setup(ctx, inputs, output):
    ctx.mask = inputs[1]


restrict.register_autograd(_restrict_bwd, setup_context=_restrict_setup)

# ----------------------------------------------------------------------------
# Lowering: each ATen node becomes a CTE with columns i0..ik-1, val.


class Rel:
    def __init__(self, name, shape, sql=None, partial=False):
        self.name, self.shape, self.sql, self.partial = name, tuple(shape), sql, partial


def lower(gm, prefix):
    """The ATen graph `gm` as a list of CTEs, its input tables and its
    output relations."""
    ctes, inputs, outputs = [], [], []
    env = {}

    def new(node, shape, sql, partial=False, suffix=""):
        r = Rel(f"{prefix}_{node.name}{suffix}", shape, sql, partial)
        ctes.append(r)
        return r

    def elementwise(node, a, b, op):
        na = a if isinstance(a, Rel) else None
        nb = b if isinstance(b, Rel) else None
        shape = torch.broadcast_shapes(*(r.shape for r in (na, nb) if r))
        sel, on = [], []
        for d in range(len(shape)):
            src = []
            for r, al in ((na, "a"), (nb, "b")):
                if r is None:
                    continue
                k = d - (len(shape) - len(r.shape))
                if k >= 0 and r.shape[k] == shape[d]:
                    src.append(f"{al}.i{k}")
            sel.append(f"{src[0]} AS i{d}")
            on += [f"{src[0]} = {s}" for s in src[1:]]
        ea = "a.val" if na else repr(float(a))
        eb = "b.val" if nb else repr(float(b))
        frm = [f"{r.name} {al}" for r, al in ((na, "a"), (nb, "b")) if r]
        join = frm[0] if len(frm) == 1 else (
            f"{frm[0]} JOIN {frm[1]} ON {' AND '.join(on)}" if on else f"{frm[0]} CROSS JOIN {frm[1]}")
        partial = any(r.partial for r in (na, nb) if r)
        return new(node, shape, f"SELECT {', '.join(sel + [f'({ea}) {op} ({eb}) AS val'])} FROM {join}", partial)

    def mm(node, a, b, suffix=""):
        sql = (f"SELECT a.i0 AS i0, b.i1 AS i1, SUM(a.val * b.val) AS val FROM {a.name} a "
               f"JOIN {b.name} b ON a.i1 = b.i0 GROUP BY a.i0, b.i1")
        return new(node, (a.shape[0], b.shape[1]), sql, suffix=suffix)

    def reduce(node, a, dims, mean):
        dims = sorted(d % len(a.shape) for d in dims) if dims else list(range(len(a.shape)))
        keep = [k for k in range(len(a.shape)) if k not in dims]
        n = float(np.prod([a.shape[d] for d in dims]))
        agg = f"SUM(val) / {n!r}" if mean else "SUM(val)"
        sel = [f"i{k} AS i{j}" for j, k in enumerate(keep)] + [f"{agg} AS val"]
        group = f" GROUP BY {', '.join(f'i{k}' for k in keep)}" if keep else ""
        return new(node, [a.shape[k] for k in keep], f"SELECT {', '.join(sel)} FROM {a.name}{group}")

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            if not isinstance(node.meta["val"], torch.Tensor):
                raise NotImplementedError(f"symbolic size input {node.meta['val']} (compile with dynamic=False)")
            shape = tuple(node.meta["val"].shape)
            for s in shape:
                if not isinstance(s, int):
                    raise NotImplementedError(f"symbolic size {s} (compile with dynamic=False)")
            r = Rel(f"{prefix}_in{len(inputs)}", shape)
            inputs.append(r)
            env[node] = r
            continue
        if node.op == "output":
            outputs = [env[a] for a in node.args[0]]
            continue
        f = node.target
        args = [env[a] if isinstance(a, torch.fx.Node) else a for a in node.args]
        if f is aten.permute.default and list(args[1]) == [1, 0] or f is aten.t.default:
            a = args[0]
            r = new(node, a.shape[::-1], f"SELECT i1 AS i0, i0 AS i1, val FROM {a.name}", a.partial)
        elif f is aten.mm.default:
            r = mm(node, *args)
        elif f is aten.addmm.default:
            r = elementwise(node, mm(node, args[1], args[2], suffix="_mm"), args[0], "+")
        elif f in (aten.add.Tensor, aten.sub.Tensor, aten.mul.Tensor):
            r = elementwise(node, args[0], args[1], {aten.add.Tensor: "+", aten.sub.Tensor: "-",
                                                     aten.mul.Tensor: "*"}[f])
        elif f is aten.tanh.default:
            a = args[0]
            cols = ", ".join(f"i{k}" for k in range(len(a.shape)))
            r = new(node, a.shape, f"SELECT {cols}, tanh(val) AS val FROM {a.name}", a.partial)
        elif f is aten.pow.Tensor_Scalar:
            a = args[0]
            cols = ", ".join(f"i{k}" for k in range(len(a.shape)))
            r = new(node, a.shape, f"SELECT {cols}, power(val, {float(args[1])!r}) AS val FROM {a.name}", a.partial)
        elif f in (aten.sum.default, aten.sum.dim_IntList):
            r = reduce(node, args[0], args[1] if len(args) > 1 else None, mean=False)
        elif f in (aten.mean.default, aten.mean.dim):
            if args[0].partial:
                raise NotImplementedError("mean of a partial tensor (it would count only rows that exist)")
            r = reduce(node, args[0], args[1] if len(args) > 1 else None, mean=True)
        elif f is torch.ops.torch_sql.restrict.default:
            x, m = args
            cols = [f"x.i{k}" for k in range(len(x.shape))]
            on = " AND ".join(f"x.i{k} = m.i{k}" for k in range(len(x.shape)))
            r = new(node, x.shape, f"SELECT {', '.join(c + ' AS ' + c[2:] for c in cols)}, x.val "
                                   f"FROM {x.name} x JOIN {m.name} m ON {on} WHERE m.val <> 0", partial=True)
        else:
            raise NotImplementedError(f"OperatorNotFound: {f}")
        env[node] = r
    return ctes, inputs, outputs


# ----------------------------------------------------------------------------
# Running SQL, and the boundary between long tables and dense tensors.

CTX = Context()
STATS = {"compiles": 0, "graphs": [], "queries": 0, "regions": []}


def register(name, t):
    t = t.detach()
    idx = np.indices(t.shape).reshape(t.dim(), -1) if t.dim() else np.zeros((0, 1), int)
    cols = {f"i{k}": pa.array(idx[k], pa.int64()) for k in range(t.dim())}
    cols["val"] = pa.array(t.reshape(-1).to(torch.float64).numpy())
    CTX.deregister_table(name)
    CTX.register_record_batches(name, [[pa.record_batch(cols)]])


def with_clause(ctes):
    return "WITH " + ", ".join(f"{r.name} AS ({r.sql})" for r in ctes)


def dense(rows, shape, dtype):
    """Long rows to a dense tensor; absent rows (a partial tensor) are zero."""
    out = np.zeros(shape, dtype=np.float64)
    if len(shape) == 0:
        return torch.tensor(rows["val"][0] if len(rows["val"]) else 0.0, dtype=dtype)
    idx = tuple(rows[f"i{k}"].to_numpy() for k in range(len(shape)))
    out[idx] = rows["val"].to_numpy()
    return torch.from_numpy(out).to(dtype)


def run(sql):
    STATS["queries"] += 1
    return CTX.sql(sql).to_pandas()


class SqlGraph(torch.autograd.Function):
    @staticmethod
    def forward(ctx, plan, *inputs):
        ctes, ins, outs = plan
        for r, t in zip(ins, inputs):
            register(r.name, t)
        w = with_clause(ctes)
        results = []
        for o in outs:
            cols = [f"i{k}" for k in range(len(o.shape))]
            q = f"{w} SELECT {', '.join(cols + ['val'])} FROM {o.name}"
            results.append(dense(run(q), o.shape, inputs[0].dtype))
        ctx.plan = plan
        ctx.save_for_backward(*inputs)
        return tuple(results)

    @staticmethod
    def backward(ctx, *cts):
        ctes, ins, outs = ctx.plan
        inputs = ctx.saved_tensors
        for r, t in zip(ins, inputs):  # the same names may have been reused since
            register(r.name, t)
        terms = []
        for j, (o, ct) in enumerate(zip(outs, cts)):
            if ct is None:
                continue
            name = f"{o.name}_ct"
            register(name, ct)
            on = " AND ".join(f"o.i{k} = c.i{k}" for k in range(len(o.shape)))
            join = f"JOIN {name} c ON {on}" if on else f"CROSS JOIN {name} c"
            terms.append(f"SELECT SUM(o.val * c.val) AS l FROM {o.name} o {join}")
        loss = "loss AS (SELECT SUM(l) AS l FROM (" + " UNION ALL ".join(terms) + "))"
        w = with_clause(ctes) + ", " + loss
        grads = [None]
        for k, (r, t) in enumerate(zip(ins, inputs)):
            if not ctx.needs_input_grad[k + 1]:
                grads.append(None)
                continue
            cols = ", ".join(f"i{d}" for d in range(len(r.shape)))
            q = f"{w} SELECT {cols + ', ' if cols else ''}grad(loss, val) AS val FROM {r.name}"
            rows = run(q)
            rows["val"] = rows["val"].fillna(0.0)
            grads.append(dense(rows, r.shape, t.dtype))
        return tuple(grads)


def sql_backend(gm, example_inputs):
    STATS["compiles"] += 1
    prefix = f"g{STATS['compiles']}"
    t0 = time.perf_counter()
    # Regions first: make_fx inlines checkpointed bodies, so read them here.
    for node in gm.graph.nodes:
        if node.target is torch.ops.higher_order.tag_activation_checkpoint:
            body = getattr(gm, node.args[0].target)
            vals = [a.meta["example_value"] for a in node.args[1:]]
            with torch.no_grad():
                traced = make_fx(body, decomposition_table=core_aten_decompositions())(*vals)
            ops = [str(n.target) for n in traced.graph.nodes if n.op == "call_function"]
            STATS["regions"].append((node.args[0].target, ops))
    with torch.no_grad():
        aten_gm = make_fx(gm, decomposition_table=core_aten_decompositions())(*example_inputs)
    plan = lower(aten_gm, prefix)
    STATS["graphs"].append({
        "dynamo_inputs": [type(e).__name__ for e in example_inputs],
        "aten_ops": [str(n.target) for n in aten_gm.graph.nodes if n.op == "call_function"],
        "dynamo_nodes": [str(n.target) for n in gm.graph.nodes if n.op == "call_function"],
        "compile_ms": (time.perf_counter() - t0) * 1e3,
    })
    return lambda *args: SqlGraph.apply(plan, *args)


# ----------------------------------------------------------------------------
# Models and checks.

FAILED = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    if not ok:
        FAILED.append(name)


class MLP(torch.nn.Module):
    def __init__(self, brk=False, ckpt=False, mask=None):
        super().__init__()
        torch.manual_seed(0)
        self.lin1 = torch.nn.Linear(4, 3).double()
        self.lin2 = torch.nn.Linear(3, 2).double()
        self.brk, self.ckpt, self.mask = brk, ckpt, mask

    def block(self, x):
        return torch.tanh(self.lin1(x))

    def forward(self, x, y):
        h = checkpoint(self.block, x, use_reentrant=False) if self.ckpt else self.block(x)
        if self.brk and h.sum().item() > -1e30:  # .item() breaks the graph
            pass
        out = self.lin2(h)
        if self.mask is not None:
            return (torch.ops.torch_sql.restrict(out - y, self.mask) ** 2).sum()
        return torch.nn.functional.mse_loss(out, y)


def grads_vs_eager(make, x, y, label):
    ref, m = make(), make()
    ref(x, y).backward()
    before = STATS["compiles"]
    c = torch.compile(m, backend=sql_backend, dynamic=False)
    loss = c(x, y)
    loss.backward()
    err = max((dict(m.named_parameters())[k].grad - p.grad).abs().max().item()
              for k, p in ref.named_parameters())
    check(f"{label}: grads == eager autograd", err < 1e-12,
          f"max|diff|={err:.1e}, graphs={STATS['compiles'] - before}")
    return STATS["compiles"] - before


def ddp_worker(rank, world, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    # Buckets of ~64 bytes, so the tiny model has several, and Dynamo's DDP
    # optimizer splits the graph at their boundaries. It asserts that the
    # first bucket is no larger than the rest, so shrink that one too.
    dist._DEFAULT_FIRST_BUCKET_BYTES = 64
    cap = 64 / 2**20
    torch.manual_seed(100 + rank)
    x, y = torch.randn(6, 4, dtype=torch.float64), torch.randn(6, 2, dtype=torch.float64)
    ref = torch.nn.parallel.DistributedDataParallel(MLP(), bucket_cap_mb=cap)
    ref(x, y).backward()
    ddp = torch.nn.parallel.DistributedDataParallel(MLP(), bucket_cap_mb=cap)
    c = torch.compile(ddp, backend=sql_backend, dynamic=False)
    c(x, y).backward()
    err = max((dict(ddp.module.named_parameters())[k].grad - p.grad).abs().max().item()
              for k, p in ref.module.named_parameters())
    # Averaged means: equal on every rank.
    g = dict(ddp.module.named_parameters())["lin1.weight"].grad.clone()
    others = [torch.zeros_like(g) for _ in range(world)]
    dist.all_gather(others, g)
    same = max((o - g).abs().max().item() for o in others)
    if rank == 0:
        check("DDP: compiled grads == eager DDP grads", err < 1e-12,
              f"max|diff|={err:.1e}, backend graphs on rank 0={STATS['compiles']}")
        check("DDP: gradients equal across ranks (averaged)", same < 1e-15, f"{same:.1e}")
        sys.exit(1 if FAILED else 0)
    dist.destroy_process_group()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "ddp":
        ddp_worker(int(sys.argv[2]), 2, int(sys.argv[3]))
        return
    torch.manual_seed(1)
    x, y = torch.randn(5, 4, dtype=torch.float64), torch.randn(5, 2, dtype=torch.float64)

    grads_vs_eager(MLP, x, y, "MLP")
    g = STATS["graphs"][-1]
    check("parameters arrive as graph inputs", g["dynamo_inputs"].count("Parameter") == 4, str(g["dynamo_inputs"]))
    print(f"      ATen ops: {g['aten_ops']}")

    n = grads_vs_eager(lambda: MLP(brk=True), x, y, "graph break (.item())")
    check("a graph break gives two graphs", n == 2, f"{n}")

    grads_vs_eager(lambda: MLP(ckpt=True), x, y, "checkpointed block")
    hop = [d for d in STATS["graphs"][-1]["dynamo_nodes"] if "checkpoint" in d]
    check("checkpoint reaches the backend as a marked subgraph", bool(hop) and bool(STATS["regions"]),
          f"{hop} body={STATS['regions'][-1] if STATS['regions'] else None}")
    inlined = not any("checkpoint" in o for o in STATS["graphs"][-1]["aten_ops"])
    check("make_fx inlines the checkpointed body (marker lost after tracing)", inlined)

    mask = (torch.rand(5, 2, generator=torch.Generator().manual_seed(3)) > 0.4).double()
    grads_vs_eager(lambda: MLP(mask=mask), x, y, "restrict (partial tensor)")
    kept = "torch_sql.restrict.default" in STATS["graphs"][-1]["aten_ops"]
    check("restrict survives Dynamo and make_fx as one node", kept)

    # Guards and recompiles.
    torch._dynamo.reset()  # compiled graphs are reused across MLP instances otherwise
    m = MLP()
    c = torch.compile(m, backend=sql_backend, dynamic=False)
    before = STATS["compiles"]
    for b in (5, 7, 5, 7):
        c(torch.randn(b, 4, dtype=torch.float64), torch.randn(b, 2, dtype=torch.float64))
    check("dynamic=False: one compile per new batch size, reuse after", STATS["compiles"] - before == 2,
          f"compiles={STATS['compiles'] - before}")
    torch._dynamo.reset()
    c = torch.compile(MLP(), backend=sql_backend)
    c(torch.randn(5, 4, dtype=torch.float64), torch.randn(5, 2, dtype=torch.float64))
    try:
        c(torch.randn(7, 4, dtype=torch.float64), torch.randn(7, 2, dtype=torch.float64))
        r = "no error"
    except Exception as e:  # noqa: BLE001
        r = f"{type(e).__name__}: " + next((l for l in str(e).splitlines() if "symbolic" in l), str(e)[:90]).strip()
    check("default dynamic: second size arrives symbolic", "symbolic" in r, r)

    # Cost per step.
    torch._dynamo.reset()
    m = MLP()
    c = torch.compile(m, backend=sql_backend, dynamic=False)
    c(x, y).backward()
    q0, t0 = STATS["queries"], time.perf_counter()
    for _ in range(10):
        c(x, y).backward()
    sql_ms = (time.perf_counter() - t0) / 10 * 1e3
    nq = (STATS["queries"] - q0) / 10
    e = MLP()
    t0 = time.perf_counter()
    for _ in range(10):
        e(x, y).backward()
    eager_ms = (time.perf_counter() - t0) / 10 * 1e3
    torch._dynamo.reset()  # Dynamo's own cost: a backend that runs the FX graph as is
    d = torch.compile(MLP(), backend=lambda gm, ex: gm.forward, dynamic=False)
    d(x, y).backward()
    t0 = time.perf_counter()
    for _ in range(10):
        d(x, y).backward()
    dynamo_ms = (time.perf_counter() - t0) / 10 * 1e3
    compile_ms = [round(g["compile_ms"], 1) for g in STATS["graphs"]]
    print(f"      per step: SQL {sql_ms:.1f} ms ({nq:.0f} queries), eager {eager_ms:.3f} ms, "
          f"Dynamo with a pass-through backend {dynamo_ms:.3f} ms; "
          f"compile ms per graph: {compile_ms}")

    # DDP, in two processes.
    import subprocess
    port = 29500 + os.getpid() % 1000
    procs = [subprocess.Popen([sys.executable, __file__, "ddp", str(r), str(port)]) for r in (0, 1)]
    codes = [p.wait(timeout=300) for p in procs]
    check("DDP processes exit cleanly", codes == [0, 0], str(codes))

    if FAILED:
        print(f"\n{len(FAILED)} check(s) failed: {FAILED}")
        sys.exit(1)
    print("\nAll S6 checks pass.")


if __name__ == "__main__":
    main()
