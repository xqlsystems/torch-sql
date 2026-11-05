"""Spike S5: which ATen ops does Samudra's optimizer path call, and do
PyTorch's decompositions cover them?

Question (design.md §11.5, §16 S5). ``torch.optim`` and
``clip_grad_norm_`` are ordinary tensor code, so on ``device="sql"`` their ops
reach torch-sql's dispatch like any other. By default they use ``_foreach_*``
ops, which apply one op to a list of tensors at once. Which ops arrive, in
which path, and what must torch-sql lower itself?

Samudra's optimizer path (``samudra.train`` and ``samudra.utils.ema`` at
commit 4f86be4): ``torch.optim.Adam(params, lr=6e-4)`` with every other
argument at its default; ``CosineAnnealingLR`` stepped once per epoch;
``clip_grad_norm_(params, 1.0, error_if_nonfinite=True)`` before each step;
and an EMA of the weights after each step, written with ``torch.min`` on
0-dim CPU tensors and an in-place ``sub_``.

The toy environment is a wrapper tensor subclass on a ``"sql"`` device set
up with ``_setup_privateuseone_for_python_backend`` (spike S2, finding 0).
Its ``__torch_dispatch__`` records each ATen op and computes it on a CPU
tensor it holds. Only ops that touch a ``"sql"`` tensor are recorded, which
is what torch-sql would see. A small dispatch mode stands in for the
environment's handling of ops that create ``"sql"`` tensors.

Run::

    uv run --project docs/spikes python docs/spikes/s5_foreach_ops.py

Findings (torch 2.15.0, CPU):

1. On a torch-sql tensor, the optimizers take the per-tensor loop by
   default, not ``_foreach_*``. PyTorch's default checks
   ``type(p) in (Tensor, Parameter)`` as well as the device, so a tensor
   subclass never qualifies, whatever its device. ``foreach=True`` and
   ``clip_grad_norm_(foreach=True)`` still reach torch-sql as foreach ops.
   ``fused=True`` and ``capturable=True`` create state on the ``"sql"``
   device with factory ops (``empty``, ``zeros``), which only the
   environment's dispatch mode can catch (design §9.1); with it, ``fused``
   calls ``_fused_adam_``, which torch-sql refuses (design §11.5), and
   ``capturable`` runs.
2. PyTorch has no decomposition for any ``_foreach_*`` op: none of the 1,100
   entries in ``torch._decomp.decomposition_table`` is one, so torch-sql must
   lower them. One generic rule covers the 19 seen here: the per-element op
   is the overload of ``aten.<op>`` whose argument types are the foreach op's
   with each list replaced by its element type, applied element by element.
   The one exception is ``_foreach_norm``, whose per-element op is
   ``linalg_vector_norm``. Applied inside dispatch, the rule reproduces
   Adam, AdamW and SGD bit for bit, with and without weight decay and
   ``capturable``.
3. Underneath are in-place pointwise ops (``mul_``, ``add_``, ``lerp_``,
   ``addcmul_``, ``addcdiv_``, ...). Each is the in-place form of a
   functional op that is core ATen or has a decomposition, so torch-sql's
   in-place rule (design §8.6) and PyTorch's decompositions cover them.
4. With ``capturable=False`` (the default), Adam keeps ``step`` as a 0-dim
   CPU tensor and computes the bias corrections in Python. They reach
   torch-sql as Python scalars that change every step (in ``div`` and
   ``addcdiv_``). So do the learning rate after each ``CosineAnnealingLR``
   step, which stays a Python float, and the EMA's decay, a 0-dim CPU tensor
   (in ``mul``, ``mul_`` and ``sub``). A program cached on those values
   would be rebuilt every step (design §9.6): torch-sql must treat scalar
   arguments and 0-dim CPU tensors as parameters of the cached program, not
   as literals. ``capturable=True`` keeps them on ``"sql"`` instead, at 57 ops
   per step against 21.
5. ``clip_grad_norm_(..., error_if_nonfinite=True)``, as Samudra calls it,
   reads two values back from ``"sql"`` (``_local_scalar_dense`` after
   ``isnan``/``isinf``/``logical_or``). On the device that is a sync in the
   middle of every step, before Adam. Without ``error_if_nonfinite``,
   clipping reads nothing back; neither do Adam, AdamW, SGD or the EMA.
6. Samudra's whole optimizer path (Adam, cosine schedule, clipping, EMA)
   gives the same results on the toy ``"sql"`` tensor as on plain CPU
   tensors, and needs 18 ops once foreach and in-place ops are rewritten:
   11 core ops (``add``, ``mul``, ``sub``, ``div``, ``sqrt``,
   ``reciprocal``, ``clamp``, ``isnan``, ``isinf``, ``logical_or``,
   ``_to_copy``), six with PyTorch decompositions (``addcdiv``, ``addcmul``,
   ``lerp``, ``linalg_vector_norm``, ``stack``, ``zeros_like``), and the
   read-back ``_local_scalar_dense``.
"""

import contextlib
import sys

import torch
from torch.utils._pytree import tree_map
from torch.utils.backend_registration import _setup_privateuseone_for_python_backend

_setup_privateuseone_for_python_backend("sql")
torch.set_default_dtype(torch.float64)

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


# The toy environment.

LOG = []  # (op, scalar arguments) for every op that touched a "sql" tensor
UNROLL = {"on": False}  # apply the generic foreach rule inside dispatch


def is_core(func):
    return torch.Tag.core in func.tags


def functional(func):
    """The functional op an in-place op is a form of, or the op itself."""
    name = func._schema.name.split("::")[1]
    if name.endswith("_") and not name.startswith("_"):
        packet = getattr(torch.ops.aten, name[:-1], None)
        if packet is not None and func._overloadname in packet.overloads():
            return getattr(packet, func._overloadname)
    return func


# The one foreach op seen whose per-element op has another name.
NORM = torch.ops.aten._foreach_norm.Scalar


def foreach_base(func):
    """The per-element op for a foreach op: aten._foreach_mul_.Scalar is
    aten.mul_.Scalar. The overload is the one whose argument types are the
    foreach op's with each list type replaced by its element type."""
    if func == NORM:
        return torch.ops.aten.linalg_vector_norm.default
    name = func._schema.name.split("::")[1]
    assert name.startswith("_foreach_")
    packet = getattr(torch.ops.aten, name[len("_foreach_"):])
    element = lambda t: str(t).removeprefix("List[").removesuffix("]")
    want = [element(a.type) for a in func._schema.arguments]
    for overload in packet.overloads():
        op = getattr(packet, overload)
        have = [str(a.type) for a in op._schema.arguments]
        if have[: len(want)] == want and all(
                a.has_default_value() for a in op._schema.arguments[len(want):]):
            return op
    raise LookupError(f"no per-element op for {func}")


def unroll_foreach(func, args, kwargs):
    """Apply the per-element op to each element of the foreach op's lists.

    The foreach op's arguments map onto the per-element op's in order; its
    names differ (``tensors1`` for ``end``, ``scalars`` for ``other``), and
    some arguments that are positional in one are keyword-only in the other.
    """
    base = foreach_base(func)
    if func == NORM:  # (self, ord, dtype): ord is positional in both, dtype isn't
        ord_ = args[1] if len(args) > 1 else kwargs.get("ord", 2)
        dtype = args[2] if len(args) > 2 else kwargs.get("dtype")
        return [base(x, ord_, dtype=dtype) for x in args[0]]
    given = [named for named in zip(func._schema.arguments, args)]
    given += [(a, kwargs[a.name]) for a in func._schema.arguments if a.name in kwargs]
    targets = list(base._schema.arguments)
    n = len(next(v for _, v in given if isinstance(v, (list, tuple))))
    out = []
    for i in range(n):
        each = {}
        for (_, v), target in zip(given, targets):
            each[target.name] = v[i] if isinstance(v, (list, tuple)) else v
        out.append(base(**each))
    return None if not func._schema.returns else out


class SqlTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, elem, requires_grad=False):
        return torch.Tensor._make_wrapper_subclass(
            cls, elem.shape, dtype=elem.dtype, device="sql", requires_grad=requires_grad)

    def __init__(self, elem, requires_grad=False):
        self.elem = elem

    def __repr__(self):
        return f"SqlTensor({self.elem})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        scalars = [a for a in torch.utils._pytree.tree_leaves((args, kwargs))
                   if isinstance(a, (int, float)) and not isinstance(a, bool)
                   or isinstance(a, torch.Tensor) and not isinstance(a, SqlTensor)]
        LOG.append((func, scalars))
        if UNROLL["on"] and func._schema.name.startswith("aten::_foreach_"):
            return unroll_foreach(func, args, kwargs)
        unwrap = lambda t: t.elem if isinstance(t, SqlTensor) else t
        uargs, ukwargs = tree_map(unwrap, args), tree_map(unwrap, kwargs)
        out = func(*uargs, **ukwargs)
        if not func._schema.returns:  # in-place foreach ops return nothing
            return None
        if func._schema.name.endswith("_"):
            return args[0]
        return tree_map(lambda t: SqlTensor(t) if isinstance(t, torch.Tensor) else t, out)


class Env(torch.utils._python_dispatch.TorchDispatchMode):
    """The device environment's part here: ops that create a "sql" tensor."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = dict(kwargs or {})
        device = kwargs.get("device")
        if device is not None and torch.device(device).type == "sql":
            kwargs["device"] = "cpu"
            LOG.append((func, []))
            return SqlTensor(func(*args, **kwargs))
        return func(*args, **kwargs)


SHAPES = [(4, 3), (3,), (2, 2)]


def make_params(seed=0, sql=True):
    g = torch.Generator().manual_seed(seed)
    ps = []
    for s in SHAPES:
        t = torch.randn(s, generator=g)
        p = SqlTensor(t, requires_grad=True) if sql else t.clone().requires_grad_(True)
        ps.append(p)
    return ps


def set_grads(ps, step, sql=True):
    g = torch.Generator().manual_seed(100 + step)
    for p in ps:
        t = torch.randn(p.shape, generator=g)
        p.grad = SqlTensor(t) if sql else t


def values(ps):
    return [p.elem if isinstance(p, SqlTensor) else p.detach() for p in ps]


class Ema:
    """Samudra's EMATracker update, reduced to the ops it runs."""

    def __init__(self, ps, decay=0.999):
        self.decay = torch.tensor(decay, dtype=torch.float32)
        self.num_updates = torch.tensor(0, dtype=torch.int)
        self.params = [p.detach().clone() for p in ps]

    def __call__(self, ps):
        self.num_updates += 1
        decay = torch.min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        self.cur_decay = decay
        with torch.no_grad():
            for i, p in enumerate(ps):
                self.params[i] = self.params[i].type_as(p)
                self.params[i].sub_((1.0 - decay) * (self.params[i] - p))


def samudra_step(ps, opt, sched, ema, step, sql=True):
    set_grads(ps, step, sql)
    torch.nn.utils.clip_grad_norm_(ps, 1.0, error_if_nonfinite=True)
    opt.step()
    opt.zero_grad()
    ema(ps)
    sched.step()  # once per epoch in Samudra; once per step here, to see lr change


CONFIGS = {
    "Adam()": lambda ps: torch.optim.Adam(ps, lr=6e-4),
    "Adam(foreach=True)": lambda ps: torch.optim.Adam(ps, lr=6e-4, foreach=True),
    "Adam(foreach=False)": lambda ps: torch.optim.Adam(ps, lr=6e-4, foreach=False),
    "Adam(weight_decay)": lambda ps: torch.optim.Adam(ps, lr=6e-4, weight_decay=0.01),
    "Adam(weight_decay, foreach=True)": lambda ps: torch.optim.Adam(
        ps, lr=6e-4, weight_decay=0.01, foreach=True),
    "Adam(capturable)": lambda ps: torch.optim.Adam(ps, lr=6e-4, capturable=True),
    "Adam(capturable, foreach=True)": lambda ps: torch.optim.Adam(
        ps, lr=6e-4, capturable=True, foreach=True),
    "AdamW()": lambda ps: torch.optim.AdamW(ps, lr=6e-4),
    "AdamW(foreach=True)": lambda ps: torch.optim.AdamW(ps, lr=6e-4, foreach=True),
    "SGD(momentum)": lambda ps: torch.optim.SGD(ps, lr=1e-2, momentum=0.9),
    "SGD(momentum, foreach=True)": lambda ps: torch.optim.SGD(
        ps, lr=1e-2, momentum=0.9, foreach=True),
}

# Plain CPU tensors can't be capturable, so those configs are checked against
# the same math without it.
REFERENCE = {"Adam(capturable)": "Adam()", "Adam(capturable, foreach=True)": "Adam()"}


def run_optimizer(make_opt, sql=True, steps=3, env=True):
    ps = make_params(sql=sql)
    opt = make_opt(ps)
    per_step = []
    for step in range(steps):
        set_grads(ps, step, sql)
        LOG.clear()
        if env and sql:
            with Env():
                opt.step()
        else:
            opt.step()
        per_step.append(list(LOG))
    return values(ps), per_step, opt


def op_names(log):
    return sorted({str(f) for f, _ in log})


def changing_scalars(per_step):
    """Ops whose scalar arguments differ between the second and third steps
    (the first also creates the optimizer's state)."""
    a, b = per_step[1], per_step[2]
    out = set()
    def same_value(x, y):
        if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
            return (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
                    and x.shape == y.shape and torch.equal(x, y))
        return x == y

    for (fa, sa), (fb, sb) in zip(a, b):
        same = len(sa) == len(sb) and all(same_value(x, y) for x, y in zip(sa, sb))
        if fa == fb and not same:
            out.add(str(fa))
    return sorted(out)


def main():
    decomp = torch._decomp.decomposition_table

    # Finding 1: which path the optimizers take on a "sql" tensor subclass.
    from torch.optim.optimizer import _default_to_fused_or_foreach
    _, per_step, _ = run_optimizer(CONFIGS["Adam()"])
    names = op_names(per_step[0])
    check("1. Adam() on a 'sql' subclass takes the per-tensor loop",
          not any("_foreach_" in n for n in names), f"{len(names)} distinct ops")
    _, foreach_sql = _default_to_fused_or_foreach(make_params(), False, False)
    check("1. because type(p) must be exactly Tensor or Parameter",
          foreach_sql is False)
    for name in ("fused", "capturable"):
        make = lambda ps: torch.optim.Adam(ps, **{name: True})
        try:
            run_optimizer(make, env=False)
            check(f"1. Adam({name}=True) needs the factory mode", False, "ran")
        except Exception as e:  # noqa: BLE001
            check(f"1. Adam({name}=True) needs the factory mode",
                  "empty" in str(e) or "zeros" in str(e), str(e).splitlines()[0][:70])
        _, per_step, _ = run_optimizer(make)
        names = op_names(per_step[0])
        if name == "fused":
            check("1. with it, Adam(fused=True) calls _fused_adam_",
                  any("_fused_adam" in n for n in names))
        else:
            check("1. with it, Adam(capturable=True) runs", len(names) > 0,
                  f"{len(names)} distinct ops")

    # Finding 2: no foreach decompositions; the generic rule is exact.
    every = set()
    for name, make in CONFIGS.items():
        reference, _, _ = run_optimizer(CONFIGS[REFERENCE.get(name, name)], sql=False)
        UNROLL["on"] = False
        native, per_step, _ = run_optimizer(make)
        UNROLL["on"] = True
        unrolled, unrolled_steps, _ = run_optimizer(make)
        UNROLL["on"] = False
        foreach = sorted({str(f) for s in per_step for f, _ in s if "_foreach_" in str(f)})
        every.update(f for s in per_step for f, _ in s)
        exact = all(torch.equal(x, y) for x, y in zip(unrolled, native))
        close = all(torch.allclose(x, y, rtol=0, atol=1e-15)
                    for x, y in zip(native, reference))
        check(f"2. {name}: foreach rule reproduces it exactly", exact and close,
              f"{len(foreach)} foreach ops")
    clip_ops = []
    for foreach in (None, True, False):
        ps = make_params()
        set_grads(ps, 0)
        LOG.clear()
        with Env():
            torch.nn.utils.clip_grad_norm_(ps, 1.0, foreach=foreach)
        clip_ops.append(list(LOG))
        every.update(f for f, _ in LOG)
    foreach_ops = sorted({str(f) for f in every if "_foreach_" in str(f)})
    print("      foreach ops seen:", ", ".join(foreach_ops))
    check("2. no foreach op has a decomposition",
          not any(f in decomp for f in every if "_foreach_" in str(f)))
    rule_ok = all(foreach_base(f) is not None
                  for f in every if "_foreach_" in str(f))
    check("2. every foreach op seen maps to a per-element op by the rule", rule_ok)

    # Finding 3: what's underneath is the in-place form of core ops.
    under = {functional(f) for f in every if "_foreach_" not in str(f)}
    under |= {functional(foreach_base(f)) for f in every if "_foreach_" in str(f)}
    non_core = sorted(str(f) for f in under
                      if not is_core(f) and f not in decomp)
    print("      neither core nor decomposable:", ", ".join(non_core) or "none")
    check("3. after the in-place rule, every op is core or decomposes",
          not non_core)

    # Finding 4: scalars that change each step.
    _, per_step, _ = run_optimizer(CONFIGS["Adam()"])
    changing = changing_scalars(per_step)
    check("4. Adam(): scalar arguments change between steps",
          len(changing) > 0, ", ".join(changing))
    _, per_step, _ = run_optimizer(CONFIGS["Adam(capturable)"])
    cap = changing_scalars(per_step)
    cap_ops = len(per_step[1])
    _, plain, _ = run_optimizer(CONFIGS["Adam()"])
    check("4. Adam(capturable=True): no scalar changes, more ops",
          not cap and cap_ops > len(plain[1]),
          f"{cap_ops} ops against {len(plain[1])}")

    # Finding 5: values read back from "sql".
    syncs = lambda log: sum(1 for f, _ in log if "_local_scalar_dense" in str(f))
    ps = make_params()
    set_grads(ps, 0)
    LOG.clear()
    with Env():
        torch.nn.utils.clip_grad_norm_(ps, 1.0, error_if_nonfinite=True)
    check("5. clip_grad_norm_(error_if_nonfinite=True) reads values back",
          syncs(LOG) > 0, f"{syncs(LOG)} reads of {[str(f) for f, _ in LOG if 'isnan' in str(f) or 'isinf' in str(f) or 'logical' in str(f)]}")
    check("5. clip_grad_norm_ without it reads nothing back",
          all(syncs(log) == 0 for log in clip_ops))
    for name, make in CONFIGS.items():
        _, per_step, _ = run_optimizer(make)
        if syncs(per_step[1]):
            check(f"5. {name} reads nothing back", False, f"{syncs(per_step[1])}")

    # Finding 6: Samudra's whole optimizer path, and the ops it needs.
    results, logs = [], []
    for sql in (True, False):
        ps = make_params(sql=sql)
        opt = torch.optim.Adam(ps, lr=6e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=70)
        ema = Ema(ps)
        steps = []
        for step in range(3):
            LOG.clear()
            with Env() if sql else contextlib.nullcontext():
                samudra_step(ps, opt, sched, ema, step, sql)
            steps.append(list(LOG))
        results.append(values(ps) + values(ema.params))
        logs = steps if sql else logs
    check("6. Samudra's path on 'sql' matches plain PyTorch",
          all(torch.allclose(a, b, rtol=0, atol=1e-15) for a, b in zip(*results)))
    lrs = {type(g["lr"]).__name__ for g in opt.param_groups}
    check("6. the scheduler keeps lr a Python float", lrs == {"float"}, f"{lrs}")
    changing = changing_scalars(logs)
    print("      ops with per-step scalar values:", ", ".join(changing))
    needed = set()
    for f, _ in (e for s in logs for e in s):
        f = foreach_base(f) if "_foreach_" in str(f) else f
        needed.add(functional(f))
    need = sorted(str(f) for f in needed)
    print(f"      Samudra's optimizer path needs {len(need)} ops:")
    for n in need:
        print("        ", n)
    decomposed = sorted(str(f) for f in needed if not is_core(f) and f in decomp)
    print("      of which decompose to core:", ", ".join(decomposed))
    check("6. each needed op is core ATen, decomposes, or is a read-back",
          all(is_core(f) or f in decomp or "_local_scalar_dense" in str(f)
              for f in needed))

    print(f"\n{'all findings hold' if not FAILURES else f'{len(FAILURES)} failed'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
