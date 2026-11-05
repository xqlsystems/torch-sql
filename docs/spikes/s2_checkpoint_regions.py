"""Spike S2: how can torch-sql replace ``torch.utils.checkpoint`` cleanly?

Question (design.md §11.2, §16 S2). Samudra checkpoints every layer of its
U-Net. On ``device="sql"``, torch-sql doesn't want PyTorch's recomputation at
all, because ddx writes the backward pass. It only needs to know where each
checkpointed block starts and ends, and which recorded ops fall inside it, so
that it can hand ddx a *region* (a named relation inside which nothing is
saved except what a policy names). Which mechanism finds those regions, for
both ``use_reentrant`` settings, for nested checkpoints, and for the ways
Samudra calls checkpoint?

How Samudra calls it (commit 4f86be4):

- ``samudra.models.modules.unet_backbone`` (the production U-Net):
  ``torch.utils.checkpoint.checkpoint(layer, fts, use_reentrant=False)``,
  looked up through the module attribute at call time.
- ``samudra_multi`` and ``samudra_mini``: ``apply_activation_checkpointing``
  from ``torch.distributed.algorithms._checkpoint.checkpoint_wrapper``, which
  wraps modules in ``CheckpointWrapper``. That file binds
  ``from torch.utils.checkpoint import checkpoint as torch_utils_checkpoint``
  when it is imported.

A third style, ``from torch.utils.checkpoint import checkpoint`` at the top of
a user's module, is common elsewhere and is tested too.

The toy environment here is a wrapper tensor subclass on a renamed
``PrivateUse1`` device ``"sql"`` whose ``__torch_dispatch__`` records each ATen
op with the region stack current at that moment, then computes it eagerly on
a CPU tensor it holds. That is all the capture torch-sql's device needs for
this question (design §9.1).

Run::

    uv run --project docs/spikes python docs/spikes/s2_checkpoint_regions.py

Findings (torch 2.15.0, CPU):

0. How the ``"sql"`` device is set up decides whether autograd and
   ``torch.compile`` work at all, before checkpoint is even involved.
   ``rename_privateuse1_backend("sql")`` alone breaks both: a wrapper tensor
   on ``"sql"`` fails in checkpoint (no ``torch.sql`` module), and Dynamo
   fails asking the accelerator for its device. Adding a Python device
   module, as torchax does, is worse: autograd aborts the process in C++
   ("PyTorch is not linked with support for sql devices"), and Dynamo fails
   the same way. PyTorch's experimental
   ``torch.utils.backend_registration._setup_privateuseone_for_python_backend("sql")``,
   which also registers a Python device guard and hooks, makes both work: an
   eager checkpoint over a wrapper tensor on ``"sql"`` runs, and Dynamo
   compiles a checkpointed function. This matters beyond S2: spikes S1 and S6
   need it too, and it is private and marked experimental (P5).
1. On the meta device (design §9.1), PyTorch's own checkpoint runs on the toy
   tensor for both ``use_reentrant`` values, and dispatch sees each op once in
   the forward pass, with nothing extra.
2. A ``TorchFunctionMode`` never sees ``checkpoint``: it isn't an overridable
   ``torch`` function. The mode sees only the ops inside it.
3. Replacing the module attribute ``torch.utils.checkpoint.checkpoint`` while
   the environment is active catches Samudra's production call style, for
   both ``use_reentrant`` values, and nested checkpoints give nested regions.
   It misses names bound before the patch: ``from torch.utils.checkpoint
   import checkpoint``, and ``CheckpointWrapper``.
4. ``CheckpointWrapper`` keeps its checkpoint function on each instance
   (``checkpoint_fn``), so ``.to("sql")`` can replace it module by module.
   That catches ``apply_activation_checkpointing``, for both implementations.
   Region names should drop ``._checkpoint_wrapped_module``, as
   ``state_dict`` keys do.
5. Detecting checkpoint from inside ``__torch_dispatch__`` works for
   ``use_reentrant=False`` only, through the private
   ``torch._C._autograd._top_saved_tensors_default_hooks``: while a block
   runs, the top pack hook is a closure from ``torch.utils.checkpoint``. It
   sees every call style but gives no region name, and reentrant checkpoint
   leaves no such trace. It works as a guard: an op recorded under
   checkpoint's hook with no region open means a checkpoint torch-sql
   missed, and torch-sql should raise (P3). The guard stays quiet when the
   replacement caught everything.
6. ``nn.Module`` forward hooks see every module call but can't tell a
   checkpointed call from a plain one.

Under ``torch.compile``, Dynamo records a checkpointed call as one
``torch.ops.higher_order.tag_activation_checkpoint`` node (finding 0's
probe), so the compile backend gets regions from the graph and needs none of
this.

Recommendation: while the environment is active, torch-sql replaces
``torch.utils.checkpoint.checkpoint`` with its own function, which opens a
region named for the block's module path, calls the block directly (no
recomputation, RNG or device bookkeeping, since ddx writes the backward pass)
and closes the region. ``.to("sql")`` also replaces ``checkpoint_fn`` on
every ``CheckpointWrapper``. Dispatch raises if checkpoint's saved-tensor hook
is active with no region open, which catches early-bound calls with
``use_reentrant=False``; an early-bound reentrant call is the one case left
undetected, and Samudra makes none.
"""

import functools
import subprocess
import sys
import textwrap
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    CheckpointWrapper,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.overrides import TorchFunctionMode
from torch.utils._pytree import tree_map
from torch.utils.checkpoint import checkpoint as early_bound_checkpoint

FAILURES = []
DEVICE = "meta"  # the toy lives on meta, as torchax's tensors do (design §9.1)


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


# The toy environment: records (op, region path) and computes eagerly.

RECORD = []
REGIONS = []  # stack of region names
DETECT = {"on": False, "unclaimed": 0}


def checkpoint_hook_active():
    """Whether non-reentrant checkpoint's pack hook is on top (private API)."""
    top = torch._C._autograd._top_saved_tensors_default_hooks(False)
    return top is not None and top[0].__module__ == "torch.utils.checkpoint"


class SqlTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, elem):
        return torch.Tensor._make_wrapper_subclass(
            cls, elem.shape, dtype=elem.dtype, device=DEVICE,
            requires_grad=elem.requires_grad,
        )

    def __init__(self, elem):
        self.elem = elem.detach()

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if DETECT["on"] and checkpoint_hook_active() and not REGIONS:
            DETECT["unclaimed"] += 1
        RECORD.append((func.__name__, tuple(REGIONS)))
        unwrap = lambda t: t.elem if isinstance(t, SqlTensor) else t
        out = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs or {}))
        wrap = lambda t: SqlTensor(t) if isinstance(t, torch.Tensor) else t
        return tree_map(wrap, out)


@contextmanager
def region(name):
    REGIONS.append(name)
    try:
        yield
    finally:
        REGIONS.pop()


_CHECKPOINT_KWARGS = (
    "use_reentrant", "preserve_rng_state", "context_fn", "determinism_check",
    "debug", "early_stop", "respect_saved_tensors_hooks",
)
NAMES = {}  # id(module) -> module path, filled from named_modules


def sql_checkpoint(function, *args, **kwargs):
    """torch-sql's replacement: a region around a direct call."""
    for k in _CHECKPOINT_KWARGS:
        kwargs.pop(k, None)
    name = NAMES.get(id(function), getattr(function, "__name__", "block"))
    with region(name):
        return function(*args, **kwargs)


# Models in each call style.

class Block(nn.Module):
    def __init__(self, inner=None):
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 3, padding=1)
        self.inner = inner

    def forward(self, x):
        x = F.gelu(self.conv(x))
        if self.inner is not None:  # a nested checkpoint
            x = torch.utils.checkpoint.checkpoint(self.inner, x, use_reentrant=False)
        return x


class Net(nn.Module):
    def __init__(self, style, use_reentrant, nested=False):
        super().__init__()
        self.layers = nn.ModuleList(
            [Block(Block() if nested else None), Block()]
        )
        self.style, self.use_reentrant = style, use_reentrant
        if style == "wrapper":
            impl = CheckpointImpl.REENTRANT if use_reentrant else CheckpointImpl.NO_REENTRANT
            apply_activation_checkpointing(
                self, check_fn=lambda m: isinstance(m, Block),
                checkpoint_wrapper_fn=functools.partial(
                    checkpoint_wrapper, checkpoint_impl=impl))

    def forward(self, x):
        for layer in self.layers:
            if self.style == "attribute":  # Samudra's unet_backbone
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, use_reentrant=self.use_reentrant)
            elif self.style == "early_bound":
                x = early_bound_checkpoint(layer, x, use_reentrant=self.use_reentrant)
            else:  # "wrapper": CheckpointWrapper calls checkpoint itself
                x = layer(x)
        return x


def run(net):
    RECORD.clear()
    x = SqlTensor(torch.randn(1, 2, 4, 4, requires_grad=True))
    net(x)
    return list(RECORD)


def regions_of(record):
    """Region path -> number of ops recorded directly in it."""
    out = {}
    for _, path in record:
        if path:
            out[path] = out.get(path, 0) + 1
    return out


@contextmanager
def patched_attribute():
    original = torch.utils.checkpoint.checkpoint
    torch.utils.checkpoint.checkpoint = sql_checkpoint
    try:
        yield
    finally:
        torch.utils.checkpoint.checkpoint = original


def replace_wrapper_fns(net):
    for name, m in net.named_modules():
        # state_dict keys drop the wrapper's prefix; region names do too
        NAMES[id(m)] = name.replace("._checkpoint_wrapped_module", "")
        if isinstance(m, CheckpointWrapper):
            m.checkpoint_fn = sql_checkpoint


# Finding 0's probe and the setups it tries: (eager works, compile works).
SETUPS = {
    "rename": (False, False),                  # rename_privateuse1_backend only
    "rename+module": (False, False),           # plus a Python device module, as torchax does
    "python_backend": (True, True),            # _setup_privateuseone_for_python_backend
}
DEVICE_PROBE = """
    import torch, torch.utils.checkpoint
    from torch.utils._pytree import tree_map
    setup = {setup!r}
    if setup == "python_backend":
        from torch.utils.backend_registration import _setup_privateuseone_for_python_backend
        _setup_privateuseone_for_python_backend("sql")
    else:
        torch.utils.rename_privateuse1_backend("sql")
    if setup == "rename+module":
        class M:
            is_available = staticmethod(lambda: True)
            device_count = staticmethod(lambda: 1)
            current_device = staticmethod(lambda: 0)
            get_rng_state = staticmethod(lambda: torch.empty(0))
            set_rng_state = staticmethod(lambda s: None)
        torch._register_device_module("sql", M)

    class T(torch.Tensor):
        @staticmethod
        def __new__(cls, e):
            return torch.Tensor._make_wrapper_subclass(
                cls, e.shape, dtype=e.dtype, device="sql", requires_grad=True)
        def __init__(self, e):
            self.e = e
        @classmethod
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
            u = lambda t: t.e if isinstance(t, T) else t
            out = func(*tree_map(u, args), **tree_map(u, kwargs or {{}}))
            return tree_map(lambda t: T(t) if isinstance(t, torch.Tensor) else t, out)

    try:
        torch.utils.checkpoint.checkpoint(torch.sin, T(torch.randn(2)), use_reentrant=False)
        print("EAGER OK")
    except Exception as e:
        print("EAGER", type(e).__name__, e)
    graphs = []
    def backend(gm, inputs):
        graphs.append(gm)
        return gm.forward
    f = lambda x: torch.utils.checkpoint.checkpoint(torch.sin, x, use_reentrant=False)
    torch.compile(f, backend=backend)(torch.randn(2, requires_grad=True))
    names = [str(n.target) for g in graphs for n in g.graph.nodes]
    print("COMPILE OK", " ".join(n for n in names if "checkpoint" in n))
"""


def main():
    torch.manual_seed(0)

    # Finding 0: how torch-sql sets up the "sql" device decides whether
    # autograd, checkpoint and torch.compile work at all. Each setup runs in a
    # fresh process: an eager checkpoint over a wrapper tensor on "sql", then
    # torch.compile of a checkpointed function on plain CPU tensors.
    for setup, works in SETUPS.items():
        res = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(DEVICE_PROBE.format(setup=setup))],
            capture_output=True, text=True)
        out = res.stdout + res.stderr
        eager = "EAGER OK" in out
        compiled = "COMPILE OK tag_activation_checkpoint" in out
        why = [l for l in out.splitlines() if "Error" in l or "terminate" in l][:1]
        check(f"0. setup {setup!r}: eager {'works' if works[0] else 'fails'}, "
              f"compile {'works' if works[1] else 'fails'}",
              (eager, compiled) == works, why[0][:90] if why else "")

    # Finding 1: on the meta device, PyTorch's own checkpoint runs on the toy
    # tensor, and dispatch sees each op once during the forward pass.
    for ur in (True, False):
        rec = run(Net("attribute", ur))
        ops = [op for op, _ in rec]
        check(f"1. meta-device toy, checkpoint(use_reentrant={ur}) runs",
              ops == ["convolution.default", "gelu.default"] * 2, f"ops: {ops}")

    # Finding 2: TorchFunctionMode doesn't see checkpoint.
    seen = []

    class Log(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            seen.append(getattr(func, "__name__", str(func)))
            return func(*args, **(kwargs or {}))

    with Log():
        run(Net("attribute", False))
    check("2. TorchFunctionMode never sees checkpoint",
          not any("checkpoint" in s for s in seen), f"saw {sorted(set(seen))[:8]}...")

    # Finding 3: patching the module attribute.
    expected = 2  # two checkpointed blocks
    for style in ("attribute", "early_bound", "wrapper"):
        for ur in (True, False):
            net = Net(style, ur)
            NAMES.update({id(m): n for n, m in net.named_modules()})
            with patched_attribute():
                found = regions_of(run(net))
            caught = len(found) == expected
            want = style == "attribute"
            check(f"3. patched attribute, {style}, use_reentrant={ur}: "
                  f"{'caught' if want else 'missed'}", caught == want,
                  f"regions {sorted(found)}")
    net = Net("attribute", False, nested=True)
    NAMES.update({id(m): n for n, m in net.named_modules()})
    with patched_attribute():
        found = regions_of(run(net))
    nested_ok = ("layers.0", "layers.0.inner") in found and ("layers.0",) in found
    check("3. patched attribute handles nesting", nested_ok, f"regions {sorted(found)}")

    # Finding 4: replacing CheckpointWrapper.checkpoint_fn per instance.
    for ur in (True, False):
        net = Net("wrapper", ur)
        replace_wrapper_fns(net)
        found = regions_of(run(net))
        check(f"4. replaced checkpoint_fn, wrapper, use_reentrant={ur}",
              len(found) == expected, f"regions {sorted(found)}")

    # Finding 5: detection inside dispatch, through a private API.
    DETECT["on"] = True
    for style in ("attribute", "early_bound", "wrapper"):
        for ur in (True, False):
            DETECT["unclaimed"] = 0
            run(Net(style, ur))
            detected = DETECT["unclaimed"] > 0
            check(f"5. hook detection, {style}, use_reentrant={ur}: "
                  f"{'seen' if not ur else 'not seen'}", detected == (not ur),
                  f"{DETECT['unclaimed']} ops under checkpoint's hook")
    DETECT["unclaimed"] = 0
    net = Net("early_bound", False)
    with patched_attribute():
        run(net)
    check("5. guard flags the early-bound call the patch missed",
          DETECT["unclaimed"] > 0)
    DETECT["unclaimed"] = 0
    net = Net("attribute", False)
    with patched_attribute():
        run(net)
    check("5. guard is quiet when the patch caught everything",
          DETECT["unclaimed"] == 0)
    DETECT["on"] = False

    # Finding 6: module hooks see calls but not whether they're checkpointed.
    calls = []
    h = nn.modules.module.register_module_forward_pre_hook(
        lambda m, a: calls.append((type(m).__name__, checkpoint_hook_active())))
    try:
        run(Net("attribute", True))
    finally:
        h.remove()
    blocks = [c for c in calls if c[0] == "Block"]
    check("6. module hooks see Block calls, with nothing public marking checkpoint",
          len(blocks) == 2 and not any(flag for _, flag in blocks), f"{blocks}")

    print(f"\n{'all findings hold' if not FAILURES else f'{len(FAILURES)} failed'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
