"""Random tensor programs, for property-based tests of the device.

A *program* is a list of steps over float64 *leaves* (tensors that require a
gradient). Each step applies one op from :data:`OPS` to earlier values, and
the last value is the program's output. The same program runs on PyTorch's
CPU, which is the oracle, and on ``device="sql"``; :func:`run` is the one
interpreter both use, so there is no second description of the program to
drift from the first (ddx's fuzzer does the same with a jaxpr).

Programs are drawn with Hypothesis (:func:`programs`). Shapes are inferred
while drawing by running each step on PyTorch's ``meta`` device, so every
drawn program is one PyTorch accepts.

Values are drawn in *data modes*, after ddx's fuzzer:

- ``smooth``: normal values, no ties, away from kinks. Gradients are
  compared here.
- ``ties``: values from a small set, so maxima, comparisons and ``relu``
  meet ties and zeros.
- ``extreme``: NaN of either sign, infinities, ``-0.0``, huge and tiny
  values. Only forward values are compared: torch-sql promises the same
  numbers as PyTorch (design §1.3 item 4), not the same subgradients.

The interpreter also records *screens*: whether an op met a tie or a kink
(where the derivative is a convention, not a fact), or an argument outside
its domain. A gradient comparison is skipped when a screen fires, and the
tests assert a retention floor so the screens can't quietly skip everything.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from hypothesis import strategies as st

F64 = torch.float64


@dataclass(frozen=True)
class Step:
    op: str
    args: tuple[int, ...]
    params: tuple = ()

    def __repr__(self):
        p = f", {self.params!r}" if self.params else ""
        return f"Step({self.op!r}, {self.args!r}{p})"


@dataclass
class Program:
    leaves: list[tuple[int, ...]]
    steps: list[Step]
    #: The values of the leaves, as nested lists of floats.
    data: list[list] = field(default_factory=list)
    mode: str = "smooth"

    def __repr__(self):
        lines = [f"Program(mode={self.mode!r}, leaves={self.leaves!r}, steps=["]
        lines += [f"    {s!r}," for s in self.steps]
        lines.append(f"], data={self.data!r})")
        return "\n".join(lines)


class Screens:
    """What the interpreter saw that makes a derivative a convention."""

    def __init__(self):
        self.kink = False
        self.domain = False
        #: A sum met values near overflow, where the order of the terms
        #: decides between a finite result, an infinity and a NaN.
        self.overflow = False
        #: A division by an exact zero, or its log. The infinity's sign is then the
        #: zero's, and a sum's zero has a sign its kernel chooses: PyTorch's
        #: own sum of [-0.0] is +0.0, its matrix product's -0.0.
        self.zero_division = False


def _real(*xs) -> bool:
    """Whether the screens can look at values: not while drawing, when the
    interpreter runs on ``meta`` tensors to infer shapes."""
    return not any(isinstance(x, torch.Tensor) and x.device.type == "meta" for x in xs)


def _tie(a: torch.Tensor, b) -> bool:
    return bool((a == b).any())


def _group_tie(x: torch.Tensor, dims) -> bool:
    """Whether two elements of one reduction group are equal maxima/minima."""
    if x.numel() == 0:
        return False
    if not dims:
        dims = list(range(x.dim()))
    m = x.amax(dim=dims, keepdim=True) if dims else x
    n = x.amin(dim=dims, keepdim=True) if dims else x
    return bool(((x == m).sum(dim=dims) > 1).any() or ((x == n).sum(dim=dims) > 1).any())


# ----------------------------------------------------------------------------
# The ops. Each takes the screens, its argument tensors and its params.


def _unary(fn, *, kink=None, domain=None):
    def apply(s, x):
        if kink is not None and _real(x) and kink(x):
            s.kink = True
        if domain is not None and _real(x) and not domain(x):
            s.domain = True
        return fn(x)

    return apply


def _near(x, points, eps=1e-3):
    return any(bool(((x - p).abs() < eps).any()) for p in points)


UNARY = {
    "neg": _unary(torch.neg),
    "exp": _unary(torch.exp),
    "tanh": _unary(torch.tanh),
    "sigmoid": _unary(torch.sigmoid),
    "sin": _unary(torch.sin),
    "cos": _unary(torch.cos),
    "abs": _unary(torch.abs, kink=lambda x: _near(x, [0])),
    "relu": _unary(torch.relu, kink=lambda x: _near(x, [0])),
    "sqrt": _unary(torch.sqrt, domain=lambda x: bool((x > 1e-3).all())),
    "rsqrt": _unary(torch.rsqrt, domain=lambda x: bool((x > 1e-3).all())),
    "log": _unary(torch.log, domain=lambda x: bool((x > 1e-3).all())),
    "reciprocal": _unary(torch.reciprocal, domain=lambda x: bool((x.abs() > 1e-3).all())),
    "square": _unary(lambda x: x**2),
    "cube": _unary(lambda x: x**3),
    "gelu": _unary(lambda x: torch.nn.functional.gelu(x, approximate="tanh")),
    # No gradient flows through these (design §11.3).
    "detach": _unary(torch.Tensor.detach),
    "no_grad_double": _unary(lambda x: _no_grad(lambda: x * 2.0)),
}


def _no_grad(fn):
    with torch.no_grad():
        return fn()


def _binary(fn, *, kink=None, domain=None):
    def apply(s, a, b):
        if kink is not None and _real(a, b) and kink(a, b):
            s.kink = True
        if domain is not None and _real(a, b) and not domain(a, b):
            s.domain = True
        return fn(a, b)

    return apply


def _near_tie(a, b, eps=1e-3):
    return bool(((a - b).abs() < eps).any())


BINARY = {
    "add": _binary(torch.add),
    "sub": _binary(torch.sub),
    "mul": _binary(torch.mul),
    "div": _binary(torch.div, domain=lambda a, b: bool((b.abs() > 1e-3).all())),
    "maximum": _binary(torch.maximum, kink=_near_tie),
    "minimum": _binary(torch.minimum, kink=_near_tie),
    "where_gt": _binary(lambda a, b: torch.where(a > b, a, b * 2.0), kink=_near_tie),
}


def _reduce(name):
    fn = {"sum": torch.sum, "mean": torch.mean, "amax": torch.amax, "amin": torch.amin}[name]

    def apply(s, x, dims, keepdim):
        if name in ("amax", "amin") and _real(x) and _group_tie(x, list(dims)):
            s.kink = True
        if name in ("amax", "amin"):
            return fn(x, dim=list(dims), keepdim=keepdim)
        return fn(x, dim=list(dims) or None, keepdim=keepdim)

    return apply


def _clamp(s, x, lo, hi):
    if _real(x) and _near(x, [lo, hi]):
        s.kink = True
    return torch.clamp(x, lo, hi)


def _scalar_op(s, x, kind, c):
    return {"add": x + c, "mul": x * c, "rsub": c - x}[kind]


def _softmax(s, x, dim, log):
    return torch.log_softmax(x, dim) if log else torch.softmax(x, dim)


def _layer_norm(s, x, k):
    return torch.nn.functional.layer_norm(x, x.shape[x.dim() - k :])


def _matmul_t(s, a):
    return a @ a.transpose(-1, -2)


def _matmul(s, a, b):
    return a @ b


def _permute(s, x, perm):
    return x.permute(*perm)


def _unsqueeze(s, x, d):
    return x.unsqueeze(d)


def _squeeze(s, x, d):
    return x.squeeze(d)


def _reshape(s, x, shape):
    return x.reshape(shape)


def _slice(s, x, d, start, stop):
    return x.narrow(d, start, stop - start)


def _select(s, x, d, i):
    return x.select(d, i)


def _expand(s, x, shape):
    return x.expand(*shape)


def _cat(s, a, b, d):
    return torch.cat([a, b], d)


OTHER = {
    "reduce_sum": _reduce("sum"),
    "reduce_mean": _reduce("mean"),
    "reduce_amax": _reduce("amax"),
    "reduce_amin": _reduce("amin"),
    "clamp": _clamp,
    "scalar": _scalar_op,
    "softmax": _softmax,
    "layer_norm": _layer_norm,
    "matmul_t": _matmul_t,
    "matmul": _matmul,
    "permute": _permute,
    "unsqueeze": _unsqueeze,
    "squeeze": _squeeze,
    "reshape": _reshape,
    "slice": _slice,
    "select": _select,
    "expand": _expand,
    "cat": _cat,
}

OPS = {**UNARY, **BINARY, **OTHER}


def run(program: Program, leaves: list[torch.Tensor], screens: Screens | None = None):
    """Every value of ``program``: its leaves, then each step's result."""
    s = screens or Screens()
    values = list(leaves)
    for step in program.steps:
        args = [values[i] for i in step.args]
        if step.op in SUMS and _real(*args) and any(_huge(a) for a in args):
            s.overflow = True
        if (
            step.op in ("reciprocal", "rsqrt", "div", "log")
            and _real(*args)
            and bool((args[-1] == 0).any())
        ):
            s.zero_division = True
        values.append(OPS[step.op](s, *args, *step.params))
    return values


#: Ops that add many terms, whose rounding near overflow depends on order.
SUMS = {"reduce_sum", "reduce_mean", "layer_norm", "softmax", "matmul", "matmul_t"}


def _huge(x: torch.Tensor) -> bool:
    x = x.detach()
    finite = x[torch.isfinite(x)]
    return bool(finite.numel()) and float(finite.abs().max()) > 1e150


# ----------------------------------------------------------------------------
# Drawing programs


def _shape_of(op, shapes, args, params):
    meta = [torch.empty(shapes[i], dtype=F64, device="meta") for i in args]
    with torch.no_grad():
        return tuple(OPS[op](Screens(), *meta, *params).shape)


def _factorizations(n: int):
    return [(a, n // a) for a in range(1, n + 1) if n % a == 0]


@st.composite
def _step(draw, shapes: list[tuple[int, ...]]):
    """One step over values of ``shapes``, or None if the draw doesn't fit."""
    k = len(shapes)
    i = draw(st.integers(0, k - 1))
    x = shapes[i]
    r = len(x)
    family = draw(
        st.sampled_from(["unary", "binary", "reduce", "nn", "shape", "shape", "linalg", "misc"])
    )
    if family == "unary":
        return Step(draw(st.sampled_from(sorted(UNARY))), (i,))
    if family == "binary":
        # A partner that broadcasts with x: x itself, another value of a
        # shape that broadcasts, or a reduction of x with keepdim.
        partners = [
            j for j, y in enumerate(shapes) if _broadcasts(x, y) and max(len(x), len(y)) <= 4
        ]
        j = draw(st.sampled_from(partners))
        a, b = draw(st.sampled_from([(i, j), (j, i)]))
        return Step(draw(st.sampled_from(sorted(BINARY))), (a, b))
    if family == "reduce":
        if r == 0:
            return None
        dims = tuple(sorted(draw(st.sets(st.integers(0, r - 1), max_size=r))))
        name = draw(st.sampled_from(["reduce_sum", "reduce_mean", "reduce_amax", "reduce_amin"]))
        if name in ("reduce_amax", "reduce_amin") and not dims:
            dims = tuple(range(r))
        return Step(name, (i,), (dims, draw(st.booleans())))
    if family == "nn":
        if r == 0:
            return None
        kind = draw(st.sampled_from(["softmax", "log_softmax", "layer_norm"]))
        if kind == "layer_norm":
            k = draw(st.integers(1, r))
            if math.prod(x[r - k :]) < 2:
                return None
            return Step("layer_norm", (i,), (k,))
        return Step("softmax", (i,), (draw(st.integers(-r, r - 1)), kind == "log_softmax"))
    if family == "linalg":
        if r not in (2, 3):
            return None
        partners = [
            j
            for j, y in enumerate(shapes)
            if len(y) == r and y[-2] == x[-1] and (r == 2 or y[0] == x[0])
        ]
        if partners and draw(st.booleans()):
            return Step("matmul", (i, draw(st.sampled_from(partners))))
        return Step("matmul_t", (i,))
    if family == "misc":
        kind = draw(st.sampled_from(["clamp", "scalar"]))
        if kind == "clamp":
            lo = draw(st.sampled_from([-1.0, -0.5, 0.0]))
            return Step("clamp", (i,), (lo, lo + draw(st.sampled_from([0.5, 1.0, 2.0]))))
        c = draw(st.sampled_from([-2.0, -1.0, 0.5, 3.0]))
        return Step("scalar", (i,), (draw(st.sampled_from(["add", "mul", "rsub"])), c))
    # shape ops
    kind = draw(
        st.sampled_from(
            ["permute", "unsqueeze", "squeeze", "reshape", "slice", "select", "expand", "cat"]
        )
    )
    if kind == "permute" and r >= 2:
        return Step("permute", (i,), (tuple(draw(st.permutations(range(r)))),))
    if kind == "unsqueeze" and r < 4:
        return Step("unsqueeze", (i,), (draw(st.integers(0, r)),))
    if kind == "squeeze" and 1 in x:
        return Step("squeeze", (i,), (draw(st.sampled_from([d for d in range(r) if x[d] == 1])),))
    if kind == "reshape" and r >= 1:
        n = math.prod(x)
        target = draw(
            st.sampled_from(
                [
                    (n,),
                    *_factorizations(n),
                    *[
                        (a, b, n // (a * b))
                        for a, b0 in _factorizations(n)
                        for b in range(1, b0 + 1)
                        if b0 % b == 0
                    ],
                ]
            )
        )
        return Step("reshape", (i,), (target,))
    if kind in ("slice", "select") and r >= 1:
        d = draw(st.integers(0, r - 1))
        if x[d] < 2:
            return None
        if kind == "select":
            return Step("select", (i,), (d, draw(st.integers(0, x[d] - 1))))
        start = draw(st.integers(0, x[d] - 1))
        stop = draw(st.integers(start + 1, x[d]))
        return Step("slice", (i,), (d, start, stop))
    if kind == "expand" and 1 in x:
        shape = tuple(draw(st.integers(1, 3)) if n == 1 else n for n in x)
        return Step("expand", (i,), (shape,))
    if kind == "cat" and r >= 1:
        d = draw(st.integers(0, r - 1))
        partners = [
            j
            for j, y in enumerate(shapes)
            if len(y) == r and all(y[e] == x[e] for e in range(r) if e != d)
        ]
        return Step(
            "cat",
            (
                i,
                draw(st.sampled_from(partners)),
            ),
            (d,),
        )
    return None


def _broadcasts(x, y) -> bool:
    for a, b in zip(reversed(x), reversed(y), strict=False):
        if a != b and a != 1 and b != 1:
            return False
    return True


_dims = st.integers(1, 4)
shapes = st.lists(_dims, min_size=0, max_size=3).map(tuple)


def _values(mode: str):
    if mode == "smooth":
        magnitude = st.floats(1e-2, 2.0, allow_nan=False, allow_infinity=False)
        return st.tuples(st.sampled_from([-1.0, 1.0]), magnitude).map(lambda t: t[0] * t[1])
    if mode == "ties":
        return st.sampled_from([-1.0, -0.5, 0.0, 0.5, 1.0])
    special = st.sampled_from(
        [math.nan, -math.nan, math.inf, -math.inf, -0.0, 0.0, 1e300, -1e300, 1e-300, 5e-324]
    )
    return st.one_of(special, special, st.floats(-2.0, 2.0))


#: Values that tell NaN's sign apart: ``-math.nan`` is a NaN with its sign
#: bit set, which ``float('nan')`` isn't. x86 hardware makes this one for
#: ``0/0`` and ``inf - inf``.
NEG_NAN = -math.nan


@st.composite
def programs(draw, *, modes=("smooth", "ties", "extreme"), max_steps=8, max_leaves=3):
    mode = draw(st.sampled_from(list(modes)))
    leaves = draw(st.lists(shapes, min_size=1, max_size=max_leaves))
    current = list(leaves)
    steps: list[Step] = []
    target = draw(st.integers(1, max_steps))
    attempts = 0
    while len(steps) < target and attempts < 4 * max_steps:
        attempts += 1
        step = draw(_step(current))
        if step is None:
            continue
        try:
            shape = _shape_of(step.op, current, step.args, step.params)
        except (RuntimeError, ValueError, IndexError):
            continue
        if math.prod(shape) > 256 or len(shape) > 4:
            continue
        steps.append(step)
        current.append(shape)
    data = []
    for shape in leaves:
        n = math.prod(shape)
        data.append(draw(st.lists(_values(mode), min_size=n, max_size=n)))
    return Program(leaves, steps, data, mode)


def leaf_tensors(program: Program, device: str, *, requires_grad: bool) -> list[torch.Tensor]:
    out = []
    for shape, values in zip(program.leaves, program.data, strict=True):
        t = torch.tensor(values, dtype=F64).reshape(shape)
        if device != "cpu":
            t = t.to(device)
        out.append(t.requires_grad_(requires_grad))
    return out


def output_of(program: Program) -> int:
    return len(program.leaves) + len(program.steps) - 1


def scale(values: list[torch.Tensor]) -> float:
    """The largest finite magnitude among ``values``, at least 1: the scale
    the rounding error of a result is measured against."""
    m = 1.0
    for v in values:
        v = v.detach()
        finite = v[torch.isfinite(v)]
        if finite.numel():
            m = max(m, float(finite.abs().max()))
    return m
