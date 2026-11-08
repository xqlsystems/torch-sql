"""Elementwise ops (design §8.4, §9.4): a projection of one input, or a join
of several on the index columns they share.

The SQL must mean what PyTorch means (P1), and three things differ between
SQL engines and IEEE floating point, which PyTorch follows:

- **NaN in comparisons.** In PyTorch every comparison with NaN is false
  except ``!=``. SQL says NaN equals NaN, and its engines don't agree on
  where NaN sorts: DuckDB puts every NaN above every number, and DataFusion
  uses IEEE's total order, which puts a NaN with its sign bit set (the NaN
  x86 gives for ``0/0``) below ``-inf``. So every float comparison and
  every ``CASE`` that a NaN could reach checks ``isnan`` (:func:`compare`).
- **Domain errors.** DuckDB raises on ``ln(0)``, ``ln(-1)`` and
  ``sqrt(-1)``, and DataFusion on ``sqrt(-1)``, where PyTorch returns
  ``-inf`` or NaN, and DuckDB on ``sin`` and ``cos`` of an infinity, where
  PyTorch returns NaN. Those ops guard their argument with ``CASE``, which
  both engines evaluate only on the rows that take a branch.
- **Ties.** ``maximum`` and ``minimum`` are ``CASE`` expressions, so ddx's
  gradient at a tie goes to the first argument, where PyTorch's splits it
  in half.
"""

from __future__ import annotations

import math

import torch

from .. import graph as g
from . import Context, op
from ._build import elementwise, python_dtype, scalar

aten = torch.ops.aten
FLOATS = (torch.float64,)

NAN = g.Lit(math.nan, torch.float64)
NEG_INF = g.Lit(-math.inf, torch.float64)


def out_of(ctx: Context):
    return ctx.out.shape, ctx.out.dtype


def unary(ctx, x, fn):
    shape, dtype = out_of(ctx)
    return elementwise(shape, dtype, [x], fn)


def binary(ctx, a, b, fn, *, compute=None):
    shape, dtype = out_of(ctx)
    return elementwise(shape, dtype, [a, b], fn, compute=compute)


def isnan(e: g.Expr) -> g.Expr | None:
    """``isnan(e)``, or None when ``e`` is a literal that isn't NaN."""
    if isinstance(e, g.Lit):
        return (
            g.Lit(True, torch.bool) if isinstance(e.value, float) and math.isnan(e.value) else None
        )
    return g.call("isnan", e)


def any_of(*es):
    es = [e for e in es if e is not None]
    if not es:
        return g.Lit(False, torch.bool)
    out = es[0]
    for e in es[1:]:
        out = g.call("or", out, e)
    return out


def all_of(*es):
    es = [e for e in es if e is not None]
    out = es[0]
    for e in es[1:]:
        out = g.call("and", out, e)
    return out


def case(cond, then, otherwise):
    return g.call("case", cond, then, otherwise)


def compare(cmp: str, a: g.Expr, b: g.Expr, floating: bool) -> g.Expr:
    """``a cmp b`` as IEEE floating point means it, when ``floating``."""
    plain = g.call(cmp, a, b)
    if not floating:
        return plain
    nan_a, nan_b = isnan(a), isnan(b)
    if nan_a is None and nan_b is None:
        return plain
    if cmp == "<>":
        return any_of(plain, nan_a, nan_b)
    nots = [g.call("not", e) for e in (nan_a, nan_b) if e is not None]
    return all_of(plain, *nots)


def comparison_dtype(ctx: Context, a, b) -> torch.dtype:
    """The dtype PyTorch compares ``a`` and ``b`` in. For an integer tensor
    against a Python float, PyTorch compares in the default float dtype,
    float32; torch-sql compares in float64, which it supports, and which
    agrees with float32 except for integers beyond 2**24 that float32 can't
    tell apart."""

    def meta(x):
        if isinstance(x, g.Node):
            return torch.empty(x.shape, dtype=x.dtype, device="meta")
        return x

    with torch._C.DisableTorchFunction():
        dtype = torch.result_type(meta(a), meta(b))
    return torch.float64 if dtype.is_floating_point else dtype


# ----------------------------------------------------------------------------
# Arithmetic


@op(aten.add.Tensor, aten.add.Scalar)
def add(ctx, a, b, *, alpha=1):
    if alpha == 1:
        return binary(ctx, a, b, lambda x, y: x + y)
    return binary(ctx, a, b, lambda x, y: x + y * scalar(alpha, ctx.out.dtype))


@op(aten.sub.Tensor, aten.sub.Scalar)
def sub(ctx, a, b, *, alpha=1):
    if alpha == 1:
        return binary(ctx, a, b, lambda x, y: x - y)
    return binary(ctx, a, b, lambda x, y: x - y * scalar(alpha, ctx.out.dtype))


@op(aten.rsub.Tensor, aten.rsub.Scalar)
def rsub(ctx, a, b, *, alpha=1):
    return sub(ctx, b, a, alpha=alpha)


@op(aten.mul.Tensor, aten.mul.Scalar)
def mul(ctx, a, b):
    if ctx.out.dtype == torch.bool:
        return binary(ctx, a, b, lambda x, y: g.call("and", x, y))
    return binary(ctx, a, b, lambda x, y: x * y)


@op(aten.div.Tensor, aten.div.Scalar)
def div(ctx, a, b):
    # A true division: PyTorch's result is a float, and the operands are
    # converted to it first, so SQL never divides integers.
    return binary(ctx, a, b, lambda x, y: x / y)


@op(aten.neg.default)
def neg(ctx, x):
    return unary(ctx, x, lambda v: -v)


@op(aten.reciprocal.default)
def reciprocal(ctx, x):
    return unary(ctx, x, lambda v: scalar(1.0, ctx.out.dtype) / v)


def _pow(v, p):
    if isinstance(p, g.Lit) and p.value == 2:
        return v * v
    if isinstance(p, g.Lit) and p.value == 1:
        return v
    return g.call("power", v, p)


@op(aten.pow.Tensor_Scalar)
def pow_tensor_scalar(ctx, x, p):
    shape, dtype = out_of(ctx)
    if dtype not in FLOATS:
        raise NotImplementedError(f"pow of {dtype} tensors")
    return elementwise(shape, dtype, [x], lambda v: _pow(v, scalar(p, dtype)))


@op(aten.pow.Tensor_Tensor)
def pow_tensor_tensor(ctx, x, p):
    return binary(ctx, x, p, lambda v, q: g.call("power", v, q))


@op(aten.pow.Scalar)
def pow_scalar(ctx, base, x):
    return unary(ctx, x, lambda v: g.call("power", scalar(base, ctx.out.dtype), v))


# ----------------------------------------------------------------------------
# Functions


@op(aten.exp.default)
def exp(ctx, x):
    return unary(ctx, x, lambda v: g.call("exp", v))


@op(aten.log.default)
def log(ctx, x):
    # ln(0) is -inf and ln(x < 0) is NaN, as in PyTorch; DuckDB raises on
    # both. A NaN takes the first branch on DuckDB and the last on
    # DataFusion if its sign bit is set; either way the result is NaN, so
    # this needs no isnan. sqrt and rsqrt are the same.
    def fn(v):
        zero = scalar(0.0, ctx.out.dtype)
        return case(g.call(">", v, zero), g.call("ln", v), case(g.call("=", v, zero), NEG_INF, NAN))

    return unary(ctx, x, fn)


@op(aten.sqrt.default)
def sqrt(ctx, x):
    return unary(
        ctx, x, lambda v: case(g.call(">=", v, scalar(0.0, ctx.out.dtype)), g.call("sqrt", v), NAN)
    )


@op(aten.rsqrt.default)
def rsqrt(ctx, x):
    one, zero = scalar(1.0, ctx.out.dtype), scalar(0.0, ctx.out.dtype)
    return unary(ctx, x, lambda v: case(g.call(">=", v, zero), one / g.call("sqrt", v), NAN))


@op(aten.tanh.default)
def tanh(ctx, x):
    return unary(ctx, x, lambda v: g.call("tanh", v))


@op(aten.sigmoid.default)
def sigmoid(ctx, x):
    one = scalar(1.0, ctx.out.dtype)
    return unary(ctx, x, lambda v: one / (one + g.call("exp", -v)))


@op(aten.abs.default)
def abs_(ctx, x):
    return unary(ctx, x, lambda v: g.call("abs", v))


def _periodic(name):
    # sin and cos of ±inf are NaN in PyTorch; DuckDB raises. v - v is NaN
    # exactly when v is infinite or NaN.
    def lower(ctx, x):
        return unary(ctx, x, lambda v: case(g.call("isnan", v - v), NAN, g.call(name, v)))

    return lower


op(aten.sin.default)(_periodic("sin"))
op(aten.cos.default)(_periodic("cos"))


@op(aten.relu.default)
def relu(ctx, x):
    zero = scalar(0, ctx.out.dtype)
    if ctx.out.dtype not in FLOATS:
        return unary(ctx, x, lambda v: case(g.call("<=", v, zero), zero, v))
    # NaN passes through, as in PyTorch.
    return unary(
        ctx, x, lambda v: case(g.call("isnan", v), v, case(g.call("<=", v, zero), zero, v))
    )


#: GELU's tanh form, with PyTorch's constant sqrt(2/pi).
_GELU_BETA = math.sqrt(2.0) * (2.0 / math.sqrt(math.pi)) * 0.5


@op(aten.gelu.default)
def gelu(ctx, x, *, approximate="none"):
    if approximate != "tanh":
        raise NotImplementedError(
            "exact GELU needs erf, which neither DataFusion nor DuckDB has and ddx "
            "can't differentiate yet (xqlsystems/ddx#256); use "
            "nn.GELU(approximate='tanh'), which differs by less than 5e-4 (design §7.3)"
        )
    dt = ctx.out.dtype

    def fn(v):
        inner = scalar(_GELU_BETA, dt) * (v + scalar(0.044715, dt) * v * v * v)
        return scalar(0.5, dt) * v * (scalar(1.0, dt) + g.call("tanh", inner))

    return unary(ctx, x, fn)


# ----------------------------------------------------------------------------
# Comparisons, selection and logic


def _comparison(cmp):
    def lower(ctx, a, b):
        dtype = comparison_dtype(ctx, a, b)
        return binary(ctx, a, b, lambda x, y: compare(cmp, x, y, dtype in FLOATS), compute=dtype)

    return lower


for _name, _cmp in (
    ("eq", "="),
    ("ne", "<>"),
    ("lt", "<"),
    ("le", "<="),
    ("gt", ">"),
    ("ge", ">="),
):
    op(getattr(aten, _name).Tensor, getattr(aten, _name).Scalar)(_comparison(_cmp))


@op(aten.where.self)
def where(ctx, cond, a, b):
    # The condition stays boolean; the branches take the result's dtype.
    shape, dtype = out_of(ctx)
    return elementwise(shape, dtype, [cond, a, b], case, compute=[torch.bool, dtype, dtype])


def _extremum(cmp):
    # A NaN in either argument wins, as in PyTorch.
    def lower(ctx, a, b):
        def fn(x, y):
            pick = case(g.call(cmp, x, y), x, y)
            if ctx.out.dtype not in FLOATS:
                return pick
            return case(any_of(isnan(x), isnan(y)), NAN, pick)

        return binary(ctx, a, b, fn)

    return lower


op(aten.maximum.default)(_extremum(">="))
op(aten.minimum.default)(_extremum("<="))


@op(aten.clamp.default, aten.clamp.Tensor)
def clamp(ctx, x, min=None, max=None):
    """``min`` and ``max`` are numbers or tensors. A NaN in ``x`` stays NaN,
    where SQL would find it greater than ``max``; a NaN bound gives NaN, as
    in PyTorch."""
    shape, dt = out_of(ctx)
    operands = [x] + [b for b in (min, max) if b is not None]

    def fn(v, *bounds):
        bounds = list(bounds)
        lo = bounds.pop(0) if min is not None else None
        hi = bounds.pop(0) if max is not None else None
        # min(max(v, lo), hi), as PyTorch computes it: hi wins when lo > hi.
        out = v
        if lo is not None:
            out = case(g.call("<", out, lo), lo, out)
        if hi is not None:
            out = case(g.call(">", out, hi), hi, out)
        if dt in FLOATS:
            nan = any_of(isnan(v), *(isnan(b) for b in (lo, hi) if b is not None))
            out = case(nan, NAN, out)
        return out

    return elementwise(shape, dt, operands, fn)


@op(aten.clamp_min.default, aten.clamp_min.Tensor)
def clamp_min(ctx, x, min):
    return clamp(ctx, x, min=min)


@op(aten.clamp_max.default, aten.clamp_max.Tensor)
def clamp_max(ctx, x, max):
    return clamp(ctx, x, max=max)


@op(aten.isnan.default)
def isnan_(ctx, x):
    if x.dtype not in FLOATS:
        return unary(ctx, x, lambda v: g.Lit(False, torch.bool))
    return elementwise(
        ctx.out.shape, torch.bool, [x], lambda v: g.call("isnan", v), compute=x.dtype
    )


@op(aten.logical_not.default, aten.bitwise_not.default)
def logical_not(ctx, x):
    if ctx.func is aten.bitwise_not.default and x.dtype != torch.bool:
        raise NotImplementedError("bitwise_not of integers")
    return elementwise(
        ctx.out.shape, torch.bool, [x], lambda v: g.call("not", _truth(v, x.dtype)), compute=x.dtype
    )


def _truth(e, dtype):
    return e if dtype == torch.bool else g.call("<>", e, scalar(0, dtype))


def _logical(fn_name):
    def lower(ctx, a, b):
        if ctx.out.dtype != torch.bool:  # bitwise_and or bitwise_or of integers
            raise NotImplementedError(f"{ctx.func} of integers")

        def fn(x, y):
            return g.call(fn_name, _truth(x, dtype_of(a)), _truth(y, dtype_of(b)))

        return elementwise(ctx.out.shape, torch.bool, [a, b], fn, compute=[None, None])

    return lower


def dtype_of(x) -> torch.dtype:
    return x.dtype if isinstance(x, g.Node) else python_dtype(x)


op(aten.logical_and.default, aten.bitwise_and.Tensor)(_logical("and"))
op(aten.logical_or.default, aten.bitwise_or.Tensor)(_logical("or"))
