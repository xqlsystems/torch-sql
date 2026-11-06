"""Spike S8: Samudra's building blocks as portable SQL through index tables.

Question
--------
Do the layers Samudra is built from lower to plain SQL through index tables
(design §9.5), give the same numbers as PyTorch, and differentiate with ddx?
And does dropping land rows (a *partial* tensor, design §8.3) give the masked
model of design §7.3 item 1?

An *index table* lists, for an op whose inputs and outputs are related by
position, which input positions feed which output positions. It is computed
once in NumPy from the op's parameters. The op is then a join in which every
condition is an equality, followed by an aggregate. Tensors are long tables
with one row per index tuple: ``x(n, c, lat, lon, v)``.

Checks, all in float64 against PyTorch 2.15 on the CPU:

  A. 3x3 convolution with dilations 1, 2, 4 and 8, circular padding in
     longitude and zero padding in latitude, exactly as Samudra pads
     (``F.pad(circular)`` then ``F.pad(constant)``, then a valid conv), with
     a bias; and the 1x1 convolution.
  B. ``AvgPool2d(2)`` and Samudra's ``ZonallyPeriodicBilinearUpsample``
     (circular pad of 1, bilinear x2 with ``align_corners=False``, crop),
     the second through a tap table with weights.
  C. BatchNorm in training mode (two-pass variance, design §9.4),
     InstanceNorm, and CappedGELU.
  D. ddx gradients of each against PyTorch's autograd.
  E. The partial (masked) conv and BatchNorm, forward and gradients, against
     the dense masked reference; a NaN on land shows absence is not zero.
  F. The same forward SQL on DuckDB.
  G. Plan size and time for ConvNeXt blocks (design §9.6).

Design refs: §8.3, §9.4, §9.5, §9.6, decisions D4 and D7.

Run (from the repository root; NixOS needs the LD_LIBRARY_PATH of
``flake.nix``-style shells for torch):

    uv run --project docs/spikes python docs/spikes/s8_index_table_conv.py

Exits non-zero if a finding stops holding.

Findings (2026-11-05; ddxdb 0.4.6, DataFusion 54.0.0, DuckDB 2.0.0, torch 2.15)
--------------------------------------------------------------------------------
All checks pass. Grid 8 x 16, batch 2, 3 channels (6 hidden).

1. Every forward op matches PyTorch to at most 2.7e-15 (float64 rounding), on
   DataFusion and on DuckDB, from the same SQL text with no dialect changes.
   Circular padding is which rows the longitude index table has; zero padding
   is absent rows; each dilation (1, 2, 4, 8) is a different table. Samudra's
   zonally periodic upsample is a separable tap table: the circular pad and
   the crop fold into the longitude weights.
2. ddx differentiates every op, with respect to inputs, weights, biases and
   norm parameters, and matches autograd to at most 3.2e-14. ``LEAST``
   differentiates as a clamp, so the cap of CappedGELU needs nothing new.
3. Neither engine has ``erf``, and ddx has no rule for it (ddx#256 is still
   open), so this spike's CappedGELU uses the tanh form of GELU and checks it
   against ``F.gelu(approximate="tanh")``. Exact GELU needs ``erf`` in both
   engines and in ddx.
4. The partial conv and BatchNorm (land rows absent, 30% land) equal the
   dense masked reference ``conv(x * m) * m`` and ocean-only statistics, in
   the forward pass and in their gradients. With a NaN in one land cell, the
   dense reference gives NaN at 36 ocean outputs (``NaN * 0`` is NaN); the
   partial SQL never reads the land row and is unchanged.
5. Plan size (design §9.6). For a stack of ConvNeXt blocks written as one
   query with CTEs:

       blocks   DataFusion operators, plan s, run s   DuckDB operators, s
       1          441      0.1      0.1                    75     0.05
       2        7,978      3.9      2.5                   151     0.09
       3      136,107     66.5    145.9                   227     0.20

   DataFusion grows about 17x per block, because it plans a CTE again at
   every read, and BatchNorm reads its input three times (mean, variance,
   normalize), so the reads multiply through the stack. DuckDB grows by 75
   operators per block. Samudra has about 8 blocks per rollout step, so on
   DataFusion a whole step can't be one query: cut points (design §9.6) are
   required, at least at every norm.
6. ddx's gradient of one block with respect to its first conv runs in 0.9 s
   and matches autograd to 1.3e-14.
7. A trap for generated SQL: a CTE named like a registered table shadows it
   in every later CTE. A first draft named blocks ``b1``, ``b2`` and silently
   read the 1x1 conv's bias table ``b1`` as a block's output. torch-sql must
   generate names that can't collide with registered tables.
"""

import sys
import time

import ddxdb
import duckdb
import numpy as np
import pyarrow as pa
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(0)

N, C, H, W = 2, 3, 8, 16  # batch, channels, latitude, longitude
HIDDEN = 6  # ConvNeXt's channels after its upscale factor (2 here)
EPS = 1e-5
CAP = 10.0
FAILURES = []


def check(name, err, tol):
    ok = bool(np.isfinite(err) and err <= tol)
    print(f"{'PASS' if ok else 'FAIL'}  {name:58s} max|diff| = {err:.2e}")
    if not ok:
        FAILURES.append(name)


# --- Tensors as long tables -------------------------------------------------

DIMS4 = ("n", "c", "lat", "lon")


def to_table(t, dims=DIMS4, keep=None):
    """A long table with one row per index of ``t``; ``keep`` drops rows."""
    a = t.detach().numpy()
    idx = np.indices(a.shape).reshape(a.ndim, -1)
    vals = a.reshape(-1)
    sel = np.ones(vals.shape, bool) if keep is None else keep.reshape(-1)
    cols = {d: pa.array(idx[i][sel], pa.int64()) for i, d in enumerate(dims)}
    cols["v"] = pa.array(vals[sel], pa.float64())
    return pa.table(cols)


def from_rows(tbl, shape, dims=DIMS4, value="v"):
    """Dense array from result rows; cells with no row are NaN."""
    out = np.full(shape, np.nan)
    d = tbl.to_pydict()
    out[tuple(np.asarray(d[k]) for k in dims)] = d[value]
    return out


def index_table(rows, names):
    rows = np.asarray(rows, dtype=np.int64).reshape(-1, len(names))
    return pa.table({k: pa.array(rows[:, i]) for i, k in enumerate(names)})


# --- Index tables, computed in NumPy ---------------------------------------


def conv_lat(h, d):
    """3-tap conv along latitude, dilation d, zero padding: taps that fall
    off the grid have no row."""
    return index_table(
        [(o, k, o + (k - 1) * d) for o in range(h) for k in range(3)
         if 0 <= o + (k - 1) * d < h],
        ("out_lat", "ky", "in_lat"))


def conv_lon(w, d):
    """3-tap conv along longitude, dilation d, circular padding."""
    return index_table(
        [(o, k, (o + (k - 1) * d) % w) for o in range(w) for k in range(3)],
        ("out_lon", "kx", "in_lon"))


def pool(n):
    return index_table([(i // 2, i) for i in range(n)], ("out", "in"))


def bilinear_weights(n_in):
    """PyTorch's bilinear x2 weights along one axis, align_corners=False, as
    a dense (2 n_in, n_in) matrix."""
    m = np.zeros((2 * n_in, n_in))
    for o in range(2 * n_in):
        src = max((o + 0.5) / 2 - 0.5, 0.0)
        i0 = int(np.floor(src))
        i1 = min(i0 + 1, n_in - 1)
        lam = src - i0
        m[o, i0] += 1 - lam
        m[o, i1] += lam
    return m


def tap_table(m, names):
    o, i = np.nonzero(m)
    return pa.table({names[0]: pa.array(o, pa.int64()),
                     names[1]: pa.array(i, pa.int64()),
                     "w": pa.array(m[o, i], pa.float64())})


def upsample_lat(h):
    return tap_table(bilinear_weights(h), ("out_lat", "in_lat"))


def upsample_lon(w):
    """Samudra's zonally periodic upsample: pad one column each side
    circularly, upsample the padded width, crop, and fold the padded columns
    back onto the real ones."""
    m = bilinear_weights(w + 2)[2:2 + 2 * w]  # crop: start = scale = 2
    folded = np.zeros((2 * w, w))
    for j in range(w + 2):
        folded[:, (j - 1) % w] += m[:, j]
    return tap_table(folded, ("out_lon", "in_lon"))


# --- SQL, one function per op ------------------------------------------------
# Each returns a SELECT over relations named by its arguments, producing
# (n, c, lat, lon, v). Programs are WITH lists of these.


def conv3_sql(x, w, b, ia, io):
    return f"""SELECT y.n, y.c, y.lat, y.lon, y.v + {b}.v AS v FROM (
        SELECT {x}.n, {w}.co AS c, {ia}.out_lat AS lat, {io}.out_lon AS lon,
               SUM({x}.v * {w}.v) AS v
        FROM {x}
        JOIN {ia} ON {x}.lat = {ia}.in_lat
        JOIN {io} ON {x}.lon = {io}.in_lon
        JOIN {w} ON {x}.c = {w}.ci AND {ia}.ky = {w}.ky AND {io}.kx = {w}.kx
        GROUP BY {x}.n, {w}.co, {ia}.out_lat, {io}.out_lon) y
        JOIN {b} ON y.c = {b}.c"""


def conv1_sql(x, w, b):
    return f"""SELECT y.n, y.c, y.lat, y.lon, y.v + {b}.v AS v FROM (
        SELECT {x}.n, {w}.co AS c, {x}.lat, {x}.lon, SUM({x}.v * {w}.v) AS v
        FROM {x} JOIN {w} ON {x}.c = {w}.ci
        GROUP BY {x}.n, {w}.co, {x}.lat, {x}.lon) y
        JOIN {b} ON y.c = {b}.c"""


def restrict_sql(x, ocean):
    """Keep only rows on the domain (design §8.3)."""
    return f"""SELECT {x}.n, {x}.c, {x}.lat, {x}.lon, {x}.v FROM {x}
        JOIN {ocean} ON {x}.lat = {ocean}.lat AND {x}.lon = {ocean}.lon"""


def pool_sql(x, pa_, po):
    return f"""SELECT {x}.n, {x}.c, {pa_}."out" AS lat, {po}."out" AS lon,
               AVG({x}.v) AS v
        FROM {x} JOIN {pa_} ON {x}.lat = {pa_}."in"
                 JOIN {po} ON {x}.lon = {po}."in"
        GROUP BY {x}.n, {x}.c, {pa_}."out", {po}."out\""""


def upsample_sql(x, ua, uo):
    return f"""SELECT {x}.n, {x}.c, {ua}.out_lat AS lat, {uo}.out_lon AS lon,
               SUM({x}.v * {ua}.w * {uo}.w) AS v
        FROM {x} JOIN {ua} ON {x}.lat = {ua}.in_lat
                 JOIN {uo} ON {x}.lon = {uo}.in_lon
        GROUP BY {x}.n, {x}.c, {ua}.out_lat, {uo}.out_lon"""


def norm_ctes(p, x, g=None, b=None, by=("c",)):
    """BatchNorm (by c) or InstanceNorm (by n, c): an AVG for the mean, a
    second AVG for the variance (two passes, design §9.4), joined back.
    Returns CTEs; the last, ``{p}``, is the output."""
    keys = ", ".join(by)
    on = lambda a, b_: " AND ".join(f"{a}.{k} = {b_}.{k}" for k in by)
    affine = (f" * {g}.v + {b}.v", f" JOIN {g} ON {x}.c = {g}.c JOIN {b} ON {x}.c = {b}.c") if g else ("", "")
    return [
        (f"{p}_mean", f"SELECT {keys}, AVG(v) AS mu FROM {x} GROUP BY {keys}"),
        (f"{p}_var", f"""SELECT {', '.join(f'{x}.{k}' for k in by)},
               AVG(({x}.v - m.mu) * ({x}.v - m.mu)) AS var
               FROM {x} JOIN {p}_mean m ON {on(x, 'm')}
               GROUP BY {', '.join(f'{x}.{k}' for k in by)}"""),
        (p, f"""SELECT {x}.n, {x}.c, {x}.lat, {x}.lon,
               ({x}.v - m.mu) / SQRT(s.var + {EPS}){affine[0]} AS v
               FROM {x} JOIN {p}_mean m ON {on(x, 'm')}
               JOIN {p}_var s ON {on(x, 's')}{affine[1]}"""),
    ]


GELU_C = float(np.sqrt(2 / np.pi))


def capped_gelu_sql(x):
    """CappedGELU with the tanh form of GELU, since no engine here has erf."""
    return f"""SELECT n, c, lat, lon,
        LEAST(0.5 * v * (1 + tanh({GELU_C!r} * (v + 0.044715 * v * v * v))), {CAP}) AS v
        FROM {x}"""


def program(ctes, final):
    return "WITH " + ",\n".join(f"{n} AS ({q})" for n, q in ctes) + "\n" + final


def loss_ctes(y, r):
    """A scalar loss whose gradient is the vector-Jacobian product of y with r."""
    return [("loss", f"""SELECT SUM({y}.v * {r}.v) AS l FROM {y}
             JOIN {r} ON {y}.n = {r}.n AND {y}.c = {r}.c
             AND {y}.lat = {r}.lat AND {y}.lon = {r}.lon""")]


# --- Engines ------------------------------------------------------------------

df = ddxdb.Context()  # DataFusion, with ddx's grad in its SQL
dk = duckdb.connect()
TABLES = {}


def register(name, tbl):
    TABLES[name] = tbl
    if df.table_exist(name):
        df.deregister_table(name)
    df.register_record_batches(name, [tbl.to_batches()])
    dk.register(name, tbl)


def run_df(sql):
    return df.sql(sql).to_arrow_table()


def run_dk(sql):
    return dk.sql(sql).to_arrow_table()


def grad_of(ctes, y, r, wrt, dims):
    """ddx's gradient of SUM(y * r) with respect to table ``wrt``'s v."""
    sql = program(ctes + loss_ctes(y, r),
                  f"SELECT {', '.join(dims)}, grad(loss, v) AS g FROM {wrt}")
    return run_df(sql)


# --- PyTorch references ---------------------------------------------------------


def t_conv3(x, w, b, d):
    x = F.pad(x, (d, d, 0, 0), mode="circular")
    x = F.pad(x, (0, 0, d, d), mode="constant")
    return F.conv2d(x, w, b, dilation=d)


def t_upsample(x):
    width = x.shape[-1]
    p = F.pad(x, (1, 1, 0, 0), mode="circular")
    u = F.interpolate(p, scale_factor=(2, 2), mode="bilinear", align_corners=False)
    return u[..., 2:2 + width * 2]


def t_capped_gelu(x):
    return torch.clamp(F.gelu(x, approximate="tanh"), max=CAP)


def rand(*shape, scale=1.0):
    return torch.tensor(rng.standard_normal(shape) * scale, requires_grad=True)


WDIMS = ("co", "ci", "ky", "kx")


def main():
    x = rand(N, C, H, W)
    register("x", to_table(x))
    for d in (1, 2, 4, 8):
        register(f"ia{d}", conv_lat(H, d))
        register(f"io{d}", conv_lon(W, d))

    # A. 3x3 conv with each dilation, and 1x1 conv ---------------------------
    print("A. convolution through index tables")
    w3 = rand(HIDDEN, C, 3, 3, scale=0.3)
    b3 = rand(HIDDEN)
    register("w3", to_table(w3, WDIMS))
    register("b3", to_table(b3, ("c",)))
    for d in (1, 2, 4, 8):
        sql = conv3_sql("x", "w3", "b3", f"ia{d}", f"io{d}")
        ref = t_conv3(x, w3, b3, d).detach().numpy()
        for eng, run in (("DataFusion", run_df), ("DuckDB", run_dk)):
            out = from_rows(run(sql), ref.shape)
            check(f"conv3x3 d={d} {eng}", np.max(np.abs(out - ref)), 1e-12)
    w1 = rand(C, HIDDEN, 1, 1, scale=0.3)
    b1 = rand(C)
    register("w1", to_table(w1, WDIMS))
    register("b1", to_table(b1, ("c",)))
    h = t_conv3(x, w3, b3, 1)
    register("h", to_table(h))
    ref = F.conv2d(h, w1, b1).detach().numpy()
    for eng, run in (("DataFusion", run_df), ("DuckDB", run_dk)):
        out = from_rows(run(conv1_sql("h", "w1", "b1")), ref.shape)
        check(f"conv1x1 {eng}", np.max(np.abs(out - ref)), 1e-12)

    # B. pooling and Samudra's upsample ----------------------------------------
    print("B. pooling and zonally periodic upsampling")
    register("pa", pool(H))
    register("po", pool(W))
    register("ua", upsample_lat(H))
    register("uo", upsample_lon(W))
    for name, sql, ref in (
        ("avg_pool2d(2)", pool_sql("x", "pa", "po"), F.avg_pool2d(x, 2)),
        ("ZonallyPeriodicBilinearUpsample", upsample_sql("x", "ua", "uo"), t_upsample(x)),
    ):
        ref = ref.detach().numpy()
        for eng, run in (("DataFusion", run_df), ("DuckDB", run_dk)):
            out = from_rows(run(sql), ref.shape)
            check(f"{name} {eng}", np.max(np.abs(out - ref)), 1e-12)

    # C. norms and activation ---------------------------------------------------
    print("C. BatchNorm, InstanceNorm, CappedGELU")
    g = rand(C)
    bb = rand(C)
    register("g", to_table(g, ("c",)))
    register("bb", to_table(bb, ("c",)))
    bn_ref = F.batch_norm(x, None, None, g, bb, training=True, eps=EPS)
    in_ref = F.instance_norm(x, eps=EPS)
    big = rand(N, C, H, W, scale=6.0)  # reaches the cap of 10
    register("big", to_table(big))
    cases = (
        ("BatchNorm2d (training)", program(norm_ctes("bn", "x", "g", "bb"), "SELECT * FROM bn"), bn_ref),
        ("InstanceNorm2d", program(norm_ctes("inn", "x", by=("n", "c")), "SELECT * FROM inn"), in_ref),
        ("CappedGELU (tanh GELU)", capped_gelu_sql("big"), t_capped_gelu(big)),
    )
    for name, sql, ref in cases:
        ref = ref.detach().numpy()
        for eng, run in (("DataFusion", run_df), ("DuckDB", run_dk)):
            out = from_rows(run(sql), ref.shape)
            check(f"{name} {eng}", np.max(np.abs(out - ref)), 1e-12)
    for eng, con in (("DataFusion", lambda q: df.sql(q).collect()), ("DuckDB", dk.sql)):
        try:
            con("SELECT erf(0.5)")
            print(f"NOTE  {eng} has erf")
        except Exception:
            print(f"NOTE  {eng} has no erf; exact GELU needs an engine function")
    print(f"NOTE  ddx has rules for: {', '.join(sorted(ddxdb.supported_functions()))}")

    # D. ddx gradients against autograd --------------------------------------
    print("D. ddx gradients (vjp with a random cotangent r)")

    def vjp_check(name, ctes, y_name, y_ref, wrt):
        r = torch.tensor(rng.standard_normal(y_ref.shape))
        register("r", to_table(r))
        params = [p for _, p, _ in wrt]
        for p in params:
            p.grad = None
        (y_ref * r).sum().backward()
        for tname, p, dims in wrt:
            got = from_rows(grad_of(ctes, y_name, "r", tname, dims), p.shape, dims, "g")
            check(f"d/d{tname} of {name}", np.max(np.abs(got - p.grad.numpy())), 1e-10)

    for d in (1, 4):
        vjp_check(f"conv3x3 d={d}", [("y", conv3_sql("x", "w3", "b3", f"ia{d}", f"io{d}"))], "y",
                  t_conv3(x, w3, b3, d),
                  [("x", x, DIMS4), ("w3", w3, WDIMS), ("b3", b3, ("c",))])
    vjp_check("avg_pool2d", [("y", pool_sql("x", "pa", "po"))], "y", F.avg_pool2d(x, 2), [("x", x, DIMS4)])
    vjp_check("upsample", [("y", upsample_sql("x", "ua", "uo"))], "y", t_upsample(x), [("x", x, DIMS4)])
    vjp_check("BatchNorm2d", norm_ctes("bn", "x", "g", "bb"), "bn",
              F.batch_norm(x, None, None, g, bb, training=True, eps=EPS),
              [("x", x, DIMS4), ("g", g, ("c",)), ("bb", bb, ("c",))])
    vjp_check("InstanceNorm2d", norm_ctes("inn", "x", by=("n", "c")), "inn",
              F.instance_norm(x, eps=EPS), [("x", x, DIMS4)])
    vjp_check("CappedGELU", [("y", capped_gelu_sql("big"))], "y", t_capped_gelu(big), [("big", big, DIMS4)])

    # E. partial tensors: land rows dropped ------------------------------------
    print("E. partial tensors (land rows absent)")
    land = rng.random((H, W)) < 0.3
    ocean = ~land
    m = torch.tensor(ocean, dtype=torch.float64)
    oc = np.argwhere(ocean)
    register("ocean", pa.table({"lat": pa.array(oc[:, 0], pa.int64()), "lon": pa.array(oc[:, 1], pa.int64())}))
    keep = np.broadcast_to(ocean, (N, C, H, W))
    xp = rand(N, C, H, W)
    register("xp", to_table(xp, keep=keep))
    sel = np.broadcast_to(ocean, (N, HIDDEN, H, W))

    conv_p = [("y0", conv3_sql("xp", "w3", "b3", "ia2", "io2")), ("y", restrict_sql("y0", "ocean"))]
    ref = (t_conv3(xp * m, w3, b3, 2) * m)
    out = from_rows(run_df(program(conv_p, "SELECT * FROM y")), ref.shape)
    check("partial conv == conv(x*m)*m on ocean", np.max(np.abs(out[sel] - ref.detach().numpy()[sel])), 1e-12)
    check("partial conv has no land rows", float(np.isfinite(out[~sel]).sum()), 0)

    r = torch.tensor(rng.standard_normal(ref.shape)) * m
    register("r", to_table(r))
    for p in (xp, w3):
        p.grad = None
    (ref * r).sum().backward()
    gx = from_rows(grad_of(conv_p, "y", "r", "xp", DIMS4), xp.shape, DIMS4, "g")
    gw = from_rows(grad_of(conv_p, "y", "r", "w3", WDIMS), w3.shape, WDIMS, "g")
    check("partial conv d/dx on ocean", np.max(np.abs(gx[keep] - xp.grad.numpy()[keep])), 1e-10)
    check("partial conv d/dw", np.max(np.abs(gw - w3.grad.numpy())), 1e-10)

    def masked_bn(t):
        cnt = t.shape[0] * m.sum()
        mu = (t * m).sum((0, 2, 3), keepdim=True) / cnt
        var = (((t - mu) * m) ** 2).sum((0, 2, 3), keepdim=True) / cnt
        return ((t - mu) / torch.sqrt(var + EPS) * g.view(1, -1, 1, 1) + bb.view(1, -1, 1, 1)) * m

    bn_p = norm_ctes("bn", "xp", "g", "bb")
    ref = masked_bn(xp)
    out = from_rows(run_df(program(bn_p, "SELECT * FROM bn")), ref.shape)
    check("partial BatchNorm == ocean-only statistics", np.max(np.abs(out[keep] - ref.detach().numpy()[keep])), 1e-12)
    r = torch.tensor(rng.standard_normal(ref.shape)) * m
    register("r", to_table(r))
    for p in (xp, g):
        p.grad = None
    (ref * r).sum().backward()
    gx = from_rows(grad_of(bn_p, "bn", "r", "xp", DIMS4), xp.shape, DIMS4, "g")
    gg = from_rows(grad_of(bn_p, "bn", "r", "g", ("c",)), g.shape, ("c",), "g")
    check("partial BatchNorm d/dx on ocean", np.max(np.abs(gx[keep] - xp.grad.numpy()[keep])), 1e-10)
    check("partial BatchNorm d/dgamma", np.max(np.abs(gg - g.grad.numpy())), 1e-10)

    # A NaN in one land cell: multiplying by the mask doesn't remove it.
    li, lj = np.argwhere(land)[0]
    xnan = xp.detach().clone()
    xnan[:, :, li, lj] = float("nan")
    dense = (t_conv3(xnan * m, w3, b3, 2) * m).detach().numpy()
    bad = int(np.isnan(dense[sel]).sum())
    print(f"NOTE  dense masked conv with one NaN on land: {bad} ocean outputs are NaN")
    if bad == 0:
        FAILURES.append("NaN on land should reach ocean cells in the dense reference")
    # The partial table has no row for that cell, so the SQL result is unchanged.
    out = from_rows(run_df(program(conv_p, "SELECT * FROM y")), dense.shape)
    clean = (t_conv3(xp * m, w3, b3, 2) * m).detach().numpy()
    check("partial conv ignores the NaN land cell", np.max(np.abs(out[sel] - clean[sel])), 1e-12)

    # F is covered above: every forward check ran on DuckDB with the same SQL.

    # G. ConvNeXt blocks: plan size and time -------------------------------------
    print("G. ConvNeXt blocks (conv, BN, GELU, conv, BN, GELU, 1x1, residual)")
    wb = rand(HIDDEN, HIDDEN, 3, 3, scale=0.2)
    bnb = rand(HIDDEN)
    gh, bh = rand(HIDDEN), rand(HIDDEN)
    register("w3b", to_table(wb, WDIMS))
    register("b3b", to_table(bnb, ("c",)))
    register("gh", to_table(gh, ("c",)))
    register("bh", to_table(bh, ("c",)))

    def block_ctes(p, x, d):
        return ([(f"{p}c1", conv3_sql(x, "w3", "b3", f"ia{d}", f"io{d}"))]
                + norm_ctes(f"{p}n1", f"{p}c1", "gh", "bh")
                + [(f"{p}a1", capped_gelu_sql(f"{p}n1")),
                   (f"{p}c2", conv3_sql(f"{p}a1", "w3b", "b3b", f"ia{d}", f"io{d}"))]
                + norm_ctes(f"{p}n2", f"{p}c2", "gh", "bh")
                + [(f"{p}a2", capped_gelu_sql(f"{p}n2")),
                   (f"{p}pw", conv1_sql(f"{p}a2", "w1", "b1")),
                   (p, f"""SELECT {x}.n, {x}.c, {x}.lat, {x}.lon, {x}.v + {p}pw.v AS v
                        FROM {x} JOIN {p}pw ON {x}.n = {p}pw.n AND {x}.c = {p}pw.c
                        AND {x}.lat = {p}pw.lat AND {x}.lon = {p}pw.lon""")])

    def t_block(t, d):
        bn = lambda u: F.batch_norm(u, None, None, gh, bh, training=True, eps=EPS)
        a = t_capped_gelu(bn(t_conv3(t, w3, b3, d)))
        a = t_capped_gelu(bn(t_conv3(a, wb, bnb, d)))
        return t + F.conv2d(a, w1, b1)

    print(f"      {'blocks':>6} {'DF ops':>7} {'DF plan s':>9} {'DF run s':>8} {'Duck ops':>8} {'Duck s':>7}")
    for nb in (1, 2, 3):
        ctes, prev, ref = [], "x", x
        for k in range(nb):
            ctes += block_ctes(f"blk{k}", prev, 1)
            prev, ref = f"blk{k}", t_block(ref, 1)
        sql = program(ctes, f"SELECT * FROM {prev}")
        t0 = time.perf_counter()
        plan = df.sql("EXPLAIN " + sql).to_pylist()[-1]["plan"]
        t1 = time.perf_counter()
        out_df = from_rows(run_df(sql), ref.shape)
        t2 = time.perf_counter()
        js = dk.sql("EXPLAIN (FORMAT json) " + sql).fetchall()[0][1]
        t3 = time.perf_counter()
        out_dk = from_rows(run_dk(sql), ref.shape)
        t4 = time.perf_counter()
        df_ops = sum(1 for line in plan.splitlines() if "Exec" in line)
        dk_ops = js.count('"name"')
        print(f"      {nb:>6} {df_ops:>7} {t1 - t0:>9.3f} {t2 - t1:>8.3f} {dk_ops:>8} {t4 - t3:>7.3f}")
        r_ = ref.detach().numpy()
        check(f"{nb} ConvNeXt block(s) DataFusion", np.max(np.abs(out_df - r_)), 1e-11)
        check(f"{nb} ConvNeXt block(s) DuckDB", np.max(np.abs(out_dk - r_)), 1e-11)

    ctes = block_ctes("blk0", "x", 1)
    r = torch.tensor(rng.standard_normal((N, C, H, W)))
    register("r", to_table(r))
    w3.grad = None
    (t_block(x, 1) * r).sum().backward()
    t0 = time.perf_counter()
    gw = from_rows(grad_of(ctes, "blk0", "r", "w3", WDIMS), w3.shape, WDIMS, "g")
    print(f"NOTE  ddx gradient of one block w.r.t. its first conv: {time.perf_counter() - t0:.2f} s")
    check("ConvNeXt block d/dw (first conv)", np.max(np.abs(gw - w3.grad.numpy())), 1e-10)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {FAILURES}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
