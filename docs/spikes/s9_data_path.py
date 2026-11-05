"""Spike S9: how fast is the co-processor's data path today?

Question (design.md §6.3, milestone C1, experiment E2; samudra-on-beta.md
stage 1). The co-processor's first job is Samudra's data loader: read a
rollout window from the OM4 Zarr store, normalize each channel with its mean
and standard deviation, mask land, and hand dense tensors to PyTorch. This
spike builds a small synthetic store shaped like Samudra's one-degree OM4
and times that path three ways:

- **baseline**: plain Xarray and torch, as Samudra's ``cpu`` loader does it
  (``XarrayCanonicalReader.read``, which materializes one Dask graph per
  request, then ``BatchPreprocessor.prepare_planes``:
  ``((x - mean) / std).nan_to_num(0)``, then ``where(mask, x, 0)``);
- **baseline-zarr**: the same, but reading Zarr directly with no Dask
  (``open_zarr(chunks=None)``), a stronger baseline;
- **xql-dense**: xarray-sql on DataFusion, normalizing and masking in SQL,
  keeping every row, then the layout step of spike S4 (a reshape when rows
  arrive in order);
- **xql-ocean**: the same, but dropping rows that are land at the surface
  (land at every depth), then a scatter into a dense tensor;

and, if xarray-sql's DuckDB adapter takes the Dataset, **duck-dense** on
DuckDB.

The synthetic store copies OM4's layout (Samudra's docs/data.md): one
variable per channel and depth (``thetao_0`` .. ``thetao_18``,
``so_0`` .. ``so_18``, plus ``zos`` and ``hfds``: 40 channels of float32 on
``(time, y, x)`` = ``(T, 180, 360)``), Zarr chunks of ``(1, 180, 360)``, NaN
on land, and one land mask per depth (``mask_0`` .. ``mask_18``, deeper
levels having more land). The real store has 100 variables and 4,745 times;
this one has 40 and 24, to fit the machine.

Run:
    uv run --project docs/spikes python docs/spikes/s9_data_path.py

Exits non-zero if the batches disagree or a finding below stops holding.

Findings (2026-11-05; xarray-sql 0.6.0, datafusion 54.0.0, duckdb 2.0.0,
torch 2.15.0+cpu, zarr 3.4.2; 12-core x86 shared with other jobs, load
average 5 to 10, so times vary by about 30% between runs; 2.1 GB peak).
One window is 5 times × 40 channels × 180 × 360, 52 MB of float32. Best of
three warm runs, seconds per window:

    baseline (Dask, as Samudra's cpu loader)   0.17 to 0.25
    baseline-zarr (no Dask)                    0.16 to 0.24
    xql-dense (DataFusion)                     0.40 to 0.48
    xql-ocean (DataFusion, land rows dropped)  0.36 to 0.43
    duck-dense (DuckDB)                        0.49 to 0.54

1. Every path produces the same batches as Samudra's own normalization and
   masking: equal zeros on land, values within 1e-5.
2. xarray-sql's path is 1.6 to 3.2 times slower than the baselines today,
   on DataFusion and on DuckDB, across five runs. One run, the first after
   the store was written, had the Dask baseline at 1.5 s per window and
   xql-dense ahead of it; no other run did.
3. The scan dominates. Turning one window of Zarr into Arrow rows costs
   about 240 ms, as much as the baselines' whole pipeline; the mask join and
   normalization add about 90 ms, and the layout step (spike S4) about 40
   ms. The scan writes three int64 index columns for every grid cell, which
   the dense baselines never materialize.
4. The time filter prunes partitions: a 5-of-24 window scan costs 0.21 of a
   full scan, so reads scale with the window, not the store.
5. Dropping land rows (xql-ocean) saves layout and output rows but not
   reading, because the mask is a join, not a filter pushed into the Zarr
   scan; land chunks are still read and decoded. Its gain is small and
   inside the noise.
6. DataFusion returns rows out of order (it scans partitions in parallel),
   so the layout step must scatter rather than reshape. Ordered output, or
   a fact that says rows are ordered within each partition, would make
   layout free (spike S4).

What this means for stage 1 of samudra-on-beta.md: the XQL data path is
correct today but not yet competitive on speed, before even meeting the
``rust`` loader. The gap is in xarray-sql's scan (index columns and row
construction), not in SQL's normalization or in torch-sql's layout. Real
OM4 batches add the half-degree grid (4× the cells) and Samudra's batch of
4 windows; each DDP rank has spare Grace cores, which this test doesn't
use beyond DataFusion's own threads.
"""

import os
import shutil
import sys
import time
import warnings

import datafusion
import duckdb
import numpy as np
import pyarrow as pa
import torch
import xarray as xr
import xarray_sql as xs

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")

SCRATCH = os.environ.get(
    "S9_SCRATCH",
    "/tmp/claude-1000/-home-alxmrs-git-torch-sql/6269e2f7-871b-4697-8627-7086b2eff6d6/scratchpad/s9",
)
T, LEV, H, W = 24, 19, 180, 360
WINDOW = 5  # one input step and four rollout steps
CHANNELS = [f"thetao_{k}" for k in range(LEV)] + [f"so_{k}" for k in range(LEV)] + ["zos", "hfds"]
C = len(CHANNELS)
LEVEL = {name: int(name.rsplit("_", 1)[1]) if name[-1].isdigit() else 0 for name in CHANNELS}

failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        failures.append(name)


# The synthetic store -------------------------------------------------------


def build_store(path: str) -> None:
    rng = np.random.default_rng(0)
    yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(0, 2 * np.pi, W), indexing="ij")
    # Land grows with depth: a continent band plus a shelf that widens.
    masks = {}
    for k in range(LEV):
        land = (np.abs(xx - 4.5) < 0.6 + 0.03 * k) | (np.abs(yy) > 0.92 - 0.01 * k)
        masks[f"mask_{k}"] = (("y", "x"), ~land)
    data = {}
    for name in CHANNELS:
        k = LEVEL[name]
        base = 10 + 5 * np.cos(np.pi * yy) - 0.3 * k if name.startswith("thetao") else 35 + 0.1 * np.sin(xx)
        field = base[None] + 0.5 * rng.standard_normal((T, H, W))
        field = np.where(masks[f"mask_{k}"][1][None], field, np.nan).astype("float32")
        data[name] = (("time", "y", "x"), field)
    ds = xr.Dataset(
        {**data, **masks},
        coords={"time": np.arange(T), "y": np.arange(H), "x": np.arange(W)},
    )
    encoding = {name: {"chunks": (1, H, W)} for name in CHANNELS}
    ds.to_zarr(path, mode="w", encoding=encoding, consolidated=True)
    means = xr.Dataset({n: ds[n].mean().astype("float32") for n in CHANNELS})
    stds = xr.Dataset({n: ds[n].std().astype("float32") for n in CHANNELS})
    means.to_zarr(path + ".means", mode="w")
    stds.to_zarr(path + ".stds", mode="w")


store = os.path.join(SCRATCH, "OM4_synth.zarr")
if not os.path.exists(store + ".stds"):
    os.makedirs(SCRATCH, exist_ok=True)
    print(f"building {store}")
    build_store(store)
ds = xr.open_zarr(store)
means = xr.open_zarr(store + ".means").load()
stds = xr.open_zarr(store + ".stds").load()
mean_np = np.array([float(means[n]) for n in CHANNELS], dtype="float32")
std_np = np.array([float(stds[n]) for n in CHANNELS], dtype="float32")
mask_np = np.stack([ds[f"mask_{LEVEL[n]}"].values for n in CHANNELS])  # (C, H, W)
size_mb = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(store) for f in fs) / 1e6
print(f"store: {C} channels x {T} times x {H} x {W}, {size_mb:.0f} MB on disk")
out_mb = WINDOW * C * H * W * 4 / 1e6

# Baseline: Samudra's cpu loader path --------------------------------------

mean_t = torch.from_numpy(mean_np).reshape(1, C, 1, 1)
std_t = torch.from_numpy(std_np).reshape(1, C, 1, 1)
mask_t = torch.from_numpy(mask_np)


ds_nodask = xr.open_zarr(store, chunks=None)


def baseline(t0: int, source: xr.Dataset = ds) -> torch.Tensor:
    selected = source[CHANNELS].isel(time=np.arange(t0, t0 + WINDOW))
    planes = selected.to_dataarray("channel").transpose("time", "channel", "y", "x").values
    data = torch.from_numpy(planes)
    data = ((data - mean_t) / std_t).nan_to_num(nan=0.0)
    return torch.where(mask_t, data, 0.0)


# xarray-sql on DataFusion --------------------------------------------------

grid = ds[CHANNELS].drop_vars([v for v in ds.coords if v not in ("time", "y", "x")])
mask_ds = ds[[f"mask_{k}" for k in range(LEV)]]
stats = pa.table(
    {**{f"m_{n}": [float(mean_np[i])] for i, n in enumerate(CHANNELS)},
     **{f"s_{n}": [float(std_np[i])] for i, n in enumerate(CHANNELS)}}
)


def select_list() -> str:
    cols = []
    for n in CHANNELS:
        norm = f"COALESCE((g.{n} - s.m_{n}) / s.s_{n}, 0)"
        cols.append(f"CAST(CASE WHEN m.mask_{LEVEL[n]} THEN {norm} ELSE 0 END AS FLOAT) AS {n}")
    return ", ".join(cols)


def window_sql(t0: int, ocean_only: bool) -> str:
    where = f"g.time BETWEEN {t0} AND {t0 + WINDOW - 1}"
    if ocean_only:
        where += " AND m.mask_0"
    return (
        f"SELECT g.time, g.y, g.x, {select_list()} "
        "FROM grid g JOIN masks m ON g.y = m.y AND g.x = m.x CROSS JOIN stats s "
        f"WHERE {where}"
    )


def to_tensor(tbl: pa.Table, t0: int) -> tuple[torch.Tensor, bool]:
    """The layout step: (time, y, x, one column per channel) rows to a
    (time, channel, y, x) tensor. Reshape when rows are in order, else
    scatter by position (spike S4)."""
    tbl = tbl.combine_chunks()
    pos = ((tbl.column("time").to_numpy() - t0) * H + tbl.column("y").to_numpy()) * W + tbl.column("x").to_numpy()
    n = WINDOW * H * W
    in_order = len(pos) == n and bool(np.all(pos[1:] > pos[:-1]))
    out = torch.zeros((C, n), dtype=torch.float32)
    idx = None if in_order else torch.from_numpy(pos)
    for i, name in enumerate(CHANNELS):
        col = torch.from_numpy(tbl.column(name).to_numpy())
        if in_order:
            out[i] = col
        else:
            out[i, idx] = col
    return out.reshape(C, WINDOW, H, W).transpose(0, 1), in_order


def run_engine(execute, t0: int, ocean_only: bool):
    t_a = time.perf_counter()
    tbl = execute(window_sql(t0, ocean_only))
    t_b = time.perf_counter()
    x, in_order = to_tensor(tbl, t0)
    t_c = time.perf_counter()
    return x, in_order, t_b - t_a, t_c - t_b


ctx = xs.Context()
t_reg = time.perf_counter()
ctx.from_dataset("grid", grid, chunks={"time": 1})
ctx.from_dataset("masks", mask_ds)
ctx.register_record_batches("stats", [stats.to_batches()])
t_reg = time.perf_counter() - t_reg


def datafusion_execute(sql: str) -> pa.Table:
    return ctx.sql(sql).to_arrow_table()


# DuckDB through xarray-sql's adapter --------------------------------------

duck = None
try:
    duck = duckdb.connect()
    xs.register(duck, "grid", grid, chunks={"time": 1})
    xs.register(duck, "masks", mask_ds)
    duck.register("stats", stats)
except Exception as e:  # noqa: BLE001
    print(f"      DuckDB through xarray-sql: not run ({type(e).__name__}: {e})")
    duck = None


def duckdb_execute(sql: str) -> pa.Table:
    return duck.sql(sql).fetch_arrow_table()


# Runs ----------------------------------------------------------------------

starts = list(range(0, T - WINDOW + 1, 2))  # 10 windows
results = {}
paths = [("xql-dense", datafusion_execute, False), ("xql-ocean", datafusion_execute, True)]
if duck is not None:
    paths.append(("duck-dense", duckdb_execute, False))

reference = {}
t0_ = time.perf_counter()
for t0 in starts:
    reference[t0] = baseline(t0)
base_s = (time.perf_counter() - t0_) / len(starts)
results["baseline"] = (base_s, None, None, None)
t0_ = time.perf_counter()
for t0 in starts:
    check_nodask = baseline(t0, ds_nodask)
zarr_s = (time.perf_counter() - t0_) / len(starts)
results["baseline-zarr"] = (zarr_s, None, None, None)
check("baseline-zarr gives the same batches", torch.equal(check_nodask, reference[starts[-1]]))

for label, execute, ocean in paths:
    run_engine(execute, starts[0], ocean)  # warm up: first planning, file opens
    q_total = l_total = 0.0
    orders = []
    for t0 in starts:
        x, in_order, q_s, l_s = run_engine(execute, t0, ocean)
        q_total += q_s
        l_total += l_s
        orders.append(in_order)
        ref = reference[t0]
        if ocean:
            ocean_cells = torch.from_numpy(mask_np[0]).expand(WINDOW, C, H, W)
            same = torch.allclose(x[ocean_cells], ref[ocean_cells], atol=1e-5) and bool(
                (x[~ocean_cells] == 0).all() and (ref[~ocean_cells] == 0).all()
            )
        else:
            same = torch.allclose(x, ref, atol=1e-5)
            same = same and torch.equal(x == 0, ref == 0)
        if not same:
            check(f"{label} window {t0} matches the baseline", False, f"max diff {(x - ref).abs().max():.2e}")
    k = len(starts)
    results[label] = ((q_total + l_total) / k, q_total / k, l_total / k, all(orders))
    check(f"{label} matches the baseline on {k} windows", not any(f.startswith(label) for f in failures))

cols = ", ".join(CHANNELS)


def best_of(sql: str, k: int = 3) -> float:
    best = float("inf")
    for _ in range(k):
        t_a = time.perf_counter()
        ctx.sql(sql).to_arrow_table()
        best = min(best, time.perf_counter() - t_a)
    return best


scan_s = best_of(f"SELECT time, y, x, {cols} FROM grid WHERE time BETWEEN 4 AND {4 + WINDOW - 1}")
full_s = best_of(f"SELECT time, y, x, {cols} FROM grid")
join_s = best_of(window_sql(4, False))
print(
    f"\nDataFusion phases for one window: scan {scan_s * 1e3:.0f} ms, "
    f"scan + mask join + normalize {join_s * 1e3:.0f} ms; "
    f"scan of all {T} times {full_s * 1e3:.0f} ms"
)
check(
    "the time filter prunes partitions (a window scan costs well under a full scan)",
    scan_s < 0.6 * full_s,
    f"{scan_s / full_s:.2f} of a full scan for {WINDOW / T:.2f} of the rows",
)
print(f"registration (once): {t_reg * 1e3:.0f} ms")
print(f"one window = {WINDOW} times x {C} channels x {H} x {W} = {out_mb:.0f} MB of float32\n")
print(f"{'path':12s} {'s/window':>9s} {'windows/s':>10s} {'MB/s':>7s} {'query':>8s} {'layout':>8s}  rows in order")
for label, (s, q, l, ordered) in results.items():
    extra = "" if q is None else f" {q * 1e3:7.0f}ms {l * 1e3:7.0f}ms  {ordered}"
    print(f"{label:12s} {s:9.3f} {1 / s:10.2f} {out_mb / s:7.0f}{extra}")

xql = results["xql-dense"][0]
print(
    f"\nxql-dense takes {xql / base_s:.2f}x the Dask baseline's time and "
    f"{xql / zarr_s:.2f}x the direct-Zarr baseline's; xql-ocean "
    f"{results['xql-ocean'][0] / zarr_s:.2f}x the direct-Zarr baseline's"
)
check(
    "DataFusion returns rows out of order, so layout must scatter",
    not results["xql-dense"][3],
)
check(
    "the query, not the layout, dominates xarray-sql's time",
    results["xql-dense"][1] > results["xql-dense"][2],
)

if os.environ.get("S9_KEEP") != "1":
    shutil.rmtree(SCRATCH, ignore_errors=True)
if failures:
    print(f"\n{len(failures)} finding(s) no longer hold: {failures}")
    sys.exit(1)
print("\nall findings hold")
