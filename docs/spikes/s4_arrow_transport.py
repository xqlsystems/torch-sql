"""Spike S4: can dense arrays move from XQL's engines to PyTorch without copies?

Question (design.md §16, S4; §6.6, §12.2, D13). The boundary between PyTorch
and XQL has a *transport* part, which moves bytes, and a *layout* part, which
turns a long table (one row per index tuple) into a dense tensor and back.
This spike answers, from Python on one x86 machine with no GPU in use:

1. Can a query result reach a torch tensor without a copy? Arrow's PyCapsule
   interface (``__arrow_c_stream__``) moves record batches between
   libraries, and DLPack (``__dlpack__``, ``torch.from_dlpack``) moves dense
   buffers into PyTorch.
2. Which objects in the stack export Arrow's *device* capsules
   (``__arrow_c_device_array__``, ``__arrow_c_device_stream__``), the
   variant for memory that may live on a GPU, and can torch read them?
3. Does Arrow's fixed-shape tensor extension type (a column whose values are
   dense arrays of one shape) survive DataFusion's and DuckDB's operators,
   and does its buffer reach torch without a copy?
4. What does the layout step cost for a Samudra-sized batch: rows already in
   order (a reshape), rows in any order (a scatter by position), and a
   partial tensor with land rows absent (a scatter plus a mask)?

What it can't answer: whether a B200 can read Arrow buffers in place from
its Grace CPU's memory over NVLink-C2C. See "Not tested here" below.

Run:
    uv run --project docs/spikes python docs/spikes/s4_arrow_transport.py

Exits non-zero if any finding below stops holding.

Findings (2026-11-05; torch 2.15.0+cpu, pyarrow 26.0.1, datafusion 54.0.0,
duckdb 2.0.0, xarray-sql 0.6.0; 12-core x86, 2.6 GB peak; times vary by
machine):

1. A pyarrow float32 or float64 array with no nulls goes to torch through
   DLPack with no copy (same data pointer). An array with nulls is refused
   (``ArrowTypeError``), so the boundary must fill absent values itself
   (design P6). A DataFusion result arrives as many chunks (124 for a 1M-row
   projection): each chunk is zero-copy into torch, but one tensor across
   chunks needs a concatenation, which copies. xarray-sql's tables behave
   the same, one chunk per registered partition.
2. Only pyarrow's ``Array`` and ``RecordBatch`` export
   ``__arrow_c_device_array__``, and this pyarrow wheel has no CUDA support
   (``pyarrow.cuda`` is absent). Nothing exports
   ``__arrow_c_device_stream__``. DataFusion's ``DataFrame``, DuckDB's
   relations and xarray-sql's readers export the host stream only. Torch
   reads no Arrow capsule; it reads DLPack. So today's path to torch is the
   Arrow host stream, then DLPack per column chunk.
3. DataFusion keeps the fixed-shape tensor type through a projection, a
   filter and a sort. A bare projection passes the same buffer through; a
   filter or a sort writes a new one, as expected. DuckDB 2.0 drops the
   extension type and returns a plain ``fixed_size_list``: the values
   survive, the shape does not. A tensor column's values go to torch with
   no copy through the storage array's child buffer.
4. For 162 channels on a 180 × 360 grid (10.5M rows, float32):
   - rows in order become a (C, H, W) tensor by reshape, with no copy, in
     9 us;
   - checking that rows are in order takes a pass over the index columns,
     231 ms here (NumPy int64 arithmetic), longer than the scatter itself,
     so the boundary should rely on an Exact fact (design §10) instead of
     checking;
   - rows in any order cost a scatter by computed position, 182 ms;
   - sorting in DataFusion first (``ORDER BY c, lat, lon``) costs 3.9 s,
     20 times the scatter, so the boundary should scatter, never ask the
     engine to sort;
   - a partial tensor with 30% of rows absent (land) becomes a dense tensor
     with zeros on land and an exact mask in 111 ms.

   Layout is free when the engine keeps rows in order and a fact says so,
   and about 0.1 to 0.2 s per batch on one core when it doesn't. That is
   the same order as a Samudra training step, so an ordered path matters
   for the data loader (spike S9).

Not tested here: Grace and NVLink-C2C. On Beta, the test is: allocate Arrow
buffers in Grace memory (ordinary host memory, ``cudaHostRegister``'d
memory, and ``cudaMallocManaged`` memory), hand them to torch through DLPack
as CPU, ``kDLCUDAHost`` and ``kDLCUDAManaged`` tensors, and time a B200
kernel that reads them in place against one that reads after a copy to
HBM. Samudra's ARM container has the CUDA build of torch this needs. Also
not tested: CUDA at all (this env's torch is CPU-only, and pyarrow's wheel
has no CUDA), so GPU-resident Arrow arrays were not exercised.
"""

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

failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        failures.append(name)


def data_address(arr: pa.Array) -> int:
    return arr.buffers()[1].address


def timed(fn, repeat=3):
    best = float("inf")
    out = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return out, best


# 1. Query results to torch -------------------------------------------------

print("\n1. Arrow results to torch through DLPack")
for dtype in ("float32", "float64"):
    a = pa.array(np.linspace(0, 1, 1000, dtype=dtype))
    t = torch.from_dlpack(a)
    check(f"{dtype} array, no nulls, zero-copy", t.data_ptr() == data_address(a))

try:
    torch.from_dlpack(pa.array([1.0, None, 3.0]))
    check("array with nulls is refused", False, "accepted")
except (pa.ArrowTypeError, TypeError, BufferError) as e:
    check("array with nulls is refused", True, type(e).__name__)

ctx = datafusion.SessionContext()
src = pa.table({"i": np.arange(1_000_000), "v": np.random.rand(1_000_000)})
ctx.register_record_batches("src", [src.to_batches(max_chunksize=250_000)])
result = ctx.sql("SELECT v * 2 AS v FROM src").to_arrow_table()
chunks = result.column("v").chunks
check(
    "DataFusion result: each chunk zero-copy into torch",
    all(torch.from_dlpack(c).data_ptr() == data_address(c) for c in chunks),
    f"{len(chunks)} chunks",
)
whole = result.column("v").combine_chunks()
check(
    "one tensor across chunks needs a copy",
    len(chunks) == 1 or data_address(whole) != data_address(chunks[0]),
)

ds = xr.Dataset(
    {"temp": (("lat", "lon"), np.random.rand(90, 180).astype("float32"))},
    coords={"lat": np.arange(90), "lon": np.arange(180)},
)
xctx = xs.Context()
xctx.from_dataset("temp", ds, chunks={"lat": 30})
xresult = xctx.sql("SELECT temp FROM temp").to_arrow_table()
xchunks = xresult.column("temp").chunks
check(
    "xarray-sql result chunks zero-copy into torch",
    all(torch.from_dlpack(c).data_ptr() == data_address(c) for c in xchunks),
    f"{len(xchunks)} chunks",
)

# 2. Device capsules ---------------------------------------------------------

print("\n2. Who exports Arrow device capsules")
reader = xs.read_xarray(ds, chunks={"lat": 30})
objects = {
    "pyarrow.Array": pa.array([1.0]),
    "pyarrow.RecordBatch": src.to_batches()[0],
    "pyarrow.Table": src,
    "datafusion.DataFrame": ctx.sql("SELECT 1.0 AS v"),
    "duckdb relation": duckdb.connect().sql("SELECT 1.0 AS v"),
    "xarray-sql reader": reader,
}
exports = {}
for name, obj in objects.items():
    exports[name] = (
        hasattr(obj, "__arrow_c_stream__"),
        hasattr(obj, "__arrow_c_device_array__"),
        hasattr(obj, "__arrow_c_device_stream__"),
    )
    s, da, dst = exports[name]
    print(f"      {name:22s} stream={s} device_array={da} device_stream={dst}")
check(
    "only pyarrow exports a device capsule",
    {n for n, (_, da, dst) in exports.items() if da or dst}
    <= {"pyarrow.Array", "pyarrow.RecordBatch"},
)
check(
    "DataFusion, DuckDB and xarray-sql export the host stream",
    all(exports[n][0] for n in ("datafusion.DataFrame", "duckdb relation", "xarray-sql reader")),
)
try:
    torch.from_dlpack(src.to_batches()[0])
    check("torch reads a RecordBatch directly", False, "accepted (finding changed)")
except Exception as e:  # noqa: BLE001
    check("torch doesn't read Arrow capsules, only DLPack", True, type(e).__name__)
try:
    import pyarrow.cuda  # noqa: F401

    check("pyarrow wheel has no CUDA support", False, "pyarrow.cuda imported")
except ImportError:
    check("pyarrow wheel has no CUDA support", True)

# 3. Fixed-shape tensor columns ---------------------------------------------

print("\n3. Fixed-shape tensor columns through the engines")
planes = np.random.rand(16, 180, 360).astype("float32")
tensor_col = pa.FixedShapeTensorArray.from_numpy_ndarray(planes)
tbl = pa.table({"k": np.arange(16), "t": tensor_col})
values_address = data_address(tensor_col.storage.values)

ctx3 = datafusion.SessionContext()
ctx3.register_record_batches("x", [tbl.to_batches()])
for label, query, same_buffer in [
    ("projection", "SELECT t FROM x", True),
    ("filter", "SELECT t FROM x WHERE k >= 8", False),
    ("sort", "SELECT t FROM x ORDER BY k DESC", False),
]:
    out = ctx3.sql(query).to_arrow_table()
    col = out.column("t").chunk(0)
    kept = isinstance(out.schema.field("t").type, pa.FixedShapeTensorType)
    check(f"DataFusion {label} keeps the tensor type", kept, str(out.schema.field("t").type)[:60])
    if kept:
        addr = data_address(col.storage.values)
        check(
            f"DataFusion {label} {'reuses' if same_buffer else 'rewrites'} the buffer",
            (addr == values_address) == same_buffer,
        )

con = duckdb.connect()
con.register("x", tbl)
dtype = con.sql("SELECT t FROM x WHERE k >= 8").fetch_arrow_table().schema.field("t").type
check(
    "DuckDB drops the extension type to fixed_size_list",
    pa.types.is_fixed_size_list(dtype) and not isinstance(dtype, pa.FixedShapeTensorType),
    str(dtype),
)

dense = torch.from_dlpack(tensor_col.storage.values).reshape(16, 180, 360)
check(
    "tensor column to torch, zero-copy, right values",
    dense.data_ptr() == values_address and np.array_equal(dense.numpy(), planes),
)
del planes, tensor_col, tbl, dense, ctx3, con

# 4. Layout: long table to dense tensor ------------------------------------

print("\n4. Layout for a 162 x 180 x 360 batch (10.5M rows)")
C, H, W = 162, 180, 360
n = C * H * W
rng = np.random.default_rng(0)
c, lat, lon = (a.ravel() for a in np.meshgrid(np.arange(C), np.arange(H), np.arange(W), indexing="ij"))
v = rng.standard_normal(n, dtype=np.float32)
ordered = pa.table({"c": c, "lat": lat, "lon": lon, "v": v})
del c, lat, lon


def reshape_ordered(t: pa.Table) -> torch.Tensor:
    return torch.from_dlpack(t.column("v").chunk(0)).reshape(C, H, W)


x, dt = timed(lambda: reshape_ordered(ordered))
check("ordered rows: reshape is zero-copy", x.data_ptr() == data_address(ordered.column("v").chunk(0)), f"{dt * 1e6:.0f} us")


def positions(t: pa.Table) -> np.ndarray:
    return (t.column("c").to_numpy() * H + t.column("lat").to_numpy()) * W + t.column("lon").to_numpy()


ok, dt_check = timed(lambda: bool(np.array_equal(positions(ordered), np.arange(n))))
check("checking order costs a pass over the indices", ok, f"{dt_check * 1e3:.0f} ms (an Exact fact skips it)")

perm = rng.permutation(n)
shuffled = ordered.take(perm)


def scatter(t: pa.Table, fill=0.0):
    pos = torch.from_numpy(positions(t))
    vals = torch.from_numpy(t.column("v").to_numpy())
    out = torch.full((n,), fill, dtype=torch.float32)
    out[pos] = vals
    mask = torch.zeros(n, dtype=torch.bool)
    mask[pos] = True
    return out.reshape(C, H, W), mask.reshape(C, H, W)


(xs_, mask_), dt_scatter = timed(lambda: scatter(shuffled))
check("unordered rows: scatter gives the same tensor", torch.equal(xs_, x) and bool(mask_.all()), f"{dt_scatter * 1e3:.0f} ms")
del xs_, mask_

ctx4 = datafusion.SessionContext()
ctx4.register_record_batches("b", [shuffled.to_batches()])
sorted_tbl, dt_sort = timed(lambda: ctx4.sql("SELECT * FROM b ORDER BY c, lat, lon").to_arrow_table(), repeat=1)
print(f"      sorting in DataFusion first: {dt_sort * 1e3:.0f} ms")
del sorted_tbl, ctx4

land = np.zeros((H, W), dtype=bool)
land[:, : int(W * 0.3)] = True  # about 30% land, as on Earth
keep = ~np.broadcast_to(land, (C, H, W)).ravel()
partial = shuffled.filter(pa.array(keep[perm]))
(xp, maskp), dt_partial = timed(lambda: scatter(partial))
expected_mask = torch.from_numpy(np.ascontiguousarray(~np.broadcast_to(land, (C, H, W))))
check(
    "partial tensor: dense with zeros on land, and an exact mask",
    torch.equal(maskp, expected_mask)
    and torch.equal(xp[maskp], x[maskp])
    and bool((xp[~maskp] == 0).all()),
    f"{partial.num_rows / n:.0%} rows present, {dt_partial * 1e3:.0f} ms",
)

print(
    f"\nsummary: reshape {dt * 1e6:.0f} us, order check {dt_check * 1e3:.0f} ms, "
    f"scatter {dt_scatter * 1e3:.0f} ms, DataFusion sort {dt_sort * 1e3:.0f} ms, "
    f"partial scatter {dt_partial * 1e3:.0f} ms"
)
if failures:
    print(f"\n{len(failures)} finding(s) no longer hold: {failures}")
    sys.exit(1)
print("\nall findings hold")
