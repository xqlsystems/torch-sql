"""S7 smoke test: does the XQL stack import and run on this machine?

This is the script a Beta smoke job runs inside Samudra's container (see
../s7_arm_packaging.py). It runs on any platform. It checks, in order:

1. the platform: CPU architecture, glibc, Python;
2. imports and versions of datafusion, pyarrow, numpy, xarray, zarr,
   xarray-sql, ddxdb, einfold, duckdb and (if present) torch;
3. xarray-sql: a Dataset registered as a table and queried with SQL;
4. ddx: a gradient through `grad(loss, val)` on ddxdb's DataFusion context;
5. einfold: installing it in a DataFusion session (it needs DataFusion 55);
6. DuckDB's community `substrait` extension, which ddx's DuckDB backend needs.

Checks 1 to 4 are required: the script exits 1 if one fails. Checks 5 and 6
are reported as findings (FINDING), because they fail for version reasons
that S7 documents, not because of the platform.

The last line printed is a JSON summary, for a job log to keep.
"""

import json
import platform
import sys

REQUIRED = []
REPORT = {}


def required(name, fn):
    try:
        detail = fn()
        REQUIRED.append(True)
        print(f"PASS  {name}" + (f"  ({detail})" if detail else ""))
    except Exception as e:  # noqa: BLE001 - report every failure the same way
        REQUIRED.append(False)
        detail = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
        print(f"FAIL  {name}  ({detail})")
    REPORT[name] = detail


def finding(name, fn):
    try:
        detail = "ok: " + (fn() or "")
    except Exception as e:  # noqa: BLE001
        detail = f"fails: {type(e).__name__}: {str(e).splitlines()[0][:200]}"
    print(f"FINDING  {name}  ({detail})")
    REPORT[name] = detail


def check_platform():
    libc = platform.libc_ver()
    return f"{platform.machine()}, {libc[0]} {libc[1]}, Python {platform.python_version()}"


def check_versions():
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for dist in ["datafusion", "pyarrow", "numpy", "xarray", "zarr", "xarray-sql",
                 "ddxdb", "einfold", "duckdb", "torch"]:
        try:
            out[dist] = version(dist)
        except PackageNotFoundError:
            out[dist] = None
    for module in ["datafusion", "pyarrow", "numpy", "xarray", "xarray_sql", "ddxdb", "einfold", "duckdb"]:
        __import__(module)
    REPORT["versions"] = out
    return ", ".join(f"{k} {v}" for k, v in out.items() if v)


def check_xarray_sql():
    import numpy as np
    import xarray as xr
    import xarray_sql as xql

    ds = xr.Dataset({"t": (("lat", "lon"), np.arange(12.0).reshape(3, 4))},
                    coords={"lat": [0, 1, 2], "lon": [0, 1, 2, 3]})
    ctx = xql.Context()
    ctx.from_dataset("ocean", ds)
    rows = ctx.sql("SELECT lat, SUM(t) AS s FROM ocean GROUP BY lat ORDER BY lat").to_pydict()
    assert rows["s"] == [6.0, 22.0, 38.0], rows
    return "row sums match"


def check_ddx():
    from ddxdb import Context

    ctx = Context()
    ctx.sql("CREATE TABLE w (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 2.0)")
    got = ctx.sql("WITH loss AS (SELECT SUM(val * val) AS l FROM w) "
                  "SELECT i, grad(loss, val) AS g FROM w ORDER BY i").to_pydict()
    assert got["g"] == [2.0, 4.0], got
    return "grad of SUM(val^2) is 2 val"


def check_einfold():
    from datafusion import SessionContext
    from einfold.datafusion import EinFold

    ctx = SessionContext().with_extensions(EinFold())
    ctx.sql("SELECT 1").collect()
    return "installed in a DataFusion session"


def check_duckdb_substrait():
    import duckdb

    con = duckdb.connect()
    try:
        con.execute("LOAD substrait")
    except Exception:  # noqa: BLE001 - not installed yet
        con.execute("INSTALL substrait FROM community")
        con.execute("LOAD substrait")
    return f"loaded on DuckDB {duckdb.__version__}"


if __name__ == "__main__":
    required("platform", check_platform)
    required("imports", check_versions)
    required("xarray-sql query", check_xarray_sql)
    required("ddx gradient", check_ddx)
    finding("einfold in a DataFusion session", check_einfold)
    finding("DuckDB substrait extension", check_duckdb_substrait)
    print(json.dumps(REPORT, default=str))
    sys.exit(0 if all(REQUIRED) else 1)
