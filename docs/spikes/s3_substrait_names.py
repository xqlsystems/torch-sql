"""S3: do CTE and view names survive the Substrait producers?

Question (design.md §16, S3). torch-sql replaces `torch.utils.checkpoint` with
a ddx *region*: a named relation inside which ddx saves nothing except what a
policy names (design.md §11.2, ddx#251). If the names of CTEs and views
(`WITH layers_3 AS (...)`, `CREATE VIEW layers_3 AS ...`) survive DataFusion's
and DuckDB's Substrait producers, a region needs no new SQL function: ddx
could read the name off the plan it differentiates. If they don't, a region
needs a marker function that does survive, as `ddx_stop_gradient` does.

Background. ddx differentiates Substrait plans, a standard format for query
plans (design.md §2). An engine's *producer* turns SQL into a Substrait plan;
its *consumer* turns a plan back into something it can run. A *marker* is a
function that is the identity when the query runs but tells ddx something,
like `ddx_stop_gradient(x)`.

What this spike does:

1. Plans one query with a CTE read once, a CTE read twice, and a view, on
   DataFusion, and looks for the three names in DataFusion's own logical plan
   and in the Substrait plan its producer writes.
2. Checks whether the producer shares the twice-read CTE (one subplan,
   referenced twice) or copies it.
3. Writes a marker that carries a name as a string argument,
   `ddx_name(expr, 'layers.3.conv1')`, registered as an ordinary UDF, and
   checks that the function and its string survive the producer, and that
   the consumer reads the plan back and runs it to the same rows.
4. Asks ddx (ddxdb 0.4.6) for a gradient through `ddx_stop_gradient` and
   through the new marker, to see what ddx accepts today.
5. Does 1 and 3 on DuckDB, whose producer is the community `substrait`
   extension. That extension doesn't exist for DuckDB 2.0, so the DuckDB part
   runs in a subprocess with DuckDB 1.5.6, the version ddxdb pins.

Run (from the repository root; on NixOS, first source a file that puts
libstdc++ on LD_LIBRARY_PATH):

    uv run --project docs/spikes python docs/spikes/s3_substrait_names.py

The DuckDB part needs `uv` and network access (to fetch DuckDB 1.5.6 and its
extension); without them it is reported as SKIP, not as a failure. The script
exits non-zero if any finding below stops holding.

Findings (2026-11-05; DataFusion 54.0.0, DuckDB 1.5.6, ddxdb 0.4.6):

- Names don't survive either producer. DataFusion's logical plan keeps every
  CTE and view name as a `SubqueryAlias`, but its Substrait plan contains
  none of them. DuckDB's Substrait plan contains none of them either.
- DataFusion's producer copies a twice-read CTE: the plan has two aggregates,
  not one shared subplan (no `ReferenceRel`). So even a structural reading of
  "this subplan is the region" isn't available from the producer's output.
- A marker UDF survives both producers, with its string argument:
  `ddx_name` appears as an extension function and `'layers.3.conv1'` as a
  literal. DataFusion's consumer reads the plan back and returns the same
  rows. A DuckDB macro (`CREATE MACRO`) is inlined before the producer runs,
  so a marker on DuckDB must be a real function, as ddxdb already registers
  `ddx_stop_gradient`.
- ddx today knows one marker. A gradient through `ddx_stop_gradient` works
  once `ddxdb.ad.register_stop_gradient` has registered it on the context
  (`ddxdb.Context` doesn't register it by itself). A gradient through
  `ddx_name` raises `UnsupportedExpression` ("no differentiation rule for
  `ddx_name` with 2 arguments"), which is the right refusal for a function
  ddx doesn't know.
- The DuckDB `substrait` extension isn't published for DuckDB 2.0 (HTTP 404
  from the community repository), and an environment that installs
  xarray-sql, ddxdb and einfold without extras resolves to DuckDB 2.0. ddx's
  DuckDB backend can't plan there. This matters for S7 and for design.md
  §12.3 (engine order).

Conclusion: names alone don't suffice. A region needs a marker that ddx
learns, such as `ddx_name(expr, 'name')` (the identity at run time, and the
identity for gradients) or a region marker on the region's output columns.
torch-sql writes the SQL itself, so it can emit the marker on every value a
checkpointed block produces; no user writes it by hand. This matches the
first option in the ddx#251 review ("a name: `ddx_name(expr, 'scores')`, or a
CTE or view name if those survive"): they don't, so it's the function.
"""

import json
import os
import subprocess
import sys

CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


NAMES = ["region_once", "region_twice", "block_view"]
QUERY = """
WITH region_once AS (SELECT i, v * 3 AS v FROM x),
     region_twice AS (SELECT i, SUM(v) AS v FROM x GROUP BY i)
SELECT a.i, a.v + b.v + c.v + d.v AS v
FROM region_once a
JOIN region_twice b ON a.i = b.i
JOIN region_twice c ON a.i = c.i
JOIN block_view d ON a.i = d.i
"""
MARKED = """
WITH r AS (SELECT i, ddx_name(SUM(v) * 3, 'layers.3.conv1') AS v FROM x GROUP BY i)
SELECT a.i, a.v + b.v AS v FROM r a JOIN r b ON a.i = b.i ORDER BY a.i
"""


def datafusion_part():
    import datafusion as df
    import pyarrow as pa
    from datafusion import substrait as ss, udf

    ctx = df.SessionContext()
    ctx.sql("CREATE TABLE x (i BIGINT, v DOUBLE) AS VALUES (0, 1.0), (1, 2.0)").collect()
    ctx.sql("CREATE VIEW block_view AS SELECT i, v * 2 AS v FROM x").collect()

    frame = ctx.sql(QUERY)
    logical = frame.logical_plan().display_indent()
    check("DataFusion's logical plan keeps CTE and view names",
          all(f"SubqueryAlias: {n}" in logical for n in NAMES))

    plan = json.dumps(json.loads(ss.Producer.to_substrait_plan(frame.logical_plan(), ctx).to_json()))
    found = {n: plan.count(n) for n in NAMES}
    check("DataFusion's Substrait plan drops every CTE and view name",
          not any(found.values()), f"counts {found}")
    check("DataFusion's producer copies a twice-read CTE (no shared subplan)",
          plan.count('"aggregate"') == 2 and "referenceRel" not in plan,
          f"{plan.count(chr(34) + 'aggregate' + chr(34))} aggregates")

    ctx.register_udf(udf(lambda a, b: a, [pa.float64(), pa.string()], pa.float64(),
                         "immutable", name="ddx_name"))
    marked = ctx.sql(MARKED)
    expected = marked.to_pydict()
    sub = ss.Producer.to_substrait_plan(marked.logical_plan(), ctx)
    text = sub.to_json()
    check("a marker UDF and its string survive DataFusion's producer",
          "ddx_name" in text and "layers.3.conv1" in text)
    back = ctx.create_dataframe_from_logical_plan(ss.Consumer.from_substrait_plan(ctx, sub))
    check("DataFusion's consumer runs the marked plan to the same rows",
          back.to_pydict() == expected, str(expected))


def ddx_part():
    import pyarrow as pa
    from datafusion import udf
    from ddxdb import Context, UnsupportedExpression
    from ddxdb.ad import register_stop_gradient

    ctx = Context()
    ctx.sql("CREATE TABLE w (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 2.0)")
    try:
        ctx.sql("WITH loss AS (SELECT SUM(ddx_stop_gradient(val) * val) AS l FROM w) "
                "SELECT i, grad(loss, val) AS g FROM w")
        unregistered = "accepted"
    except Exception as e:  # noqa: BLE001 - the finding is that it fails
        unregistered = type(e).__name__
    check("ddxdb.Context doesn't register ddx_stop_gradient by itself",
          unregistered != "accepted", unregistered)

    register_stop_gradient(ctx)
    got = ctx.sql("WITH loss AS (SELECT SUM(ddx_stop_gradient(val) * val) AS l FROM w) "
                  "SELECT i, grad(loss, val) AS g FROM w ORDER BY i").to_pydict()
    check("ddx differentiates through ddx_stop_gradient once registered",
          got["g"] == [1.0, 2.0], str(got["g"]))

    ctx.register_udf(udf(lambda a, b: a, [pa.float64(), pa.string()], pa.float64(),
                         "immutable", name="ddx_name"))
    try:
        ctx.sql("WITH h AS (SELECT i, ddx_name(val * val, 'blk') AS v FROM w), "
                "loss AS (SELECT SUM(v) AS l FROM h) SELECT i, grad(loss, val) AS g FROM w")
        refused = False
    except UnsupportedExpression:
        refused = True
    check("ddx refuses a marker it doesn't know (UnsupportedExpression)", refused)


def duckdb_inline_part():
    """Runs under DuckDB 1.5.6 (see duckdb_part); prints one JSON line."""
    import duckdb

    con = duckdb.connect()
    _try_install(con)
    con.execute("LOAD substrait")
    con.execute("CREATE TABLE x AS SELECT * FROM (VALUES (0, 1.0::DOUBLE), (1, 2.0)) t(i, v)")
    con.execute("CREATE VIEW block_view AS SELECT i, v * 2 AS v FROM x")
    plan = con.execute("SELECT * FROM get_substrait_json(?)", [QUERY]).fetchone()[0]
    con.create_function("ddx_name", lambda a, b: a,
                        [duckdb.sqltype("DOUBLE"), duckdb.sqltype("VARCHAR")], duckdb.sqltype("DOUBLE"))
    marked = con.execute("SELECT * FROM get_substrait_json(?)", [MARKED]).fetchone()[0]
    con.execute("CREATE MACRO ddx_macro(a, b) AS a")
    macro = con.execute("SELECT * FROM get_substrait_json(?)",
                        [MARKED.replace("ddx_name(", "ddx_macro(")]).fetchone()[0]
    print(json.dumps({
        "version": duckdb.__version__,
        "names": {n: plan.count(n) for n in NAMES},
        "udf": "ddx_name" in marked and "layers.3.conv1" in marked,
        "macro": "ddx_macro" in macro or "layers.3.conv1" in macro,
    }))


def _try_install(con):
    """Install DuckDB's community substrait extension unless it loads already."""
    try:
        con.execute("LOAD substrait")
    except Exception:  # noqa: BLE001 - any failure means "not installed yet"
        con.execute("INSTALL substrait FROM community")


def duckdb_part():
    import duckdb

    con = duckdb.connect()
    try:
        _try_install(con)
        con.execute("LOAD substrait")
        available = True
    except Exception as e:  # noqa: BLE001
        available = False
        reason = str(e).splitlines()[0][:120]
    if duckdb.__version__.startswith("2."):
        check(f"DuckDB {duckdb.__version__} has no substrait extension", not available,
              "" if available else reason)

    try:
        out = subprocess.run(
            ["uv", "run", "--no-project", "--with", "duckdb==1.5.6", "python", __file__, "--duckdb"],
            capture_output=True, text=True, timeout=600, env=os.environ)
        result = json.loads(out.stdout.strip().splitlines()[-1])
    except Exception as e:  # noqa: BLE001
        print(f"SKIP  DuckDB 1.5.6 part ({type(e).__name__}: needs uv and network)")
        return
    check(f"DuckDB {result['version']}'s Substrait plan drops every CTE and view name",
          not any(result["names"].values()), f"counts {result['names']}")
    check("a marker UDF and its string survive DuckDB's producer", result["udf"])
    check("a DuckDB macro is inlined, so it can't be a marker", not result["macro"])


if __name__ == "__main__":
    if "--duckdb" in sys.argv:
        duckdb_inline_part()
        sys.exit(0)
    datafusion_part()
    ddx_part()
    duckdb_part()
    print(f"\n{sum(CHECKS)}/{len(CHECKS)} checks hold")
    sys.exit(0 if all(CHECKS) else 1)
