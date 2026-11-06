"""S7: does the XQL stack install and run on ARM, inside Samudra's container?

Question (design.md §16, S7; samudra-on-beta.md §4, stage 0). Samudra trains
on Empire AI's Beta cluster, whose CPUs are ARM (NVIDIA Grace), inside a
container built on NVIDIA's PhysicsNeMo 26.05 image. torch-sql's co-processor
puts xarray-sql, ddx and einfold, and their engines DataFusion and DuckDB,
into that container. Do they install there, next to Samudra's own pinned
packages, and do they run?

This machine is x86_64 with no ARM emulation (no qemu-user, binfmt, docker or
podman), so the spike answers what can be answered without ARM hardware and
writes down what a Beta smoke job must still check:

1. Reads Samudra's container recipe and lock file for the Python version,
   the base image and the pinned versions of the packages XQL shares with it
   (numpy, pyarrow, xarray, zarr, pandas, dask).
2. Resolves Samudra's pins together with xarray-sql, ddxdb, einfold and
   DuckDB for `aarch64-unknown-linux-gnu` with `uv pip compile`, with and
   without the packages' engine extras.
3. Checks on PyPI that every package in that resolution has a wheel that
   installs on linux-aarch64 for CPython 3.12 (a manylinux wheel no newer
   than glibc 2.34, or a pure-Python wheel), so nothing builds from source in
   the container.
4. Runs the smoke test (s7_arm_packaging/smoke.py) on this machine, in an
   environment with exactly Samudra's pins, as the Beta job would run it.

Run (from the repository root; on NixOS, first source a file that puts
libstdc++ on LD_LIBRARY_PATH; needs uv and network):

    uv run --project docs/spikes python docs/spikes/s7_arm_packaging.py [SAMUDRA_CHECKOUT]

Without a Samudra checkout, step 1 uses the pins recorded below and says so.

What the Beta smoke job runs (stage 0 of samudra-on-beta.md), on one host:

    srun --container-image=$IMAGE_REF bash -lc '
      . /workspace/.venv/bin/activate
      uv pip install xarray-sql==0.6.0 ddxdb==0.4.6 einfold==0.2.0 duckdb==1.5.6
      python smoke.py'

It prints the platform, glibc and versions, then PASS or FAIL for an
xarray-sql query and a ddx gradient, and findings for einfold and DuckDB's
substrait extension.

Findings (2026-11-05; Samudra at 4f86be4):

- Samudra's container is PhysicsNeMo 26.05 with Python 3.12, and its lock
  pins numpy 1.26.4 (`numpy<2`), zarr 2.18.7 (`zarr<3`), xarray 2025.1.2,
  pyarrow 25.0.1, pandas 2.2.3 and dask 2025.2.0. torch comes from the image.
- Everything resolves for aarch64 with those pins. XQL adds five packages
  to Samudra's environment (xarray-sql, ddxdb, einfold, DataFusion, DuckDB),
  and each has an installable aarch64 wheel: the first four as `abi3` wheels
  for manylinux 2.28, DuckDB as a cp312 wheel. Every other package they need
  (pyarrow, numpy, xarray and the rest) is one Samudra already locks at the
  same version. Nothing builds from source. So ARM is not the blocker.
- The blocker is DataFusion versions. xarray-sql 0.6.0 pins datafusion
  54.0.0; ddxdb[datafusion] 0.4.6 and einfold[datafusion] 0.2.0 pin 55.0.0.
  With the extras, the resolution fails; without them, it resolves to
  DataFusion 54. On 54, ddx's gradients work (ddxdb's own pin is only on its
  extra), but einfold can't install: DataFusion 54's `SessionContext` has no
  `with_extensions`. einfold works on DataFusion 55 with Samudra's numpy and
  pyarrow. xarray-sql's main branch already pins datafusion 55, but also
  `einfold<0.2`, so its next release needs that range widened.
- DuckDB 2.0 has no community substrait extension (HTTP 404), and an
  unpinned install picks DuckDB 2.0. ddx's DuckDB backend needs 1.5.6, which
  ddxdb[duckdb] pins. With 1.5.6 the extension installs (it downloads at run
  time, so a Beta job needs network access or a preinstalled extension).
- The smoke test passes on x86_64 with exactly Samudra's pins: numpy 1.26
  and zarr 2 don't stop xarray-sql or ddx. (Run this way, its environment is
  layered on the spikes' environment, so the torch it reports is the spikes'
  CPU build, not one Samudra would use.)

What remains for Beta: run smoke.py in the ARM image on one host (glibc, the
Grace CPU and the image's own torch are untested here), and decide how the
image gets DuckDB's substrait extension without network access at run time.
"""

import json
import os
import subprocess
import sys
import tempfile
import tomllib
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHECKS = []

#: Samudra's pins at 4f86be4, used when no checkout is given.
RECORDED = {"numpy": "1.26.4", "pyarrow": "25.0.1", "xarray": "2025.1.2",
            "zarr": "2.18.7", "numcodecs": "0.15.1", "pandas": "2.2.3", "dask": "2025.2.0"}
XQL = {"xarray-sql": "0.6.0", "ddxdb": "0.4.6", "einfold": "0.2.0", "duckdb": "1.5.6"}
PLATFORM = "aarch64-unknown-linux-gnu"
MAX_GLIBC = (2, 34)


def check(name, ok, detail=""):
    CHECKS.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def samudra_pins(checkout):
    if checkout is None or not (checkout / "uv.lock").exists():
        print(f"NOTE  no Samudra checkout; using the pins recorded at 4f86be4: {RECORDED}")
        return dict(RECORDED), dict(RECORDED)
    dockerfile = (checkout / "containers" / "Dockerfile.physicsnemo-26.05").read_text()
    check("Samudra's container is built on PhysicsNeMo 26.05",
          "FROM nvcr.io/nvidia/physicsnemo/physicsnemo:26.05" in dockerfile)
    project = tomllib.loads((checkout / "pyproject.toml").read_text())["project"]
    check("Samudra requires Python 3.12", project["requires-python"] == ">=3.12,<3.13",
          project["requires-python"])
    lock = tomllib.loads((checkout / "uv.lock").read_text())
    locked = {p["name"]: p["version"] for p in lock["package"] if "version" in p}
    shared = {k: locked[k] for k in RECORDED}
    check("Samudra pins numpy < 2 and zarr < 3",
          shared["numpy"].startswith("1.") and shared["zarr"].startswith("2."), str(shared))
    return shared, locked


def compile_for(requirements):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("\n".join(requirements))
    out = subprocess.run(["uv", "pip", "compile", "--python-version", "3.12",
                          "--python-platform", PLATFORM, "--no-header", "--no-annotate", f.name],
                         capture_output=True, text=True)
    os.unlink(f.name)
    return out


def resolution(pins, locked):
    base = [f"{k}=={v}" for k, v in pins.items()]
    plain = compile_for(base + [f"{k}=={v}" for k, v in XQL.items()])
    # Pin every package Samudra already locks to its locked version, so the
    # resolution is what Samudra's environment plus XQL would be.
    first = dict(line.split("==") for line in plain.stdout.split() if "==" in line)
    base += [f"{k}=={locked[k]}" for k in first if k in locked and k not in pins]
    plain = compile_for(base + [f"{k}=={v}" for k, v in XQL.items()])
    check("Samudra's pins and the XQL packages resolve for aarch64", plain.returncode == 0,
          plain.stderr.strip().splitlines()[-1] if plain.returncode else "")
    resolved = dict(line.split("==") for line in plain.stdout.split() if "==" in line)
    check("without extras, the resolution picks DataFusion 54",
          resolved.get("datafusion") == "54.0.0", resolved.get("datafusion", "none"))

    extras = dict(XQL, **{"ddxdb[datafusion]": XQL["ddxdb"], "einfold[datafusion]": XQL["einfold"]})
    extras.pop("ddxdb"), extras.pop("einfold")
    conflict = compile_for(base + [f"{k}=={v}" for k, v in extras.items()])
    check("with ddxdb's and einfold's DataFusion extras, it doesn't resolve",
          conflict.returncode != 0 and "datafusion" in conflict.stderr,
          "xarray-sql pins 54, the extras pin 55")
    return resolved


def installable(filename):
    """Whether a wheel installs on linux-aarch64 CPython 3.12, glibc <= 2.34."""
    if not filename.endswith(".whl"):
        return False
    tags = filename[:-4].split("-")
    python, abi, plats = tags[-3], tags[-2], tags[-1]
    if plats == "any":
        return "py3" in python.split(".")
    if abi not in ("cp312", "abi3") or "aarch64" not in plats:
        return False
    for plat in plats.split("."):
        if plat.startswith("manylinux_") and plat.endswith("_aarch64"):
            major, minor = plat.split("_")[1:3]
            if (int(major), int(minor)) <= MAX_GLIBC:
                return True
        if plat == "manylinux2014_aarch64":
            return True
    return False


def wheels(resolved, locked):
    """Checks the packages XQL adds to Samudra's environment; the ones Samudra
    already locks, its ARM image already installs."""
    added = {k: v for k, v in resolved.items() if locked.get(k) != v}
    missing, xql = [], {}
    for name, ver in sorted(added.items()):
        with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{ver}/json") as r:
            files = [f["filename"] for f in json.load(r)["urls"]]
        ok = [f for f in files if installable(f)]
        if not ok:
            missing.append(f"{name}=={ver}")
        if name in ("xarray-sql", "ddxdb", "einfold", "datafusion", "duckdb", "pyarrow"):
            xql[name] = ok[0].rsplit("-", 3)[1:] if ok else None
    check(f"all {len(added)} packages XQL adds have an aarch64 wheel for CPython 3.12",
          not missing, f"missing: {missing}" if missing else ", ".join(sorted(added)))
    for name, tags in xql.items():
        print(f"      {name}: {'-'.join(tags) if tags else 'none'}")


def emulation():
    tools = [t for t in ["qemu-aarch64", "qemu-aarch64-static", "docker", "podman"]
             if any((Path(p) / t).exists() for p in os.environ.get("PATH", "").split(":"))]
    binfmt = Path("/proc/sys/fs/binfmt_misc/qemu-aarch64").exists()
    print(f"NOTE  ARM emulation here: {tools or 'none'}, binfmt qemu-aarch64: {binfmt}")


def smoke(pins):
    reqs = [f"{k}=={v}" for k, v in pins.items()] + [f"{k}=={v}" for k, v in XQL.items()]
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("\n".join(reqs))
    out = subprocess.run(["uv", "run", "--no-project", "--python", "3.12", "--with-requirements",
                          f.name, "python", str(HERE / "s7_arm_packaging" / "smoke.py")],
                         capture_output=True, text=True, timeout=900,
                         env={k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"})
    os.unlink(f.name)
    report = json.loads(out.stdout.strip().splitlines()[-1])
    for line in out.stdout.strip().splitlines()[:-1]:
        print("      " + line)
    check("the smoke test passes with exactly Samudra's pins (on this machine)",
          out.returncode == 0)
    check("einfold can't install on DataFusion 54 (no SessionContext.with_extensions)",
          "with_extensions" in report.get("einfold in a DataFusion session", ""))
    check("DuckDB 1.5.6 loads the substrait extension",
          report.get("DuckDB substrait extension", "").startswith("ok"))


if __name__ == "__main__":
    checkout = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    pins, locked = samudra_pins(checkout)
    resolved = resolution(pins, locked)
    wheels(resolved, locked)
    emulation()
    smoke(pins)
    print(f"\n{sum(CHECKS)}/{len(CHECKS)} checks hold")
    sys.exit(0 if all(CHECKS) else 1)
