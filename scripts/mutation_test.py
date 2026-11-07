"""Mutation testing: does the test suite notice a plausible bug?

Each *mutant* is one exact text replacement in the source, a small change
that gives wrong numbers (or should give an error) without crashing. For
each one the script applies it, runs the unit tests and then the
property-based tests in ``tests/fuzz``, and records which suite, if any,
*kills* it (fails). A mutant that survives both shows a blind spot.

As in ddx's ``.github/scripts/mutation_test.py``:

- a mutant's text must match its file exactly once, so a mutant that a
  refactor made stale reports ``STALE`` instead of silently testing nothing;
- the source is restored on any exit, including Ctrl-C;
- an unmutated baseline runs first, and the script stops if it fails, so a
  kill is the mutant's doing.

Usage::

    uv run python scripts/mutation_test.py               # every mutant
    uv run python scripts/mutation_test.py cut-order ... # some, by name
    HYPOTHESIS_PROFILE=dev uv run python scripts/mutation_test.py

It prints a Markdown table, which CI can write to the job summary.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "torch_sql"


@dataclass(frozen=True)
class Mutant:
    name: str
    file: str
    old: str
    new: str
    why: str


MUTANTS = [
    Mutant(
        "grad-no-accumulate",
        "gradients.py",
        "t.grad = grad if t.grad is None else t.grad + grad",
        "t.grad = grad",
        "a second backward() replaces .grad instead of adding to it",
    ),
    Mutant(
        "stop-gradient-ignored",
        "gradients.py",
        "if isinstance(node, g.Select) and node.stop_gradient:",
        "if False:",
        ".detach() and no_grad() let gradients through",
    ),
    Mutant(
        "cut-order",
        "program.py",
        "for node in reversed(cuts):",
        "for node in cuts:",
        "cut points' cotangents are pulled back before they're complete",
    ),
    Mutant(
        "vjp-plus",
        "program.py",
        'g.Col(0, "v") * g.Col(1, "v")),\n            group=True,\n        )\n        segment',
        'g.Col(0, "v") + g.Col(1, "v")),\n            group=True,\n        )\n        segment',
        "a segment's vector-Jacobian product adds instead of multiplying",
    ),
    Mutant(
        "vjp-own-table",
        "program.py",
        "segment = {k: v for k, v in resolve.items() if k != node.id}",
        "segment = dict(resolve)",
        "a cut point's segment reads its own table, so nothing flows past it",
    ),
    Mutant(
        "float-literal-decimal",
        "sql.py",
        'return s if "e" in s else f"{s}e0"',
        "return s",
        "float literals without an exponent, which DuckDB reads as DECIMAL",
    ),
    Mutant(
        "broadcast-no-join",
        "ops/_build.py",
        "on.extend((sources[0], s) for s in sources[1:])",
        "pass",
        "elementwise ops cross join instead of joining on shared indices",
    ),
    Mutant(
        "cat-no-offset",
        "ops/shape.py",
        "        if offset:\n",
        "        if False:\n",
        "cat stacks every part at offset 0",
    ),
    Mutant(
        "keepdim-dropped",
        "ops/reductions.py",
        "        elif keepdim:\n            index.append(g.Lit(0, torch.int64))",
        "        elif keepdim:\n            index.append(g.Lit(1, torch.int64))",
        "a kept dimension gets index 1, not 0",
    ),
    Mutant(
        "amax-nan-unchecked",
        "ops/reductions.py",
        'if agg == "MAX" and compute == torch.float64:',
        "if False:",
        "amax trusts SQL's MAX with NaN",
    ),
    Mutant(
        "compare-ignores-nan",
        "ops/elementwise.py",
        "    plain = g.call(cmp, a, b)\n    if not floating:",
        "    plain = g.call(cmp, a, b)\n    if True:",
        "float comparisons use SQL's NaN ordering",
    ),
    Mutant(
        "minimum-ignores-nan",
        "ops/elementwise.py",
        "nan = any_of(isnan(x), isnan(y))",
        "nan = g.Lit(False, torch.bool)",
        "minimum trusts SQL's NaN ordering",
    ),
    Mutant(
        "clamp-lo-strict",
        "ops/elementwise.py",
        'out = case(g.call("<", out, lo), lo, out)',
        'out = case(g.call(">", out, lo), lo, out)',
        "clamp's lower bound is flipped",
    ),
    Mutant(
        "log-zero-branch",
        "ops/elementwise.py",
        'case(g.call(">", v, zero), g.call("ln", v),',
        'case(g.call(">=", v, zero), g.call("ln", v),',
        "log(0) reaches the engine's ln, which DuckDB refuses",
    ),
    Mutant(
        "layer-norm-no-eps",
        "ops/nn.py",
        'g.call("sqrt", v + scalar(eps, F64))',
        'g.call("sqrt", v)',
        "layer_norm forgets eps",
    ),
    Mutant(
        "softmax-unnormalized",
        "ops/nn.py",
        "lambda v, t: v / t)",
        "lambda v, t: v / (t + scalar(1e-6, F64)))",
        "softmax's normalizer is off by a little",
    ),
]


def run(args: list[str], timeout: int) -> tuple[bool, float]:
    """Whether ``pytest args`` passed, and how long it took."""
    start = time.monotonic()
    try:
        proc = subprocess.run(
            # A fixed seed, so the baseline and every mutant see the same
            # programs, and a kill is the mutant's doing.
            [
                sys.executable,
                "-m",
                "pytest",
                "-x",
                "-q",
                "-p",
                "no:cacheprovider",
                "--hypothesis-seed=0",
                *args,
            ],
            cwd=ROOT,
            capture_output=True,
            timeout=timeout,
            env={**os.environ, "HYPOTHESIS_PROFILE": os.environ.get("HYPOTHESIS_PROFILE", "ci")},
        )
        ok = proc.returncode == 0
    except subprocess.TimeoutExpired:
        ok = True  # nothing failed in time
    return ok, time.monotonic() - start


def _restoring(path: Path, original: str):
    """A SIGTERM handler that puts the source back before exiting."""

    def handler(*_):
        path.write_text(original)
        sys.exit(1)

    return handler


UNIT = ["tests", "--ignore=tests/fuzz", "--ignore=tests/test_nanogpt.py"]
FUZZ = ["tests/fuzz", "-m", "not pinned"]


def main(names: list[str]) -> int:
    timeout = int(os.environ.get("MUTANT_TIMEOUT", "900"))
    chosen = [m for m in MUTANTS if not names or m.name in names]
    print("baseline ...", file=sys.stderr)
    for suite in (UNIT, FUZZ):
        ok, _ = run(suite, timeout)
        if not ok:
            print(f"the baseline fails: pytest {' '.join(suite)}", file=sys.stderr)
            return 2
    rows, survived = [], 0
    for m in chosen:
        path = SRC / m.file
        original = path.read_text()
        if original.count(m.old) != 1:
            rows.append((m, "STALE", "", ""))
            continue
        previous = signal.signal(signal.SIGTERM, _restoring(path, original))
        try:
            path.write_text(original.replace(m.old, m.new))
            print(f"{m.name} ...", file=sys.stderr)
            unit_ok, unit_t = run(UNIT, timeout)
            fuzz_ok, fuzz_t = run(FUZZ, timeout)
        finally:
            path.write_text(original)
            signal.signal(signal.SIGTERM, previous)
        if not unit_ok and not fuzz_ok:
            status = "killed by both"
        elif not unit_ok:
            status = "killed by unit tests only"
        elif not fuzz_ok:
            status = "killed by fuzz tests only"
        else:
            status = "SURVIVED"
            survived += 1
        rows.append((m, status, f"{unit_t:.0f}s", f"{fuzz_t:.0f}s"))
    print("| mutant | what it breaks | result | unit | fuzz |")
    print("|---|---|---|---|---|")
    for m, status, ut, ft in rows:
        print(f"| `{m.name}` | {m.why} | {status} | {ut} | {ft} |")
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
