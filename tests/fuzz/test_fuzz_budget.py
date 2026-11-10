"""The cut budget bounds every query (design §9.6).

``torch_sql.program`` promises that no query it hands an engine plans more
``SELECT`` blocks than the engine's ``cut_budget``, unless a single node
reads more tables than the budget allows. Values are checked elsewhere
(``test_cut_budget_changes_nothing``); this checks the size of every query
the program sends, forward and backward, by measuring the graph that
``torch_sql.sql.query`` writes out.
"""

from __future__ import annotations

import pytest
import torch
from fuzz_programs import leaf_tensors, output_of, programs, run
from harness import make_engine, refusals_are_skips, weights
from hypothesis import event, given, note
from hypothesis import strategies as st

import torch_sql
from torch_sql import graph as g
from torch_sql import sql

#: What backward() adds to a segment: a cut point's vector-Jacobian product
#: is one SELECT more, reading the cotangent's table as well (the loss's SUM
#: adds only the SELECT).
WRAPPERS = 2


def inlined_size(root: g.Node, saved) -> tuple[int, int]:
    """The number of SELECT blocks ``root``'s query plans with every CTE
    written into each place that reads it (a table counts once per read),
    and the most inputs any one node of it reads."""
    sizes, widest = {}, 0
    for node in g.topological([root], lambda n: saved.get(n.id, n)):
        if isinstance(node, g.Table):
            sizes[node.id] = 1
            continue
        reads = [saved.get(i.id, i) for i in node.inputs]
        widest = max(widest, len(reads))
        sizes[node.id] = 1 + sum(sizes[i.id] for i in reads)
    return sizes[saved.get(root.id, root).id], widest


@pytest.fixture
def measured(monkeypatch):
    """Every query's inlined size and widest node, as it's written."""
    seen = []
    original = sql.query

    def query(root, saved=None, **kwargs):
        seen.append(inlined_size(root, saved or {}))
        return original(root, saved, **kwargs)

    monkeypatch.setattr(sql, "query", query)
    return seen


@given(program=programs(modes=("smooth",)), budget=st.sampled_from([2, 3, 5, 8, 13]))
def test_no_query_passes_the_budget(measured, program, budget):
    note(repr(program))
    measured.clear()
    with refusals_are_skips(), torch_sql.Environment(make_engine("datafusion", budget)):
        leaves = leaf_tensors(program, "sql", requires_grad=True)
        out = run(program, leaves)[output_of(program)]
        if out.requires_grad:
            (out * weights(out.shape).to("sql")).sum().backward()
            event("forward and backward")
        out.detach().cpu()
    for size, widest in measured:
        allowed = max(budget, 1 + widest) + WRAPPERS
        note(f"query of {size} SELECTs, widest node reads {widest}; budget {budget}")
        assert size <= allowed, f"a query of {size} SELECTs passes the budget {budget}"
    event(f"largest query / budget: {max((s for s, _ in measured), default=0) / budget:.1f}")


@pytest.mark.pinned
def test_a_cotangent_of_many_parts_stays_within_the_budget(measured):
    """A cut point read by k later segments gets k cotangent tables. When
    ``program.total`` summed them inside the cut point's vector-Jacobian
    product, as a UNION ALL and a GROUP BY, that query grew with k, past
    the budget: here, a value read by six others at budget 3 gave a query
    of 12 SELECTs. The sum is now saved first, as a leaf's gradient of many
    parts is (issue #23)."""
    budget = 3
    with torch_sql.Environment(make_engine("datafusion", budget)):
        x = torch.tensor([1.0, 2.0], dtype=torch.float64).to("sql").requires_grad_()
        h = torch.tanh(torch.tanh(x))
        loss = sum(torch.tanh(torch.tanh(h * k)).sum() for k in range(1, 7))
        measured.clear()
        loss.backward()
    widest = max(w for _, w in measured)
    largest = max(s for s, _ in measured)
    assert largest <= max(budget, 1 + widest) + WRAPPERS, f"a query of {largest} SELECTs"
