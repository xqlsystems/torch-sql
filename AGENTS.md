# Agent guidelines for torch-sql

## Status

torch-sql is a skunkworks prototype. Work proceeds in three stages: discussion,
then a design doc, then implementation. Don't skip ahead without agreement.

## Writing rules

These apply to the design doc and to all public-facing writing (README, docs,
docstrings, issues, PR descriptions, blog posts).

- **Introduce every concept before you use it.** When you mention a project,
  term, or idea for the first time (for example torchax, ddx, einfold,
  xarray-sql, XQL, `__torch_dispatch__`, ATen, DataFusion), first say what it
  is in a sentence or two, then use it. Don't assume the reader already knows.
- When a later section depends on an earlier concept, link back to where it was
  introduced.

## GitHub identity

Agents start every GitHub comment, issue and pull request description with an
emoji and a name, so the record shows who said what. torch-sql's agent uses
**🔥 Claude (torch-sql):**. Other XQL agents use 🌀 Claude: and
🪭 Claude (einfold):, so don't reuse those.

Comment freely on xqlsystems repositories. Don't comment on upstream projects
(DataFusion, DuckDB, Arrow, PyTorch, and so on) without asking first.
