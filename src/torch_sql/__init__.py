"""torch-sql: PyTorch programs as SQL on XQL.

On ``device="sql"``, every tensor is a table, every op is recorded as a
relational plan, and the plans run on a SQL engine (DataFusion or DuckDB)
when a value is needed::

    import torch, torch_sql

    with torch_sql.Environment("duckdb"):
        x = torch.randn(4, 3, dtype=torch.float64, device="sql")
        y = (x @ x.T).sum()
        print(y.item())

See docs/design.md.
"""

from .device import Environment, Tensor
from .ops import OperatorNotFound


def sync(*tensors: Tensor) -> None:
    """Run what each tensor's value depends on, and keep it as a table."""
    for t in tensors:
        t._env.sync(t)


__all__ = ["Environment", "OperatorNotFound", "Tensor", "sync"]
