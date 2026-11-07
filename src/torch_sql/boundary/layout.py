"""Long tables to dense tensors and back (design §6.6, §8.1).

A tensor of shape ``(n0, n1, ...)`` is a table with one ``BIGINT`` column
per dimension, ``i0``, ``i1``, ..., and a value column ``v``. A table that
holds a scalar has a key column ``i``, always 0, since ddx needs a column
that identifies a table's rows (see :class:`torch_sql.graph.Table`).

Reading a table back never sorts it. An engine returns rows in any order,
so :func:`to_dense` scatters each row to its position, and checks that the
table is *complete* (§8.3): every position exactly once.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import torch

from ..graph import check_dtype, index_columns

_ARROW = {
    torch.float64: pa.float64(),
    torch.int64: pa.int64(),
    torch.bool: pa.bool_(),
}

#: The key column of a table holding a scalar.
SCALAR_KEY = "i"


class IncompleteTensor(RuntimeError):
    """A table meant to hold a complete tensor misses a position, or holds
    one twice."""


def to_table(t: torch.Tensor) -> pa.Table:
    """A CPU tensor as a long table, in C order."""
    check_dtype(t.dtype)
    t = t.detach().contiguous()
    values = pa.array(t.reshape(-1).numpy(), _ARROW[t.dtype])
    if t.dim() == 0:
        return pa.table({SCALAR_KEY: pa.array([0], pa.int64()), "v": values})
    # Each index column, computed per dimension without materializing
    # np.indices for the whole shape at once.
    cols = {}
    n = t.numel()
    for k, size in enumerate(t.shape):
        inner = int(np.prod(t.shape[k + 1 :], dtype=np.int64))
        cols[f"i{k}"] = pa.array((np.arange(n, dtype=np.int64) // inner) % size, pa.int64())
    cols["v"] = values
    return pa.table(cols)


def to_dense(table, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """A long table (anything that exports an Arrow stream) as a CPU tensor
    of ``shape``, scattered by its index columns."""
    check_dtype(dtype)
    table = table if isinstance(table, pa.Table) else pa.table(table)
    shape = tuple(int(s) for s in shape)
    n = int(np.prod(shape, dtype=np.int64))
    if table.num_rows != n:
        raise IncompleteTensor(
            f"a tensor of shape {shape} needs {n} rows; the table has {table.num_rows}"
        )
    v = table.column("v")
    if v.null_count:
        raise IncompleteTensor(f"{v.null_count} NULL values in a tensor of shape {shape}")
    values = v.to_numpy()
    if not shape:
        return torch.tensor(values[0], dtype=dtype)
    flat = np.zeros(n, dtype=np.int64)
    for k, (col, size) in enumerate(zip(index_columns(len(shape)), shape, strict=True)):
        idx = table.column(col).to_numpy()
        if n and (idx.min() < 0 or idx.max() >= size):
            raise IncompleteTensor(f"index column {col} out of range for size {size}")
        flat = flat * size + idx if k else idx.astype(np.int64, copy=True)
    out = np.empty(n, dtype=values.dtype)
    out[flat] = values
    # Every position once: n rows, each in range, and no two alike.
    if n and np.bincount(flat, minlength=n).max() != 1:
        raise IncompleteTensor(f"a position appears twice in a tensor of shape {shape}")
    return torch.from_numpy(out.reshape(shape)).to(dtype)
