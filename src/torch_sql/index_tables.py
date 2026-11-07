"""Index tables (design §9.5): small tables that say which input positions
feed which output positions, so that an op relating its input and output
by position is a join in which every condition is an equality.

An index table isn't a tensor: it has named integer columns and no value.
It's registered once per environment for each distinct key, and kept for
the environment's life.
"""

from __future__ import annotations

import math

import numpy as np
import pyarrow as pa
import torch

from . import graph as g


def reshape_groups(src: tuple[int, ...], dst: tuple[int, ...]):
    """Pair runs of ``src``'s dimensions with runs of ``dst``'s that hold the
    same elements in C order, skipping dimensions of size 1.

    Returns ``(src_dims, dst_dims)`` pairs. A reshape relates positions only
    within a pair, so each pair needs its own small index table (or none,
    when it's one dimension to one of the same size)."""
    a = [d for d, n in enumerate(src) if n != 1]
    b = [d for d, n in enumerate(dst) if n != 1]
    pairs, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        gi, gj = [a[i]], [b[j]]
        pa_, pb_ = src[a[i]], dst[b[j]]
        i, j = i + 1, j + 1
        while pa_ != pb_:
            if pa_ < pb_:
                gi.append(a[i])
                pa_ *= src[a[i]]
                i += 1
            else:
                gj.append(b[j])
                pb_ *= dst[b[j]]
                j += 1
        pairs.append((gi, gj))
    if i != len(a) or j != len(b):
        raise ValueError(f"can't reshape {src} to {dst}")
    return pairs


def reshape_table(env, src_sizes: tuple[int, ...], dst_sizes: tuple[int, ...]) -> g.Table:
    """The table relating positions of a run of dimensions of sizes
    ``src_sizes`` to a run of sizes ``dst_sizes`` holding the same elements
    in C order: columns ``a0, a1, ...`` (source) and ``b0, b1, ...``
    (destination), one row per element."""
    key = ("reshape", src_sizes, dst_sizes)
    if key not in env._index_tables:
        n = math.prod(src_sizes)
        flat = np.arange(n, dtype=np.int64)
        cols = {}
        for prefix, sizes in (("a", src_sizes), ("b", dst_sizes)):
            for k, idx in enumerate(np.unravel_index(flat, sizes)):
                cols[f"{prefix}{k}"] = pa.array(idx.astype(np.int64))
        cols["v"] = pa.array(flat)  # unused; every table torch-sql registers has a v
        table = g.Table((n,), torch.int64)
        env.engine.register(table.name, pa.table(cols))
        env._index_tables[key] = table
    return env._index_tables[key]
