import numpy as np
import pyarrow as pa
import pytest
import torch

from torch_sql.boundary.layout import IncompleteTensor, to_dense, to_table


@pytest.mark.parametrize(
    "t",
    [
        torch.randn(3, 4, 5, dtype=torch.float64),
        torch.tensor(2.5, dtype=torch.float64),
        torch.arange(6).reshape(2, 3),
        torch.tensor([[True, False], [False, True]]),
        torch.zeros(0, 3, dtype=torch.float64),
    ],
    ids=["float64", "scalar", "int64", "bool", "empty"],
)
def test_roundtrip(t):
    table = to_table(t)
    assert table.column_names[-1] == "v"
    assert torch.equal(to_dense(table, t.shape, t.dtype), t)


def test_scalar_has_a_key():
    assert to_table(torch.tensor(1.0, dtype=torch.float64)).column_names == ["i", "v"]


def test_order_does_not_matter():
    t = torch.randn(4, 3, dtype=torch.float64)
    table = to_table(t)
    shuffled = table.take(pa.array(np.random.default_rng(0).permutation(table.num_rows)))
    assert torch.equal(to_dense(shuffled, t.shape, t.dtype), t)


def test_missing_row_is_refused():
    table = to_table(torch.randn(2, 2, dtype=torch.float64)).slice(0, 3)
    with pytest.raises(IncompleteTensor, match="needs 4 rows"):
        to_dense(table, (2, 2), torch.float64)


def test_duplicate_row_is_refused():
    table = to_table(torch.randn(2, 2, dtype=torch.float64))
    dup = pa.concat_tables([table.slice(0, 3), table.slice(0, 1)])
    with pytest.raises(IncompleteTensor, match="twice"):
        to_dense(dup, (2, 2), torch.float64)


def test_null_is_refused():
    table = pa.table({"i0": pa.array([0, 1], pa.int64()), "v": pa.array([1.0, None])})
    with pytest.raises(IncompleteTensor, match="NULL"):
        to_dense(table, (2,), torch.float64)


def test_float32_is_refused_for_now():
    with pytest.raises(NotImplementedError, match="float32"):
        to_table(torch.randn(2))
