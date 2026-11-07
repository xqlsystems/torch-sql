"""Each lowered op on PyTorch's OpInfo samples (design §14).

OpInfo is the database of sample inputs PyTorch tests its own backends
with. Each sample runs on the CPU and on ``device="sql"``, in float64, and
the results must match. A sample torch-sql refuses (``NotImplementedError``,
which ``OperatorNotFound`` is) must be one this file lists in ``REFUSED``,
with the reason.
"""

import pytest
import torch
from torch.testing._internal.common_methods_invocations import op_db
from torch.utils._pytree import tree_map

import torch_sql

F64 = torch.float64

#: The OpInfo entries this PR lowers, by name and variant.
OPS = [
    "add",
    "sub",
    "rsub",
    "mul",
    "div",
    "neg",
    "exp",
    "log",
    "sqrt",
    "rsqrt",
    "tanh",
    "sigmoid",
    "abs",
    "sin",
    "cos",
    "reciprocal",
    "pow",
    "nn.functional.relu",
    "maximum",
    "minimum",
    "clamp",
    "clamp_min",
    "clamp_max",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "where",
    "logical_not",
    "logical_and",
    "logical_or",
    "isnan",
    "sum",
    "mean",
    "amax",
    "amin",
    "mm",
    "bmm",
    "addmm",
    "t",
    "permute",
    "unsqueeze",
    "squeeze",
    "expand",
    "cat",
    "full_like",
    "zeros_like",
    "ones_like",
    "new_full",
    "new_zeros",
    "new_ones",
    "nn.functional.linear",
    "nn.functional.mse_loss",
    "transpose",
    "select",
    "narrow",
    "any",
    "all",
    "reshape",
    "view",
]

#: (op, why) for samples torch-sql refuses on purpose. Each refusal raises
#: NotImplementedError rather than giving a different answer (P3).
REFUSED = {
    "div": "rounding_mode",
    "nn.functional.linear": "a 1-d input, which PyTorch's matmul squeezes in place",
}

CASES = [o for o in op_db if o.name in OPS]


def to_sql(env, x):
    return env.tensor(x) if isinstance(x, torch.Tensor) else x


@pytest.mark.parametrize("info", CASES, ids=lambda o: f"{o.name}.{o.variant_test_name}".strip("."))
def test_op(engine, info):
    env = torch_sql.Environment(engine)
    dtype = F64 if F64 in info.dtypes else (torch.int64 if torch.int64 in info.dtypes else None)
    if dtype is None:
        pytest.skip("no float64 or int64 variant")
    ran = refused = 0
    for sample in info.sample_inputs("cpu", dtype, requires_grad=False):
        if any(
            isinstance(t, torch.Tensor) and t.dtype not in (F64, torch.int64, torch.bool)
            for t in torch.utils._pytree.tree_leaves((sample.input, sample.args, sample.kwargs))
        ):
            continue
        expected = info(sample.input, *sample.args, **sample.kwargs)
        args = tree_map(lambda x: to_sql(env, x), (sample.input, sample.args, sample.kwargs))
        with env:
            try:
                got = info(args[0], *args[1], **args[2])
            except NotImplementedError as e:
                assert info.name in REFUSED, f"{sample}: {e}"
                refused += 1
                continue
            got = tree_map(lambda x: x.cpu() if isinstance(x, torch_sql.Tensor) else x, got)
        torch.testing.assert_close(
            got,
            expected,
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
            msg=lambda m, s=sample: f"{s}\n{m}",
        )
        ran += 1
    if not ran:
        assert refused, "no sample ran"
        pytest.skip(f"every sample refused: {REFUSED[info.name]}")
