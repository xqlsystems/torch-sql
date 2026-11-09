"""The device frontend, ``device="sql"`` (design §6.4, §9.1).

Importing it names PyTorch's spare device slot, ``PrivateUse1``, ``sql``, so
``device="sql"`` and ``.to("sql")`` parse, and registers a ``torch.sql``
module for it. That module says the device isn't available as an
accelerator, so PyTorch's optimizers don't look for its streams. Without
it, ``torch.optim.Adam.step()`` raises, since renaming the slot makes it
PyTorch's current accelerator.
"""

import types

import torch

from . import tensor
from .environment import Environment
from .tensor import Tensor

__all__ = ["Environment", "Tensor"]


def _register():
    if torch._C._get_privateuse1_backend_name() == "sql":
        tensor._SQL = torch.device("sql")
        return
    torch.utils.rename_privateuse1_backend("sql")
    module = types.ModuleType("torch.sql")
    module.__doc__ = "torch-sql's device. Tensors on it are tables; see torch_sql."
    module.is_available = lambda: False
    module.device_count = lambda: 0
    # torch.manual_seed asks each device module for these. torch-sql draws
    # random numbers on the CPU, so the CPU seed is the one that matters.
    module._is_in_bad_fork = lambda: False
    module.manual_seed_all = lambda seed: None
    torch._register_device_module("sql", module)
    tensor._SQL = torch.device("sql")


_register()
