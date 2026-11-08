import pytest

from torch_sql import engines


@pytest.fixture(params=sorted(engines.ENGINES))
def engine(request):
    """Each test that takes an engine runs on every engine."""
    return engines.engine(request.param)
