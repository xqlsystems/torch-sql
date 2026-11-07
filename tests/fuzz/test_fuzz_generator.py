"""The generator itself: a property test is only as good as what it draws."""

from collections import Counter

from fuzz_programs import OPS, programs
from hypothesis import HealthCheck, given, settings


def test_every_op_is_drawn():
    """Every op in the catalog appears in a thousand programs. A step the
    generator can't infer a shape for is dropped silently, so an op whose
    screens fail on ``meta`` tensors would otherwise never be tested."""
    seen = Counter()

    @settings(
        max_examples=1000,
        derandomize=True,
        database=None,
        suppress_health_check=list(HealthCheck),
    )
    @given(programs())
    def draw(program):
        seen.update(s.op for s in program.steps)

    draw()
    missing = sorted(set(OPS) - set(seen))
    assert not missing, f"never drawn: {missing}"
