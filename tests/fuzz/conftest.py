"""Hypothesis profiles for the property-based tests.

``HYPOTHESIS_PROFILE`` picks one: ``ci`` (the default, quick enough for every
pull request), ``dev`` (a few minutes) or ``nightly`` (a soak). A failure
prints the program and a ``@reproduce_failure`` blob, and Hypothesis keeps
it in ``.hypothesis/`` so the next run tries it first.
"""

import os

from hypothesis import HealthCheck, settings

_common = dict(
    deadline=None,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.filter_too_much,
        HealthCheck.function_scoped_fixture,
    ],
    print_blob=True,
)
settings.register_profile("ci", max_examples=25, **_common)
settings.register_profile("dev", max_examples=300, **_common)
settings.register_profile("nightly", max_examples=3000, **_common)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))
