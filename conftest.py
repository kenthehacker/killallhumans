"""Repository-wide pytest safety/isolation fixtures."""
from __future__ import annotations

import os
import sys

# Promotion bootstraps deliberately reject source-adjacent bytecode, including
# ignored/untracked caches.  Keep repository tests from creating such import
# alternatives while they exercise the bootstraps in-process.
sys.dont_write_bytecode = True

import pytest

from planning.artifact_cache import CACHE_ROOT_ENV


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    """Classify the default isolated tier as ``unit`` consistently.

    Tests opt out by carrying an explicit slow, benchmark, or live marker.
    This keeps ``pytest -m unit`` useful without requiring marker boilerplate
    on hundreds of existing deterministic tests.
    """

    excluded_tiers = ("slow", "benchmark", "live")
    for item in items:
        if not any(item.get_closest_marker(name) for name in excluded_tiers):
            item.add_marker(pytest.mark.unit)


@pytest.fixture(scope="session", autouse=True)
def _isolated_artifact_cache(tmp_path_factory):
    """Keep every test process out of the developer's persistent cache."""

    previous = os.environ.get(CACHE_ROOT_ENV)
    cache_root = tmp_path_factory.mktemp("aigp-test-artifacts")
    os.environ[CACHE_ROOT_ENV] = str(cache_root)
    try:
        yield cache_root
    finally:
        if previous is None:
            os.environ.pop(CACHE_ROOT_ENV, None)
        else:
            os.environ[CACHE_ROOT_ENV] = previous
