"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip integration-marked tests unless the user explicitly requests them via -m integration."""
    markexpr = getattr(config.option, "markexpr", "")
    if markexpr and "integration" in markexpr:
        return  # user asked for integration tests; let them run
    skip = pytest.mark.skip(reason="use '-m integration' to run integration tests")
    for item in items:
        if item.get_closest_marker("integration"):
            item.add_marker(skip)
