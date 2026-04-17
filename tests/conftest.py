"""Pytest configuration and shared fixtures.

Two concerns live here:
  * Inject a ``sounddevice`` stand-in into ``sys.modules`` *before* any test
    imports ``meeting_agent.audio``. sounddevice raises
    ``OSError: PortAudio library not found`` at import time on hosts without
    PortAudio (CI, headless dispatch agents).
  * Skip integration-marked tests by default. Run them explicitly with
    ``pytest -m integration``.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

if "sounddevice" not in sys.modules:
    sys.modules["sounddevice"] = MagicMock()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip integration-marked tests unless ``-m integration`` is explicitly passed."""
    markexpr = getattr(config.option, "markexpr", "")
    if markexpr and "integration" in markexpr:
        return
    skip = pytest.mark.skip(reason="use '-m integration' to run integration tests")
    for item in items:
        if item.get_closest_marker("integration"):
            item.add_marker(skip)
