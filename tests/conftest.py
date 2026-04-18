"""Pytest configuration and shared fixtures.

Three concerns live here:
  * Inject a ``sounddevice`` stand-in into ``sys.modules`` *before* any test
    imports ``meeting_agent.audio``. sounddevice raises
    ``OSError: PortAudio library not found`` at import time on hosts without
    PortAudio (CI, headless dispatch agents).
  * Inject ``mlx.*`` stubs so that ``mlx_audio.tts.utils`` is importable on
    non-Apple-Silicon hosts (Linux CI, dispatch agents). The native ``libmlx.so``
    is absent there; the stubs allow the module to load while unit tests mock
    ``load_model`` before any real inference is attempted.
  * Skip integration-marked tests by default. Run them explicitly with
    ``pytest -m integration``.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

if "sounddevice" not in sys.modules:
    sys.modules["sounddevice"] = MagicMock()

# mlx requires Apple Silicon hardware (libmlx.so). Stub it out so that
# mlx_audio.tts.utils can be imported in unit tests on Linux/CI hosts.
for _mlx_mod in ("mlx", "mlx.core", "mlx.nn", "mlx.utils", "mlx.optimizers"):
    if _mlx_mod not in sys.modules:
        sys.modules[_mlx_mod] = MagicMock()

# ollama requires a running Ollama daemon and the native C extension. Stub it
# out so that meeting_agent.classifier can be imported on headless CI hosts
# where the real package may not be installed.
if "ollama" not in sys.modules:
    sys.modules["ollama"] = MagicMock()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip integration-marked tests unless ``-m integration`` is explicitly passed."""
    markexpr = getattr(config.option, "markexpr", "")
    if markexpr and "integration" in markexpr:
        return
    skip = pytest.mark.skip(reason="use '-m integration' to run integration tests")
    for item in items:
        if item.get_closest_marker("integration"):
            item.add_marker(skip)
