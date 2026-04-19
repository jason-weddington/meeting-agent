"""End-to-end V3.0 MCP KB-grounding integration test.

Exercises the full grounded pipeline path with a minimal in-process FastMCP
server (``_fake_kb_server.py``) and a deterministic stub Bedrock client.  No
real Bedrock round-trips are made — the stub scripts a two-turn sequence
(toolUse → toolResult → end_turn) to verify the *wiring*, not LLM quality.

Real components exercised
-------------------------
* :class:`~meeting_agent.mcp_client.MCPClient` — real subprocess launch,
  stdio handshake, ``list_tools``, ``call_tool``.
* :class:`~meeting_agent.llm.BedrockClient` ``respond_stream`` tool-use loop
  — ``toolConfig`` assembly, ``toolUse`` parsing, ``toolResult`` appending.
* :class:`~meeting_agent.pipeline.Pipeline` MCP lifecycle — ``mcp_ready``,
  ``mcp_stopped`` trace events.
* :mod:`meeting_agent.trace` — structured JSON trace log flushed to a temp dir.

Stubbed components
------------------
* ``boto3.client("bedrock-runtime")`` — returns a mock whose
  ``converse_stream`` returns pre-scripted event streams (no AWS calls).
* ``StreamingASR`` — yields one canned utterance ("What is in the roadmap?"),
  then stops; the pipeline exits naturally.
* ``_build_classifier`` — always returns ``Decision(full_answer)`` so the
  pipeline responds to the utterance.
* ``TTS`` — returns a 100-sample silent audio chunk; no synthesis.
* ``audio.record_chunks`` / ``audio.play`` — no-op I/O stubs.

Marking
-------
``@pytest.mark.integration`` — skipped by default; run with::

    uv run pytest -m integration tests/integration/test_mcp_grounding.py -v

Runtime
-------
~5–10 s: FastMCP server subprocess starts, MCP handshake completes, one
pipeline turn runs.  No model downloads required.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from meeting_agent.asr import Utterance
from meeting_agent.classifier import Decision
from meeting_agent.llm import ProjectContext
from meeting_agent.mcp_client import MCPServerConfig
from meeting_agent.pipeline import Pipeline, PipelineConfig

# ---------------------------------------------------------------------------
# Canned test data
# ---------------------------------------------------------------------------

# The query the stub Bedrock will ask about (must match a key in _fake_kb_server)
_QUERY = "roadmap"

# The canned KB result the fake server will return for a "roadmap" query
_EXPECTED_KB_CONTENT = "Q2 roadmap: finish V3 grounding. Deadline 2026-05-15."

# The scripted response text in the second Bedrock stream — references KB content
_STUB_RESPONSE = f"According to the KB: {_EXPECTED_KB_CONTENT}"

# Path to the fake KB server script
_FAKE_SERVER_PATH = str(Path(__file__).parent / "_fake_kb_server.py")


# ---------------------------------------------------------------------------
# Stub Bedrock stream builders
# ---------------------------------------------------------------------------


def _make_tool_use_stream(tool_use_id: str = "tu-001", tool_name: str = "kb_search") -> dict:
    """Build a fake converse_stream response that requests a tool call.

    Produces the Bedrock event sequence:
    ``contentBlockStart(toolUse) → contentBlockDelta(input JSON) → messageStop(tool_use)``
    """
    return {
        "stream": iter(
            [
                {
                    "contentBlockStart": {
                        "contentBlockIndex": 0,
                        "start": {
                            "toolUse": {
                                "toolUseId": tool_use_id,
                                "name": tool_name,
                            }
                        },
                    }
                },
                {
                    "contentBlockDelta": {
                        "contentBlockIndex": 0,
                        "delta": {"toolUse": {"input": f'{{"query": "{_QUERY}"}}'}},
                    }
                },
                {"messageStop": {"stopReason": "tool_use"}},
            ]
        )
    }


def _make_end_turn_stream(response_text: str = _STUB_RESPONSE) -> dict:
    """Build a fake converse_stream response that ends the conversation.

    Produces: ``contentBlockDelta(text) → messageStop(end_turn)``
    """
    return {
        "stream": iter(
            [
                {
                    "contentBlockDelta": {
                        "contentBlockIndex": 0,
                        "delta": {"text": response_text},
                    }
                },
                {"messageStop": {"stopReason": "end_turn"}},
            ]
        )
    }


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_mcp_grounded_pipeline_full_path(tmp_path: Path) -> None:
    """Full grounded path: fake MCP server → stub Bedrock → trace events.

    Scenario
    --------
    1. Pipeline starts; MCPClient connects to the fake KB server subprocess.
    2. ASR yields one utterance: "What is in the roadmap?".
    3. Classifier returns ``full_answer`` → pipeline calls ``_stream_and_play``.
    4. BedrockClient fetches tools from the real MCPClient and attaches a
       ``toolConfig`` to the first ``converse_stream`` call.
    5. First stub stream returns ``tool_use`` requesting ``kb_search(query="roadmap")``.
    6. MCPClient calls the fake KB server; it returns the canned roadmap text.
    7. BedrockClient appends a ``toolResult`` message and calls
       ``converse_stream`` a second time.
    8. Second stub stream returns ``end_turn`` with text referencing the KB result.
    9. Pipeline finaliser stops the MCPClient and emits ``mcp_stopped``.

    Assertions
    ----------
    * ``mcp_ready`` trace event fired with ``tool_count >= 1`` and
      ``"kb_search"`` in ``tool_names``.
    * First ``converse_stream`` call carried ``toolConfig`` containing
      ``"kb_search"`` in its tool specs.
    * Second ``converse_stream`` call's messages include a ``toolResult``
      block whose ``content`` references the canned KB output.
    * ``tool_invoked`` trace event fired with ``tool_name == "kb_search"``
      and ``is_error == False``.
    * Trace event order: ``mcp_ready`` → ``tool_invoked`` → ``mcp_stopped``.
    """
    # ------------------------------------------------------------------
    # 1. Stub boto3 bedrock-runtime client — two scripted streams
    # ------------------------------------------------------------------
    tool_use_id = "tu-test-001"
    mock_bedrock = MagicMock()
    mock_bedrock.converse_stream.side_effect = [
        _make_tool_use_stream(tool_use_id=tool_use_id),
        _make_end_turn_stream(),
    ]

    # ------------------------------------------------------------------
    # 2. Pipeline config — real MCPClient pointing at the fake KB server
    # ------------------------------------------------------------------
    config = PipelineConfig(
        context=ProjectContext(system_prompt="You are a helpful meeting assistant."),
        mcp_server=MCPServerConfig(
            command=sys.executable,
            args=(_FAKE_SERVER_PATH,),
        ),
        trace_enabled=True,
        trace_log_dir=tmp_path,
    )
    pipeline = Pipeline(config)

    # ------------------------------------------------------------------
    # 3. Mock ASR — yields one utterance asking about roadmap, then stops
    # ------------------------------------------------------------------
    mock_utterance = Utterance(
        text="What is in the roadmap?",
        start_s=0.0,
        end_s=2.0,
        avg_logprob=-0.3,  # well above -1.0 threshold → passes gate
        no_speech_prob=0.05,  # well below 0.6 threshold → passes gate
        compression_ratio=1.2,  # well below 2.4 threshold → passes gate
    )
    mock_asr = MagicMock()
    mock_asr.transcribe_stream.return_value = iter([mock_utterance])

    # ------------------------------------------------------------------
    # 4. Mock classifier — always full_answer
    # ------------------------------------------------------------------
    mock_classifier = MagicMock()
    mock_classifier.classify.return_value = Decision(
        speaker="Jason",
        action="full_answer",
        confidence=0.95,
    )

    # ------------------------------------------------------------------
    # 5. Mock TTS — returns silent audio chunk (no synthesis needed)
    # ------------------------------------------------------------------
    _fake_audio = np.zeros(1600, dtype=np.float32)
    mock_tts = MagicMock()
    mock_tts.stream_synthesize.return_value = [_fake_audio]

    # ------------------------------------------------------------------
    # 6. Run the pipeline in a thread (it blocks on ASR)
    # ------------------------------------------------------------------
    errors: list[Exception] = []

    def _run_pipeline() -> None:
        try:
            with (
                patch("boto3.client", return_value=mock_bedrock),
                patch("meeting_agent.audio.record_chunks", return_value=iter([])),
                patch("meeting_agent.pipeline.StreamingASR", return_value=mock_asr),
                patch("meeting_agent.pipeline._build_classifier", return_value=mock_classifier),
                patch("meeting_agent.pipeline.TTS", return_value=mock_tts),
                patch("meeting_agent.audio.play"),
                patch(
                    "meeting_agent.pipeline._install_exception_log",
                    return_value=MagicMock(),
                ),
            ):
                pipeline.run()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=_run_pipeline, daemon=True)
    thread.start()
    thread.join(timeout=60)

    assert not thread.is_alive(), (
        "Pipeline thread did not complete within 60 s — possible MCP connection hang"
    )
    assert not errors, f"Pipeline raised an unexpected error: {errors[0]!r}"

    # ------------------------------------------------------------------
    # 7. Parse trace events (listener.stop() called in pipeline finally)
    # ------------------------------------------------------------------
    trace_file = tmp_path / "trace.jsonl"
    assert trace_file.exists(), "trace.jsonl was not written — trace may be disabled"
    event_records = [
        json.loads(line) for line in trace_file.read_text().splitlines() if line.strip()
    ]
    event_names = [r["event"] for r in event_records]

    # ------------------------------------------------------------------
    # 8. Assert mcp_ready event — fired with kb_search in tool_names
    # ------------------------------------------------------------------
    assert "mcp_ready" in event_names, f"Expected 'mcp_ready' trace event; got: {event_names}"
    mcp_ready = next(r for r in event_records if r["event"] == "mcp_ready")
    assert mcp_ready["tool_count"] >= 1, (
        f"mcp_ready should report >= 1 tool; got tool_count={mcp_ready['tool_count']}"
    )
    assert "kb_search" in mcp_ready["tool_names"], (
        f"mcp_ready tool_names should include 'kb_search'; got: {mcp_ready['tool_names']}"
    )

    # ------------------------------------------------------------------
    # 9. Assert toolConfig in first converse_stream call
    # ------------------------------------------------------------------
    assert mock_bedrock.converse_stream.call_count >= 2, (
        "Expected at least 2 converse_stream calls (tool_use + end_turn); "
        f"got {mock_bedrock.converse_stream.call_count}"
    )
    first_call_kwargs = mock_bedrock.converse_stream.call_args_list[0][1]
    assert "toolConfig" in first_call_kwargs, (
        "First converse_stream call must include toolConfig when MCP client is active"
    )
    tool_names_in_config = [t["toolSpec"]["name"] for t in first_call_kwargs["toolConfig"]["tools"]]
    assert "kb_search" in tool_names_in_config, (
        f"toolConfig must expose 'kb_search'; got tools: {tool_names_in_config}"
    )

    # ------------------------------------------------------------------
    # 10. Assert toolResult in second converse_stream call
    # ------------------------------------------------------------------
    second_call_kwargs = mock_bedrock.converse_stream.call_args_list[1][1]
    messages = second_call_kwargs["messages"]
    tool_result_texts = [
        content_block["toolResult"]["content"][0]["text"]
        for msg in messages
        for content_block in msg["content"]
        if isinstance(content_block, dict) and "toolResult" in content_block
    ]
    assert tool_result_texts, (
        "Second converse_stream call must include at least one toolResult message; "
        f"messages: {json.dumps(messages, indent=2)}"
    )
    assert any(_EXPECTED_KB_CONTENT in text for text in tool_result_texts), (
        f"toolResult content should reference canned KB output {_EXPECTED_KB_CONTENT!r}; "
        f"got: {tool_result_texts}"
    )

    # ------------------------------------------------------------------
    # 11. Assert tool_invoked trace event
    # ------------------------------------------------------------------
    assert "tool_invoked" in event_names, f"Expected 'tool_invoked' trace event; got: {event_names}"
    tool_invoked = next(r for r in event_records if r["event"] == "tool_invoked")
    assert tool_invoked["tool_name"] == "kb_search", (
        f"tool_invoked should report tool_name='kb_search'; got: {tool_invoked['tool_name']}"
    )
    assert not tool_invoked["is_error"], (
        f"tool_invoked should report is_error=False; got: {tool_invoked['is_error']}"
    )

    # ------------------------------------------------------------------
    # 12. Assert mcp_stopped event fired
    # ------------------------------------------------------------------
    assert "mcp_stopped" in event_names, f"Expected 'mcp_stopped' trace event; got: {event_names}"

    # ------------------------------------------------------------------
    # 13. Assert trace event order: mcp_ready → tool_invoked → mcp_stopped
    # ------------------------------------------------------------------
    mcp_ready_idx = event_names.index("mcp_ready")
    tool_invoked_idx = event_names.index("tool_invoked")
    mcp_stopped_idx = event_names.index("mcp_stopped")
    assert mcp_ready_idx < tool_invoked_idx < mcp_stopped_idx, (
        f"Expected mcp_ready({mcp_ready_idx}) < tool_invoked({tool_invoked_idx}) "
        f"< mcp_stopped({mcp_stopped_idx}) in trace; full event sequence: {event_names}"
    )
