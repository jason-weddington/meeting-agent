"""Tests for meeting_agent.context."""

from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from meeting_agent.context import load_context

# ---------------------------------------------------------------------------
# Local path — happy path
# ---------------------------------------------------------------------------


def test_loads_single_file(tmp_path):
    """Single .md file appears in output with expected separator header."""
    md = tmp_path / "001-foo.md"
    md.write_text("Hello from foo.")

    result = load_context(tmp_path)

    assert "001-foo" in result
    assert "Hello from foo." in result
    assert "---" in result


def test_separator_format(tmp_path):
    """Separator is \\n\\n---\\n# <stem>\\n\\n<contents>."""
    md = tmp_path / "001-foo.md"
    md.write_text("Body text here.")

    result = load_context(tmp_path)

    assert result == "\n\n---\n# 001-foo\n\nBody text here."


def test_sorts_numeric_then_alpha(tmp_path):
    """001-*.md < 002-*.md < speaker_patterns.md (ASCII sort order)."""
    (tmp_path / "002-b.md").write_text("B content")
    (tmp_path / "speaker_patterns.md").write_text("Speaker content")
    (tmp_path / "001-a.md").write_text("A content")

    result = load_context(tmp_path)

    idx_001 = result.index("001-a")
    idx_002 = result.index("002-b")
    idx_sp = result.index("speaker_patterns")

    assert idx_001 < idx_002 < idx_sp


def test_multiple_files_all_present(tmp_path):
    """All .md files are included in the output."""
    (tmp_path / "001-a.md").write_text("Alpha")
    (tmp_path / "002-b.md").write_text("Beta")

    result = load_context(tmp_path)

    assert "Alpha" in result
    assert "Beta" in result


# ---------------------------------------------------------------------------
# Local path — filtering
# ---------------------------------------------------------------------------


def test_ignores_non_md_files(tmp_path):
    """Non-.md files (txt, png) are silently skipped."""
    (tmp_path / "001.md").write_text("Markdown content")
    (tmp_path / "notes.txt").write_text("Plain text")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")

    result = load_context(tmp_path)

    assert "Markdown content" in result
    assert "Plain text" not in result


def test_ignores_hidden_files(tmp_path):
    """Hidden files (starting with .) are not included even if .md."""
    (tmp_path / "001.md").write_text("Visible content")
    (tmp_path / ".hidden.md").write_text("Hidden content")
    # .DS_Store has no .md extension, so it would be excluded by glob anyway
    (tmp_path / ".DS_Store").write_bytes(b"\x00\x00")

    result = load_context(tmp_path)

    assert "Visible content" in result
    assert "Hidden content" not in result


def test_empty_dir_returns_empty_string(tmp_path):
    """Directory with no .md files → empty string."""
    (tmp_path / "notes.txt").write_text("ignored")

    result = load_context(tmp_path)

    assert result == ""


def test_truly_empty_dir_returns_empty_string(tmp_path):
    """Completely empty directory → empty string."""
    result = load_context(tmp_path)
    assert result == ""


# ---------------------------------------------------------------------------
# Local path — error handling
# ---------------------------------------------------------------------------


def test_missing_local_path_raises(tmp_path):
    """Non-existent path raises FileNotFoundError."""
    missing = tmp_path / "does_not_exist"

    with pytest.raises(FileNotFoundError):
        load_context(missing)


def test_missing_local_path_as_string_raises(tmp_path):
    """Non-existent path passed as str raises FileNotFoundError."""
    missing = str(tmp_path / "does_not_exist")

    with pytest.raises(FileNotFoundError):
        load_context(missing)


# ---------------------------------------------------------------------------
# Type flexibility
# ---------------------------------------------------------------------------


def test_accepts_path_object(tmp_path):
    """Passing a Path object works."""
    (tmp_path / "001.md").write_text("Via Path")

    result = load_context(tmp_path)

    assert "Via Path" in result


def test_accepts_string(tmp_path):
    """Passing a str path works."""
    (tmp_path / "001.md").write_text("Via string")

    result = load_context(str(tmp_path))

    assert "Via string" in result


def test_accepts_string_and_path_yield_same_result(tmp_path):
    """str and Path forms of the same location produce identical output."""
    (tmp_path / "001.md").write_text("Consistent")

    via_path = load_context(tmp_path)
    via_str = load_context(str(tmp_path))

    assert via_path == via_str


# ---------------------------------------------------------------------------
# S3 URI
# ---------------------------------------------------------------------------


def _make_mock_s3_client(keys: list[str], contents: dict[str, bytes]) -> MagicMock:
    """Build a mock boto3 S3 client with canned list + get responses."""
    mock_client = MagicMock()
    mock_client.list_objects_v2.return_value = {
        "Contents": [{"Key": k} for k in keys],
    }

    def _get_object(Bucket: str, Key: str) -> dict:  # noqa: N803
        return {"Body": BytesIO(contents[Key])}

    mock_client.get_object.side_effect = _get_object
    return mock_client


def test_s3_uri_calls_list_and_get(tmp_path):
    """S3 URI triggers list_objects_v2 + get_object calls in sorted key order."""
    keys = ["context/001-a.md", "context/002-b.md"]
    file_bodies = {
        "context/001-a.md": b"First file",
        "context/002-b.md": b"Second file",
    }
    mock_client = _make_mock_s3_client(keys, file_bodies)

    with patch("boto3.client", return_value=mock_client):
        result = load_context("s3://my-bucket/context/")

    mock_client.list_objects_v2.assert_called_once_with(Bucket="my-bucket", Prefix="context/")
    assert mock_client.get_object.call_count == 2

    # Keys passed to get_object must be in sorted order
    call_keys = [call.kwargs["Key"] for call in mock_client.get_object.call_args_list]
    assert call_keys == sorted(keys)

    assert "First file" in result
    assert "Second file" in result
    assert result.index("001-a") < result.index("002-b")


def test_s3_uri_filters_non_md_keys():
    """Keys that don't end in .md are excluded from S3 load."""
    keys = ["prefix/001.md", "prefix/readme.txt", "prefix/image.png"]
    file_bodies = {"prefix/001.md": b"Markdown only"}
    mock_client = _make_mock_s3_client(keys, file_bodies)

    with patch("boto3.client", return_value=mock_client):
        result = load_context("s3://bucket/prefix/")

    assert "Markdown only" in result
    # get_object called only for the .md key
    assert mock_client.get_object.call_count == 1


def test_s3_uri_filters_hidden_keys():
    """Keys whose basename starts with '.' are excluded from S3 load."""
    keys = ["prefix/001.md", "prefix/.hidden.md"]
    file_bodies = {"prefix/001.md": b"Visible"}
    mock_client = _make_mock_s3_client(keys, file_bodies)

    with patch("boto3.client", return_value=mock_client):
        result = load_context("s3://bucket/prefix/")

    assert "Visible" in result
    assert mock_client.get_object.call_count == 1


def test_s3_empty_bucket_returns_empty_string():
    """S3 prefix with no .md files → empty string."""
    mock_client = MagicMock()
    mock_client.list_objects_v2.return_value = {"Contents": []}

    with patch("boto3.client", return_value=mock_client):
        result = load_context("s3://bucket/empty/")

    assert result == ""


def test_s3_missing_contents_key_returns_empty_string():
    """S3 response with no 'Contents' key (empty bucket) → empty string."""
    mock_client = MagicMock()
    mock_client.list_objects_v2.return_value = {}  # no "Contents" key

    with patch("boto3.client", return_value=mock_client):
        result = load_context("s3://bucket/prefix/")

    assert result == ""


# ---------------------------------------------------------------------------
# Scheme validation
# ---------------------------------------------------------------------------


def test_rejects_ftp_scheme():
    """ftp:// URI raises ValueError."""
    with pytest.raises(ValueError, match="ftp"):
        load_context("ftp://example.com/files/")


def test_rejects_http_scheme():
    """http:// URI raises ValueError."""
    with pytest.raises(ValueError, match="http"):
        load_context("http://example.com/files/")


def test_rejects_https_scheme():
    """https:// URI raises ValueError."""
    with pytest.raises(ValueError, match="https"):
        load_context("https://example.com/files/")
