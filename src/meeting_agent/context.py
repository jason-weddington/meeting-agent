"""Context loader — reads a directory of Markdown files into a system-prompt string."""

from __future__ import annotations

import urllib.parse
from pathlib import Path


def load_context(location: str | Path) -> str:
    """Load a directory of markdown files into a single concatenated string.

    Args:
        location: Local filesystem path (str or Path), or an s3:// URI
            of the form ``s3://bucket-name/prefix/``.

    Returns:
        All ``*.md`` files in the directory, concatenated in sorted name
        order, each prefixed by a separator block::

            ---
            # <filename without extension>

            <file contents>

        Returns an empty string if the directory has no ``.md`` files.

    Raises:
        FileNotFoundError: if a local path does not exist.
        ValueError: if ``location`` starts with a scheme other than ``s3://``.
    """
    if isinstance(location, Path):
        return _load_local(location)

    parsed = urllib.parse.urlparse(str(location))
    if parsed.scheme == "s3":
        return _load_s3(parsed.netloc, parsed.path.lstrip("/"))
    elif parsed.scheme == "":
        return _load_local(Path(location))
    else:
        raise ValueError(
            f"Unsupported URI scheme {parsed.scheme!r}. "
            "Only local filesystem paths and s3:// URIs are supported."
        )


def _load_local(path: Path) -> str:
    """Load markdown files from a local directory.

    Args:
        path: Directory to read from.

    Returns:
        Concatenated markdown content.

    Raises:
        FileNotFoundError: if the path does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Context directory not found: {path}")

    md_files = sorted(
        (f for f in path.glob("*.md") if not f.name.startswith(".")),
        key=lambda f: f.name,
    )

    parts: list[str] = []
    for md_file in md_files:
        stem = md_file.stem
        contents = md_file.read_text(encoding="utf-8")
        parts.append(f"\n\n---\n# {stem}\n\n{contents}")

    return "".join(parts)


def _load_s3(bucket: str, prefix: str) -> str:
    """Load markdown files from an S3 prefix.

    Args:
        bucket: S3 bucket name.
        prefix: Key prefix (path within the bucket).

    Returns:
        Concatenated markdown content.
    """
    import boto3  # noqa: PLC0415

    client = boto3.client("s3")
    response = client.list_objects_v2(Bucket=bucket, Prefix=prefix)

    raw_contents = response.get("Contents", [])
    md_keys = sorted(
        key
        for key in (obj["Key"] for obj in raw_contents)
        if key.endswith(".md") and not key.split("/")[-1].startswith(".")
    )

    parts: list[str] = []
    for key in md_keys:
        filename = key.split("/")[-1]
        stem = filename[: -len(".md")]
        obj = client.get_object(Bucket=bucket, Key=key)
        body: str = obj["Body"].read().decode("utf-8")
        parts.append(f"\n\n---\n# {stem}\n\n{body}")

    return "".join(parts)
