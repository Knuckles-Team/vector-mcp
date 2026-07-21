"""Bounded, root-confined document input resolution.

MCP callers select files relative to an administrator-configured root.  Absolute
paths, URLs, links, and traversal are deliberately outside the public contract so
tool arguments and traces never expose host filesystem details.
"""

from __future__ import annotations

from pathlib import Path, PureWindowsPath
from typing import Final

MAX_DOCUMENT_FILES: Final = 1_000
MAX_DOCUMENT_FILE_BYTES: Final = 64 * 1024 * 1024
MAX_DOCUMENT_TOTAL_BYTES: Final = 512 * 1024 * 1024
MAX_INLINE_CONTENT_BYTES: Final = 16 * 1024 * 1024


def resolve_document_inputs(
    *,
    configured_root: str | Path,
    include_configured_directory: bool,
    relative_paths: list[str] | None,
    document_contents: list[str] | None,
) -> tuple[Path | None, list[Path] | None]:
    """Resolve bounded document inputs without accepting host paths from callers."""

    contents = list(document_contents or [])
    if len(contents) > MAX_DOCUMENT_FILES:
        raise ValueError("Too many inline documents")

    inline_bytes = 0
    for content in contents:
        if not isinstance(content, str):
            raise ValueError("Inline document content must be text")
        inline_bytes += len(content.encode("utf-8"))
        if inline_bytes > MAX_INLINE_CONTENT_BYTES:
            raise ValueError("Inline document content exceeded its size limit")

    requested_paths = list(relative_paths or [])
    if len(requested_paths) > MAX_DOCUMENT_FILES:
        raise ValueError("Too many configured documents")

    requires_root = include_configured_directory or bool(requested_paths)
    if not requires_root:
        return None, None

    rendered_root = str(configured_root or "").strip()
    if not rendered_root:
        raise ValueError("Configured document root is unavailable")
    root = Path(rendered_root).expanduser()
    try:
        root = root.resolve(strict=True)
    except OSError:
        if include_configured_directory or requested_paths:
            raise ValueError("Configured document root is unavailable") from None
        return None, None
    try:
        invalid_root = root.is_symlink() or not root.is_dir()
    except OSError:
        raise ValueError("Configured document root is invalid") from None
    if invalid_root:
        raise ValueError("Configured document root is invalid")

    resolved_paths: list[Path] = []
    seen_paths: set[Path] = set()
    selected_bytes = 0
    for raw in requested_paths:
        rendered = str(raw).strip()
        candidate_relative = Path(rendered)
        windows_path = PureWindowsPath(rendered)
        if (
            not rendered
            or len(rendered) > 4_096
            or "\x00" in rendered
            or candidate_relative.is_absolute()
            or windows_path.is_absolute()
            or bool(windows_path.drive)
            or ".." in candidate_relative.parts
            or ".." in windows_path.parts
            or "://" in rendered
        ):
            raise ValueError("Document path must be relative to the configured root")

        cursor = root
        for part in candidate_relative.parts:
            cursor = cursor / part
            try:
                linked = cursor.is_symlink()
            except OSError:
                raise ValueError("Configured document could not be inspected") from None
            if linked:
                raise ValueError("Document path may not traverse symbolic links")
        try:
            candidate = (root / candidate_relative).resolve(strict=True)
        except OSError:
            raise ValueError("Configured document was not found") from None
        try:
            invalid_candidate = (
                not candidate.is_relative_to(root) or not candidate.is_file()
            )
        except OSError:
            raise ValueError("Configured document could not be inspected") from None
        if invalid_candidate:
            raise ValueError("Document path escaped the configured root")
        if candidate in seen_paths:
            continue
        try:
            size = candidate.stat().st_size
        except OSError:
            raise ValueError("Configured document could not be inspected") from None
        if size > MAX_DOCUMENT_FILE_BYTES:
            raise ValueError("Configured document exceeded its size limit")
        selected_bytes += size
        if selected_bytes > MAX_DOCUMENT_TOTAL_BYTES:
            raise ValueError("Configured documents exceeded their size limit")
        seen_paths.add(candidate)
        resolved_paths.append(candidate)

    directory: Path | None = None
    if include_configured_directory:
        total_bytes = 0
        file_count = 0
        try:
            for candidate in root.rglob("*"):
                if candidate.is_symlink():
                    raise ValueError(
                        "Configured document root contains a symbolic link"
                    )
                if not candidate.is_file():
                    continue
                file_count += 1
                if file_count > MAX_DOCUMENT_FILES:
                    raise ValueError("Configured document root contains too many files")
                size = candidate.stat().st_size
                if size > MAX_DOCUMENT_FILE_BYTES:
                    raise ValueError("Configured document exceeded its size limit")
                total_bytes += size
                if total_bytes > MAX_DOCUMENT_TOTAL_BYTES:
                    raise ValueError("Configured document root exceeded its size limit")
        except OSError:
            raise ValueError(
                "Configured document root could not be inspected"
            ) from None
        directory = root

    return directory, resolved_paths or None
