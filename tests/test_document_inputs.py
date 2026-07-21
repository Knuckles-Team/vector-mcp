from __future__ import annotations

from pathlib import Path

import pytest

import vector_mcp.document_inputs as document_inputs
from vector_mcp.document_inputs import resolve_document_inputs


def test_relative_file_is_resolved_beneath_configured_root(tmp_path: Path) -> None:
    nested = tmp_path / "approved"
    nested.mkdir()
    expected = nested / "document.md"
    expected.write_text("approved", encoding="utf-8")

    directory, paths = resolve_document_inputs(
        configured_root=tmp_path,
        include_configured_directory=False,
        relative_paths=["approved/document.md", "approved/document.md"],
        document_contents=None,
    )

    assert directory is None
    assert paths == [expected.resolve()]


@pytest.mark.parametrize(
    "untrusted_path",
    ["../outside.md", "/outside.md", "C:\\outside.md", "https://example.invalid/a"],
)
def test_host_paths_traversal_and_urls_are_rejected(
    tmp_path: Path, untrusted_path: str
) -> None:
    with pytest.raises(ValueError, match="relative"):
        resolve_document_inputs(
            configured_root=tmp_path,
            include_configured_directory=False,
            relative_paths=[untrusted_path],
            document_contents=None,
        )


def test_symbolic_links_are_rejected(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-document.md"
    outside.write_text("outside", encoding="utf-8")
    link = tmp_path / "link.md"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symbolic links are unavailable")

    with pytest.raises(ValueError, match="symbolic"):
        resolve_document_inputs(
            configured_root=tmp_path,
            include_configured_directory=False,
            relative_paths=["link.md"],
            document_contents=None,
        )


def test_explicit_files_and_inline_content_are_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = tmp_path / "bounded.md"
    document.write_text("12345", encoding="utf-8")
    monkeypatch.setattr(document_inputs, "MAX_DOCUMENT_FILE_BYTES", 4)

    with pytest.raises(ValueError, match="size limit"):
        resolve_document_inputs(
            configured_root=tmp_path,
            include_configured_directory=False,
            relative_paths=["bounded.md"],
            document_contents=None,
        )

    monkeypatch.setattr(document_inputs, "MAX_INLINE_CONTENT_BYTES", 4)
    with pytest.raises(ValueError, match="size limit"):
        resolve_document_inputs(
            configured_root=tmp_path,
            include_configured_directory=False,
            relative_paths=None,
            document_contents=["12345"],
        )


def test_directory_preflight_rejects_link_before_returning_root(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.md"
    target.write_text("data", encoding="utf-8")
    link = tmp_path / "alias.md"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symbolic links are unavailable")

    with pytest.raises(ValueError, match="symbolic"):
        resolve_document_inputs(
            configured_root=tmp_path,
            include_configured_directory=True,
            relative_paths=None,
            document_contents=None,
        )


def test_filesystem_inputs_require_an_explicit_configured_root() -> None:
    with pytest.raises(ValueError, match="root is unavailable"):
        resolve_document_inputs(
            configured_root="",
            include_configured_directory=True,
            relative_paths=None,
            document_contents=None,
        )


def test_inline_content_does_not_resolve_a_host_root() -> None:
    directory, paths = resolve_document_inputs(
        configured_root="",
        include_configured_directory=False,
        relative_paths=None,
        document_contents=["approved inline content"],
    )

    assert directory is None
    assert paths is None
