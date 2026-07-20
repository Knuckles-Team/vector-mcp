from __future__ import annotations

import pytest

from vector_mcp.vectordb.db_utils import optional_import_block, require_optional_import


def test_missing_class_dependency_blocks_only_construction() -> None:
    @require_optional_import("dependency_that_cannot_exist_7f14", "test")
    class OptionalBackend:
        def __init__(self) -> None:
            self.ready = True

        @staticmethod
        def schema_name() -> str:
            return "opaque"

    assert OptionalBackend.schema_name() == "opaque"
    with pytest.raises(ImportError, match=r"vector-mcp\[test\]"):
        OptionalBackend()


def test_optional_import_block_contains_third_party_system_exit() -> None:
    with optional_import_block() as result:
        raise SystemExit(0)

    assert result.is_successful is False
