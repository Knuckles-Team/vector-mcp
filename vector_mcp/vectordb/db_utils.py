"""Small, fail-closed optional dependency boundary for vector providers."""

from __future__ import annotations

import importlib.util
import inspect
import sys
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from functools import wraps
from typing import Any, TypeVar, cast

from agent_utilities import get_logger

logger = get_logger(__name__)
T = TypeVar("T")


class ImportResult:
    """Outcome exposed by ``optional_import_block`` after the block exits."""

    def __init__(self) -> None:
        self._successful: bool | None = None

    @property
    def is_successful(self) -> bool:
        if self._successful is None:
            raise ValueError("optional_import_incomplete")
        return self._successful


@contextmanager
def optional_import_block() -> Generator[ImportResult, None, None]:
    """Contain a missing or import-hostile optional SDK without hiding other bugs."""

    result = ImportResult()
    try:
        yield result
        result._successful = True
    except (ImportError, SystemExit) as exc:
        logger.debug(
            "Optional dependency unavailable",
            extra={"exception_type": type(exc).__name__},
        )
        result._successful = False


def _missing(modules: str | Iterable[str]) -> list[str]:
    candidates = [modules] if isinstance(modules, str) else list(modules)
    missing: list[str] = []
    for name in candidates:
        package = str(name).split("<", 1)[0].split(">", 1)[0].strip()
        if package in sys.modules:
            if sys.modules[package] is None:
                missing.append(package)
            continue
        try:
            available = importlib.util.find_spec(package) is not None
        except (ImportError, ValueError):
            available = False
        if not available:
            missing.append(package)
    return missing


def require_optional_import(
    modules: str | Iterable[str],
    dep_target: str,
    *,
    except_for: str | Iterable[str] | None = None,
) -> Callable[[T], T]:
    """Block construction/calls when an explicitly selected SDK is absent.

    Class metadata and static inspection remain available, while construction
    fails with a stable installation hint. This keeps package import lightweight
    and never converts a missing provider into a silent no-op.
    """

    del except_for
    missing = _missing(modules)

    def decorator(target: T) -> T:
        if not missing:
            return target
        message = f"Optional provider unavailable; install vector-mcp[{dep_target}]"
        if inspect.isclass(target):
            cls = cast(type[Any], target)
            original = cls.__init__

            @wraps(original)
            def blocked_init(*_args: Any, **_kwargs: Any) -> None:
                raise ImportError(message)

            cls.__init__ = blocked_init
            return target

        callable_target = cast(Callable[..., Any], target)

        @wraps(callable_target)
        def blocked(*_args: Any, **_kwargs: Any) -> Any:
            raise ImportError(message)

        return cast(T, blocked)

    return decorator


__all__ = ["optional_import_block", "require_optional_import"]
