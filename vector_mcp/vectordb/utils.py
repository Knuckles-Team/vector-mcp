#!/usr/bin/python
# coding: utf-8
import inspect
import re
import sys
import logging
from typing import Any
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from functools import wraps
from logging import getLogger
from typing import Generic, Optional, TypeVar
from packaging import version
from typing import (
    TYPE_CHECKING,
)
from typing_extensions import (
    ParamSpec,
)
from .base import QueryResults

__all__ = [
    "optional_import_block",
    "require_optional_import",
]


logger = getLogger(__name__)


def get_logger(name: str):
    logger = getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def filter_results_by_distance(
    results: QueryResults, distance_threshold: float = -1
) -> QueryResults:
    """Filters results based on a distance threshold.

    Args:
        results: QueryResults | The query results. List[List[Tuple[Document, float]]]
        distance_threshold: The maximum distance allowed for results.

    Returns:
        QueryResults | A filtered results containing only distances smaller than the threshold.
    """
    if distance_threshold > 0:
        results = [
            [(key, value) for key, value in data if value < distance_threshold]
            for data in results
        ]

    return results


def chroma_results_to_query_results(
    data_dict: dict[str, list[list[Any]]], special_key="distances"
) -> QueryResults:
    """Converts a dictionary with list-of-list values to a list of tuples.

    Args:
        data_dict: A dictionary where keys map to lists of lists or None.
        special_key: The key in the dictionary containing the special values
                    for each tuple.

    Returns:
        A list of tuples, where each tuple contains a sub-dictionary with
        some keys from the original dictionary and the value from the
        special_key.

    Example:
        ```python
        data_dict = {
            "key1s": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "key2s": [["a", "b", "c"], ["c", "d", "e"], ["e", "f", "g"]],
            "key3s": None,
            "key4s": [["x", "y", "z"], ["1", "2", "3"], ["4", "5", "6"]],
            "distances": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        }

        results = [
            [
                ({"key1": 1, "key2": "a", "key4": "x"}, 0.1),
                ({"key1": 2, "key2": "b", "key4": "y"}, 0.2),
                ({"key1": 3, "key2": "c", "key4": "z"}, 0.3),
            ],
            [
                ({"key1": 4, "key2": "c", "key4": "1"}, 0.4),
                ({"key1": 5, "key2": "d", "key4": "2"}, 0.5),
                ({"key1": 6, "key2": "e", "key4": "3"}, 0.6),
            ],
            [
                ({"key1": 7, "key2": "e", "key4": "4"}, 0.7),
                ({"key1": 8, "key2": "f", "key4": "5"}, 0.8),
                ({"key1": 9, "key2": "g", "key4": "6"}, 0.9),
            ],
        ]
        ```
    """
    keys = [
        key
        for key in data_dict
        if key != special_key
        and data_dict[key] is not None
        and isinstance(data_dict[key][0], list)
    ]
    result = []
    data_special_key = data_dict[special_key]

    for i in range(len(data_special_key)):
        sub_result = []
        for j, distance in enumerate(data_special_key[i]):
            sub_dict = {}
            for key in keys:
                if len(data_dict[key]) > i:
                    sub_dict[key[:-1]] = data_dict[key][i][
                        j
                    ]  # remove 's' in the end from key
            sub_result.append((sub_dict, distance))
        result.append(sub_result)

    return result


@dataclass
class ModuleInfo:
    name: str
    min_version: str | None = None
    max_version: str | None = None
    min_inclusive: bool = False
    max_inclusive: bool = False

    def is_in_sys_modules(self) -> str | None:
        """Check if the module is installed and satisfies the version constraints

        Returns:
            None if the module is installed and satisfies the version constraints, otherwise a message indicating the issue.

        """
        import importlib.util
        from importlib.metadata import version as get_version, PackageNotFoundError

        # Check if module spec can be found (installed)
        if not importlib.util.find_spec(self.name):
            return f"'{self.name}' is not installed."

        # Check version if constraints exist
        if self.min_version or self.max_version:
            try:
                installed_version = get_version(self.name)
            except PackageNotFoundError:
                # Fallback: try to import and check __version__ (for some packages like older sklearn)
                try:
                    module = importlib.import_module(self.name)
                    installed_version = getattr(module, "__version__", None)
                except ImportError:
                    return f"'{self.name}' is found but version could not be retrieved."

            if installed_version is None:
                return f"'{self.name}' is installed, but the version is not available."

            # Convert to version object for comparison
            installed_ver = version.parse(installed_version)

            if self.min_version:
                min_ver = version.parse(self.min_version)
                msg = f"'{self.name}' is installed, but the installed version {installed_version} is too low (required '{self}')."
                if not self.min_inclusive and installed_ver == min_ver:
                    return msg
                if self.min_inclusive and installed_ver < min_ver:
                    return msg

            if self.max_version:
                max_ver = version.parse(self.max_version)
                msg = f"'{self.name}' is installed, but the installed version {installed_version} is too high (required '{self}')."
                if not self.max_inclusive and installed_ver == max_ver:
                    return msg
                if self.max_inclusive and installed_ver > max_ver:
                    return msg

        return None

    def __repr__(self) -> str:
        s = self.name
        if self.min_version:
            s += (
                f">={self.min_version}"
                if self.min_inclusive
                else f">{self.min_version}"
            )
        if self.max_version:
            s += (
                f"<={self.max_version}"
                if self.max_inclusive
                else f"<{self.max_version}"
            )
        return s

    @classmethod
    def from_str(cls, module_info: str) -> "ModuleInfo":
        """Parse a string to create a ModuleInfo object

        Args:
            module_info (str): A string containing the module name and optional version constraints

        Returns:
            ModuleInfo: A ModuleInfo object with the parsed information

        Raises:
            ValueError: If the module information is invalid
        """
        pattern = re.compile(r"^(?P<name>[a-zA-Z0-9-_]+)(?P<constraint>.*)$")
        match = pattern.match(module_info.strip())

        if not match:
            raise ValueError(f"Invalid package information: {module_info}")

        name = match.group("name")
        constraints = match.group("constraint").strip()
        min_version = max_version = None
        min_inclusive = max_inclusive = False

        if constraints:
            constraint_pattern = re.findall(r"(>=|<=|>|<)([0-9\.]+)?", constraints)

            if not all(version for _, version in constraint_pattern):
                raise ValueError(f"Invalid module information: {module_info}")

            for operator, version in constraint_pattern:
                if operator == ">=":
                    min_version = version
                    min_inclusive = True
                elif operator == "<=":
                    max_version = version
                    max_inclusive = True
                elif operator == ">":
                    min_version = version
                    min_inclusive = False
                elif operator == "<":
                    max_version = version
                    max_inclusive = False
                else:
                    raise ValueError(f"Invalid package information: {module_info}")

        return ModuleInfo(
            name=name,
            min_version=min_version,
            max_version=max_version,
            min_inclusive=min_inclusive,
            max_inclusive=max_inclusive,
        )


class Result:
    def __init__(self) -> None:
        self._failed: bool | None = None

    @property
    def is_successful(self) -> bool:
        if self._failed is None:
            raise ValueError("Result not set")
        return not self._failed


@contextmanager
def optional_import_block() -> Generator[Result, None, None]:
    """Guard a block of code to suppress ImportErrors

    A context manager to temporarily suppress ImportErrors.
    Use this to attempt imports without failing immediately on missing modules.

    Example:
    ```python
    with optional_import_block():
        import some_module
        import some_other_module
    ```
    """
    result = Result()
    try:
        yield result
        result._failed = False
    except ImportError as e:
        # Ignore ImportErrors during this context
        logger.debug(f"Ignoring ImportError: {e}")
        result._failed = True


def get_missing_imports(modules: str | Iterable[str]) -> dict[str, str]:
    """Get missing modules from a list of module names

    Args:
        modules (Union[str, Iterable[str]]): Module name or list of module names

    Returns:
        List of missing module names
    """
    if isinstance(modules, str):
        modules = [modules]

    module_infos = [ModuleInfo.from_str(module) for module in modules]
    x = {m.name: m.is_in_sys_modules() for m in module_infos}
    return {k: v for k, v in x.items() if v}


T = TypeVar("T")
G = TypeVar("G", bound=Callable[..., Any] | type)
F = TypeVar("F", bound=Callable[..., Any])


class PatchObject(ABC, Generic[T]):
    def __init__(self, o: T, missing_modules: dict[str, str], dep_target: str):
        if not self.accept(o):
            raise ValueError(f"Cannot patch object of type {type(o)}")

        self.o = o
        self.missing_modules = missing_modules
        self.dep_target = dep_target

    @classmethod
    @abstractmethod
    def accept(cls, o: Any) -> bool: ...

    @abstractmethod
    def patch(self, except_for: Iterable[str]) -> T: ...

    def get_object_with_metadata(self) -> Any:
        return self.o

    @property
    def msg(self) -> str:
        o = self.get_object_with_metadata()
        plural = len(self.missing_modules) > 1
        fqn = f"{o.__module__}.{o.__name__}" if hasattr(o, "__module__") else o.__name__
        # modules_str = ", ".join([f"'{m}'" for m in self.missing_modules])
        msg = f"{'Modules' if plural else 'A module'} needed for {fqn} {'are' if plural else 'is'} missing:\n"
        for _, status in self.missing_modules.items():
            msg += f" - {status}\n"
        msg += f"Please install {'them' if plural else 'it'} using:\n'pip install vector-mcp[{self.dep_target}]'"
        return msg

    def copy_metadata(self, retval: T) -> None:
        """Copy metadata from original object to patched object

        Args:
            retval: Patched object

        """
        o = self.o
        if hasattr(o, "__doc__"):
            retval.__doc__ = o.__doc__
        if hasattr(o, "__name__"):
            retval.__name__ = o.__name__  # type: ignore[attr-defined]
        if hasattr(o, "__module__"):
            retval.__module__ = o.__module__

    _registry: list[type["PatchObject[Any]"]] = []

    @classmethod
    def register(cls) -> Callable[[type["PatchObject[Any]"]], type["PatchObject[Any]"]]:
        def decorator(subclass: type["PatchObject[Any]"]) -> type["PatchObject[Any]"]:
            cls._registry.append(subclass)
            return subclass

        return decorator

    @classmethod
    def create(
        cls,
        o: T,
        *,
        missing_modules: dict[str, str],
        dep_target: str,
    ) -> Optional["PatchObject[T]"]:
        for subclass in cls._registry:
            if subclass.accept(o):
                return subclass(o, missing_modules, dep_target)
        return None


@PatchObject.register()
class PatchCallable(PatchObject[F]):
    @classmethod
    def accept(cls, o: Any) -> bool:
        return inspect.isfunction(o) or inspect.ismethod(o)

    def patch(self, except_for: Iterable[str]) -> F:
        if self.o.__name__ in except_for:
            return self.o

        f: Callable[..., Any] = self.o

        # @wraps(f.__call__)  # type: ignore[operator]
        @wraps(f)
        def _call(*args: Any, **kwargs: Any) -> Any:
            raise ImportError(self.msg)

        self.copy_metadata(_call)  # type: ignore[arg-type]

        return _call  # type: ignore[return-value]


@PatchObject.register()
class PatchStatic(PatchObject[F]):
    @classmethod
    def accept(cls, o: Any) -> bool:
        # return inspect.ismethoddescriptor(o)
        return isinstance(o, staticmethod)

    def patch(self, except_for: Iterable[str]) -> F:
        if hasattr(self.o, "__name__"):
            name = self.o.__name__
        elif hasattr(self.o, "__func__"):
            name = self.o.__func__.__name__
        else:
            raise ValueError(f"Cannot determine name for object {self.o}")
        if name in except_for:
            return self.o

        f: Callable[..., Any] = self.o.__func__  # type: ignore[attr-defined]

        @wraps(f)
        def _call(*args: Any, **kwargs: Any) -> Any:
            raise ImportError(self.msg)

        self.copy_metadata(_call)  # type: ignore[arg-type]

        return staticmethod(_call)  # type: ignore[return-value]

    def get_object_with_metadata(self) -> Any:
        return self.o.__func__  # type: ignore[attr-defined]


@PatchObject.register()
class PatchInit(PatchObject[F]):
    @classmethod
    def accept(cls, o: Any) -> bool:
        return inspect.ismethoddescriptor(o) and o.__name__ == "__init__"

    def patch(self, except_for: Iterable[str]) -> F:
        if self.o.__name__ in except_for:
            return self.o

        f: Callable[..., Any] = self.o

        @wraps(f)
        def _call(*args: Any, **kwargs: Any) -> Any:
            raise ImportError(self.msg)

        self.copy_metadata(_call)  # type: ignore[arg-type]

        return staticmethod(_call)  # type: ignore[return-value]

    def get_object_with_metadata(self) -> Any:
        return self.o


@PatchObject.register()
class PatchProperty(PatchObject[Any]):
    @classmethod
    def accept(cls, o: Any) -> bool:
        return inspect.isdatadescriptor(o) and hasattr(o, "fget")

    def patch(self, except_for: Iterable[str]) -> property:
        if not hasattr(self.o, "fget"):
            raise ValueError(f"Cannot patch property without getter: {self.o}")
        f: Callable[..., Any] = self.o.fget

        if f.__name__ in except_for:
            return self.o  # type: ignore[no-any-return]

        @wraps(f)
        def _call(*args: Any, **kwargs: Any) -> Any:
            raise ImportError(self.msg)

        self.copy_metadata(_call)

        return property(_call)

    def get_object_with_metadata(self) -> Any:
        return self.o.fget


@PatchObject.register()
class PatchClass(PatchObject[type[Any]]):
    @classmethod
    def accept(cls, o: Any) -> bool:
        return inspect.isclass(o)

    def patch(self, except_for: Iterable[str]) -> type[Any]:
        if self.o.__name__ in except_for:
            return self.o

        for name, member in inspect.getmembers(self.o):
            # Patch __init__ method if possible, but not other internal methods
            if name.startswith("__") and name != "__init__":
                continue
            patched = patch_object(
                member,
                missing_modules=self.missing_modules,
                dep_target=self.dep_target,
                fail_if_not_patchable=False,
                except_for=except_for,
            )
            with suppress(AttributeError):
                setattr(self.o, name, patched)

        return self.o


def patch_object(
    o: T,
    *,
    missing_modules: dict[str, str],
    dep_target: str,
    fail_if_not_patchable: bool = True,
    except_for: str | Iterable[str] | None = None,
) -> T:
    patcher = PatchObject.create(
        o, missing_modules=missing_modules, dep_target=dep_target
    )
    if fail_if_not_patchable and patcher is None:
        raise ValueError(f"Cannot patch object of type {type(o)}")

    except_for = except_for if except_for is not None else []
    except_for = [except_for] if isinstance(except_for, str) else except_for

    return patcher.patch(except_for=except_for) if patcher else o


def require_optional_import(
    modules: str | Iterable[str],
    dep_target: str,
    *,
    except_for: str | Iterable[str] | None = None,
) -> Callable[[T], T]:
    """Decorator to handle optional module dependencies

    Args:
        modules: Module name or list of module names required
        dep_target: Target name for pip installation
        except_for: Name or list of names of objects to exclude from patching
    """
    missing_modules = get_missing_imports(modules)

    if not missing_modules:

        def decorator(o: T) -> T:
            return o

    else:

        def decorator(o: T) -> T:
            return patch_object(
                o,
                missing_modules=missing_modules,
                dep_target=dep_target,
                except_for=except_for,
            )

    return decorator


if TYPE_CHECKING:
    pass

P = ParamSpec("P")
T = TypeVar("T")
