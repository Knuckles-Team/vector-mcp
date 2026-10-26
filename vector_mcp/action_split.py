"""Serve each action of a legacy action-routed tool as its own typed MCP tool.

A registrar written for the condensed surface (``action`` plus parameters) is run
against a scratch server. Every action it declares becomes a tool named
``<prefix>_<action>`` that keeps the registrar's typed parameters, dependency
injection, confirmation gates and annotations-by-verb, so the connector serves
one tool per operation and no ``action`` argument.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Iterable
from typing import Any

from agent_connector_sdk.config import setting
from agent_connector_sdk.mcp.tool_surface import registered_tools
from fastmcp import FastMCP

_READ_ONLY = re.compile(
    r"^(list|get|read|check|test|status|show|find|search|describe|discover|inspect|"
    r"info|query|scan|audit|analyze|monitor|detect|verify|view|fetch|tail|metrics|"
    r"progress|stat|diff|compare|calculate|preview)"
)
_DESTRUCTIVE = re.compile(
    r"(remove|delete|cancel|kill|stop|rotate|revoke|reset|purge|drop|wipe|clean|"
    r"uninstall|erase|terminate|disable|truncate)"
)


def _actions(tool: Any) -> list[str]:
    prop = (tool.parameters or {}).get("properties", {}).get("action", {})
    enum = prop.get("enum")
    if enum:
        return [str(item) for item in enum]
    seen: list[str] = []
    for name in re.findall(r"'([a-z][a-z0-9_]*)'", str(prop.get("description", ""))):
        if name not in seen:
            seen.append(name)
    return seen


def _annotations(action: str, declared: Any) -> dict[str, Any]:
    read_only = bool(_READ_ONLY.match(action)) and not _DESTRUCTIVE.search(action)
    destructive = bool(_DESTRUCTIVE.search(action))
    return {
        "title": action.replace("_", " ").title(),
        "readOnlyHint": read_only,
        "destructiveHint": destructive and not read_only,
        "idempotentHint": read_only or destructive,
        "openWorldHint": getattr(declared, "open_world_hint", None) or False,
    }


def _wrapper(tool: Any, action: str) -> Callable[..., Any]:
    fn = tool.fn
    signature = inspect.signature(fn)
    parameters = [p for p in signature.parameters.values() if p.name != "action"]

    async def call(**kwargs: Any) -> Any:
        return await fn(action=action, **kwargs)

    call.__signature__ = signature.replace(parameters=parameters)  # type: ignore[attr-defined]
    call.__annotations__ = {
        key: value
        for key, value in getattr(fn, "__annotations__", {}).items()
        if key != "action"
    }
    call.__name__ = action
    call.__doc__ = fn.__doc__
    return call


def register_split_action_tools(
    mcp: Any,
    registry: Iterable[tuple[str, str, Callable[[Any], Any]]],
    *,
    prefixes: dict[str, str] | None = None,
) -> list[str]:
    """Run each enabled ``(tag, toggle, registrar)`` and serve its tools per action.

    ``prefixes`` maps an action-routed tool name to the prefix of its per-action
    tools (default: the tool name). Tools without an ``action`` argument are
    registered unchanged. Returns the tags that registered.
    """
    scratch = FastMCP("action-split")
    tags: list[str] = []
    for tag, toggle, registrar in registry:
        if not setting(toggle, True):
            continue
        registrar(scratch)
        tags.append(tag)
    for name, tool in registered_tools(scratch).items():
        actions = (
            _actions(tool)
            if "action" in (tool.parameters or {}).get("properties", {})
            else []
        )
        if not actions:
            mcp.add_tool(tool)
            continue
        prefix = (prefixes or {}).get(name, name)
        for action in actions:
            mcp.tool(
                name=f"{prefix}_{action}",
                description=f"{action.replace('_', ' ').capitalize()} ({name}). "
                + (tool.description or ""),
                tags=set(tool.tags or set()) | {"operation", "granular"},
                annotations=_annotations(action, tool.annotations),
            )(_wrapper(tool, action))
    return tags
