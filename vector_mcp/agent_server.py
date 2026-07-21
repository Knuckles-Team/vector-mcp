#!/usr/bin/python
import logging
import sys
import warnings
from contextlib import nullcontext
from importlib.resources import as_file, files
from pathlib import Path

from agent_utilities.core.config import setting

from vector_mcp import __version__

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


DEFAULT_AGENT_NAME = None
DEFAULT_AGENT_DESCRIPTION = None
DEFAULT_AGENT_SYSTEM_PROMPT = None


def agent_server():
    from agent_utilities import (
        build_system_prompt_from_workspace,
        create_agent_parser,
        create_agent_server,
        initialize_workspace,
        load_identity,
    )

    global DEFAULT_AGENT_NAME, DEFAULT_AGENT_DESCRIPTION, DEFAULT_AGENT_SYSTEM_PROMPT
    initialize_workspace()
    meta = load_identity()
    DEFAULT_AGENT_NAME = setting("DEFAULT_AGENT_NAME", meta.get("name", "Vector MCP"))
    DEFAULT_AGENT_DESCRIPTION = setting(
        "AGENT_DESCRIPTION",
        meta.get(
            "description",
            "AI agent for Vector Mcp operations.",
        ),
    )
    DEFAULT_AGENT_SYSTEM_PROMPT = setting(
        "AGENT_SYSTEM_PROMPT",
        meta.get("content") or build_system_prompt_from_workspace(),
    )

    warnings.filterwarnings("ignore", message=".*urllib3.*or chardet.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="fastmcp")

    print(f"vector-mcp agent v{__version__}", file=sys.stderr)
    parser = create_agent_parser()
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")

    packaged_config = files("vector_mcp").joinpath("bundled_mcp.json")
    config_context = (
        nullcontext(Path(args.mcp_config))
        if args.mcp_config
        else as_file(packaged_config)
    )
    with config_context as mcp_config:
        create_agent_server(
            mcp_url=args.mcp_url,
            mcp_config=str(mcp_config),
            host=args.host,
            port=args.port,
            provider=args.provider,
            model_id=args.model_id,
            router_model=args.model_id,
            agent_model=args.model_id,
            base_url=args.base_url,
            api_key=args.api_key,
            custom_skills_directory=args.custom_skills_directory,
            enable_web_ui=args.web,
            enable_terminal_ui=args.terminal,
            enable_web_logs=args.web_logs,
            workspace=args.workspace,
            name=DEFAULT_AGENT_NAME,
            system_prompt=DEFAULT_AGENT_SYSTEM_PROMPT,
            enable_otel=args.otel,
            otel_endpoint=args.otel_endpoint,
            otel_headers=args.otel_headers,
            otel_public_key=args.otel_public_key,
            otel_secret_key=args.otel_secret_key,
            otel_protocol=args.otel_protocol,
            debug=args.debug,
        )


if __name__ == "__main__":
    agent_server()
