# One slim multi-stage image (build in a builder, ship only /usr/local — no
# dev-tools/source): the MCP server. The A2A/LLM-agent entrypoint was retired
# (EH-484 -- agent_connector_sdk has no agent-runtime equivalent yet).
#
#   docker build -t vector-mcp:local .
FROM python:3.14-slim@sha256:cea0e6040540fb2b965b6e7fb5ffa00871e632eef63719f0ea54bca189ce14a6 AS builder-base
COPY --from=ghcr.io/astral-sh/uv:0.11.7@sha256:240fb85ab0f263ef12f492d8476aa3a2e4e1e333f7d67fbdd923d00a506a516a /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_SYSTEM_PYTHON=1 \
    UV_HTTP_TIMEOUT=3600
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# MCP dependency closure, including the mandatory full engine.
FROM builder-base AS builder-mcp
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --upgrade --break-system-packages --prerelease=allow vector-mcp[mcp]>=3.1.0

FROM python:3.14-slim@sha256:cea0e6040540fb2b965b6e7fb5ffa00871e632eef63719f0ea54bca189ce14a6 AS runtime-base
ARG HOST=0.0.0.0
ARG PORT=8000
ARG TRANSPORT="stdio"
ARG AUTH_TYPE="none"
ENV HOST=${HOST} \
    PORT=${PORT} \
    TRANSPORT=${TRANSPORT} \
    AUTH_TYPE=${AUTH_TYPE} \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:/usr/local/bin:${PATH}" \
    UV_HTTP_TIMEOUT=3600 \
    UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# MCP image: server plus the mandatory self-contained engine.
FROM runtime-base AS mcp
COPY --from=builder-mcp /usr/local /usr/local
CMD ["vector-mcp"]
