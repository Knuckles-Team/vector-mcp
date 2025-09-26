FROM python:3-slim

ARG HOST=0.0.0.0
ARG PORT=8009
ARG TRANSPORT="http"
ENV HOST=${HOST}
ENV PORT=${PORT}
ENV TRANSPORT=${TRANSPORT}
ENV PATH="/usr/local/bin:${PATH}"

RUN pip install uv \
    && uv pip install --system vector-mcp[all]

ENTRYPOINT exec vector-mcp --transport "${TRANSPORT}" --host "${HOST}" --port "${PORT}"
