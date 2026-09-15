# GitHub MCP server (https://github.com/github/github-mcp-server) exposed over
# Streamable HTTP so other compose services (e.g. opencode running in `dev`)
# can reach it at http://github-mcp:8082/mcp.
#
# The server's `http` command requires every client to send its own
# Authorization token, so a server-side PAT is only honored in `stdio` mode.
# This image therefore runs `github-mcp-server stdio` (which reads
# GITHUB_PERSONAL_ACCESS_TOKEN from the container environment, provided via
# `env_file` in compose.yaml) behind supergateway in stdio -> stateless
# Streamable HTTP bridge mode. The PAT never leaves this container: clients
# connect with no Authorization header.

# Stage 1: official pre-built binary (same image compose used before).
FROM ghcr.io/github/github-mcp-server:latest AS upstream

# Stage 2: Node runtime + bridge.
FROM node:22-alpine

WORKDIR /app

# Pinned version (single source of truth for reproducibility); bump to upgrade.
ARG SUPERGATEWAY_VERSION=3.4.3

RUN npm install "supergateway@${SUPERGATEWAY_VERSION}"

COPY --from=upstream /server/github-mcp-server /usr/local/bin/github-mcp-server

ENV PORT=8082
EXPOSE 8082

# Stateless Streamable HTTP endpoint defaults to /mcp. The stdio child
# inherits this container's environment, including GITHUB_PERSONAL_ACCESS_TOKEN.
# NOTE: JSON-array CMD form does no shell expansion, so a literal "${PORT}"
# would be passed through and the bridge would crash on startup (leaving the
# service name unresolvable). Use shell form so $PORT expands at runtime.
CMD node ./node_modules/supergateway/dist/index.js --stdio "github-mcp-server stdio" --outputTransport streamableHttp --port ${PORT:-8082}
