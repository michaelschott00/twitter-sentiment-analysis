# Docker Hub MCP server (https://github.com/docker/hub-mcp, image
# docker.io/mcp/dockerhub) exposed over Streamable HTTP so other
# compose services (e.g. opencode running in `dev`) can reach it at
# http://docker-hub-mcp:8083/mcp.
#
# The server's `http` transport binds to loopback by default and requires its
# own bearer token (MCP_AUTH_TOKEN) with Host allow-listing, so this image
# follows the same pattern as dev/github-mcp.Dockerfile: run the upstream
# stdio server (which reads HUB_PAT_TOKEN from the container environment,
# provided via `env_file` in compose.yaml) behind supergateway in stdio ->
# stateless Streamable HTTP bridge mode. Credentials never leave this
# container: clients connect with no Authorization header.

# Stage 1: official pre-built image. Upstream only publishes the `latest`
# tag for mcp/dockerhub, so pinning to a version is not possible; use
# latest and bump the base digest when upgrading.
ARG DOCKER_HUB_MCP_VERSION=latest
FROM docker.io/mcp/dockerhub:${DOCKER_HUB_MCP_VERSION} AS upstream

# Stage 2: Node runtime + bridge.
FROM node:22-alpine

WORKDIR /app

# Pinned version (keep in sync with dev/github-mcp.Dockerfile); bump to upgrade.
ARG SUPERGATEWAY_VERSION=3.4.3

RUN npm install "supergateway@${SUPERGATEWAY_VERSION}"

# Preserve the upstream /app layout (dist/ + node_modules/) under a
# subdirectory so Node module resolution keeps working unmodified.
COPY --from=upstream /app ./hub-upstream

ENV PORT=8083
EXPOSE 8083

# The stdio child inherits this container's environment, including
# HUB_PAT_TOKEN. HUB_USERNAME is optional: when set, it is passed as
# --username for authenticated requests; otherwise the server serves public
# content only.
# NOTE: JSON-array CMD form does no shell expansion, so a literal "${PORT}"
# would be passed through and the bridge would crash on startup (leaving the
# service name unresolvable). Use shell form so $PORT expands at runtime.
CMD if [ -n "${HUB_USERNAME}" ]; then HUB_ARGS="--transport=stdio --username=${HUB_USERNAME}"; else HUB_ARGS="--transport=stdio"; fi; node ./node_modules/supergateway/dist/index.js --stdio "node hub-upstream/dist/index.js $HUB_ARGS" --outputTransport streamableHttp --port ${PORT:-8083}
