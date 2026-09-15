# Azure MCP server (https://github.com/microsoft/mcp, image
# mcr.microsoft.com/azure-sdk/azure-mcp) exposed over Streamable HTTP so other
# compose services (e.g. opencode running in `dev`) can reach it at
# http://azure-mcp:8002/mcp.
#
# The image's native `--transport http` exits 1 with no logs in 3.0.0-beta.43
# (while `--transport stdio` stays up), so this image runs
# `server-binary server start --transport stdio` (which picks up Azure
# credentials from the container environment, provided via `env_file` in
# compose.yaml) behind supergateway in stdio -> stateless Streamable HTTP
# bridge mode. Same pattern as dev/github-mcp.Dockerfile.
#
# Base is Alpine (dotnet runtime-deps), so node comes from apk.

FROM mcr.microsoft.com/azure-sdk/azure-mcp:latest

USER root

RUN apk add --no-cache nodejs npm bash curl libc6-compat

WORKDIR /app

# Pinned versions (keep SUPERGATEWAY_VERSION in sync with
# dev/github-mcp.Dockerfile); bump to upgrade.
ARG SUPERGATEWAY_VERSION=3.4.3
ARG AZD_VERSION=1.34.0

RUN npm install "supergateway@${SUPERGATEWAY_VERSION}" \
    && chown -R mcp:mcp /app

# Azure Developer CLI, required by the server's `azd` tool (it execs `azd`
# from PATH, otherwise every call fails with "'azd' is not installed or not
# found in the system PATH"). Defaults install the binary under
# /opt/microsoft/azd with a symlink at /usr/local/bin/azd, so it is visible
# to the `mcp` runtime user regardless of HOME. The script needs bash/curl
# (added above); the symlink folder must exist.
RUN mkdir -p /usr/local/bin \
    && curl -fsSL https://aka.ms/install-azd.sh | bash -s -- --version "${AZD_VERSION}"

ENV PORT=8002
EXPOSE 8002

USER mcp

# server-binary lives in /mcp-server (the upstream WORKDIR) with its tool
# assemblies alongside, so run from there with an absolute path to the bridge.
# NOTE: the upstream image sets ENTRYPOINT ["./server-binary", "server",
# "start"], which would prefix (and break) our bridge command, so it must be
# cleared here. Shell-form CMD so $PORT expands at runtime (JSON-array form
# would pass a literal "${PORT}" through and the bridge would crash on startup).
WORKDIR /mcp-server
ENTRYPOINT []
CMD node /app/node_modules/supergateway/dist/index.js --stdio "./server-binary server start --transport stdio --outgoing-auth-strategy UseHostingEnvironmentIdentity" --outputTransport streamableHttp --port ${PORT:-8002}
