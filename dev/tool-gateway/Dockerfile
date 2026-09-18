# Tool-gateway MCP server: single authenticated exec gateway.
# Secrets enter ONLY via compose env_file; the agent container holds none.
# Pinned versions mirror dev/dev.Dockerfile conventions.
FROM python:3.12.14-slim

WORKDIR /workspace

ENV HOME=/home/agent \
    PIP_CACHE_DIR=/home/agent/.cache/pip \
    PORT=8004 \
    TOOLS_YAML=/workspace/dev/tools.yaml

ARG AZURE_CLI_VERSION=2.90.0
ARG ML_EXTENSION_VERSION=2.44.1
ARG GH_VERSION=2.101.0
ARG AZCOPY_VERSION=10.32.7
ARG RUNPODCTL_VERSION=2.14.0

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates bash \
    && rm -rf /var/lib/apt/lists/*

# Azure CLI (pinned) + ML extension.
RUN --mount=type=cache,id=pip-gw,target=/home/agent/.cache/pip,sharing=locked \
    pip install --break-system-packages azure-cli==${AZURE_CLI_VERSION} "mcp[cli]==2.2.0" pyyaml==6.0.3

# gh CLI (pinned .deb).
RUN curl -fsSL -o /tmp/gh.deb \
      https://github.com/cli/cli/releases/download/v${GH_VERSION}/gh_${GH_VERSION}_linux_amd64.deb \
    && apt-get update && apt-get install -y --no-install-recommends /tmp/gh.deb \
    && rm -rf /tmp/gh.deb /var/lib/apt/lists/*
COPY <<EOF $HOME/.gitconfig
[credential "https://github.com"]
        helper =
        helper = !/usr/bin/gh auth git-credential
[credential "https://gist.github.com"]
        helper =
        helper = !/usr/bin/gh auth git-credential
EOF

# azcopy (pinned).
RUN curl -fsSL https://github.com/Azure/azure-storage-azcopy/releases/download/v${AZCOPY_VERSION}/azcopy_linux_amd64_${AZCOPY_VERSION}.tar.gz -o /tmp/azcopy.tgz \
    && tar -xzf /tmp/azcopy.tgz -C /tmp \
    && install -m 755 /tmp/azcopy_linux_amd64_${AZCOPY_VERSION}/azcopy /usr/local/bin/azcopy \
    && rm -rf /tmp/azcopy*

# runpodctl (pinned).
RUN curl -fsSL -o /usr/local/bin/runpodctl \
      https://github.com/runpod/runpodctl/releases/download/v${RUNPODCTL_VERSION}/runpodctl-linux-amd64 \
    && chmod 755 /usr/local/bin/runpodctl

RUN useradd -m -u 1000 agent && mkdir -p /home/agent/.cache/pip && chown -R agent:agent /home/agent /workspace
USER agent

# Install the Azure ML extension (as the runtime user, so `az ml` is found)
RUN az extension add --name ml --version ${ML_EXTENSION_VERSION} --yes

# Use bind mounts for now so this container has access to the repo
# COPY dev/tool-gateway/server.py dev/tool-gateway/broker.py dev/tool-gateway/redact.py ./
# COPY dev/tools.yaml /app/tools.yaml

EXPOSE 8004

CMD ["python", "dev/tool-gateway/server.py"]
