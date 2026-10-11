# Tool-gateway MCP server: single authenticated exec gateway.
# Secrets enter ONLY via compose env_file; the agent container holds none.
# Pinned versions mirror dev/dev.Dockerfile conventions.
FROM python:3.12.14-slim

WORKDIR /workspace

ENV HOME=/home/agent \
    PIP_CACHE_DIR=/home/agent/.cache/pip \
    PORT=8004 \
    TOOLS_YAML=/workspace/dev/tools.yaml

ARG AZURE_CLI_VERSION=2.80.0
ARG ML_EXTENSION_VERSION=2.44.1
ARG GH_VERSION=2.101.0
ARG AZCOPY_VERSION=10.32.7
ARG RUNPODCTL_VERSION=2.14.0
ARG SKYPILOT_VERSION=0.14.0

# rsync + openssh-client are required by SkyPilot's launch/exec path: its
# command runner shells out to the `rsync` and `ssh` binaries to sync workdirs
# and run remote commands.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         curl ca-certificates bash less rsync openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Azure CLI (pinned) + ML extension + mlflow.
RUN --mount=type=cache,id=pip-gw,target=/home/agent/.cache/pip,sharing=locked \
    pip install --break-system-packages \
      "azure-cli==${AZURE_CLI_VERSION}" \
      "mcp[cli]==2.2.0" \
      "pyyaml==6.0.3" \
      "mlflow==2.16.2" \
      "azureml-mlflow==1.62.0.post6"

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

# SkyPilot in its own venv, mirroring the /opt/azcli isolation in
# dev/dev.Dockerfile: its dependency surface is large (the runpod extra pulls
# the runpod SDK, fastapi, opentelemetry, boto3, ...) and must not perturb the
# pinned azure-cli/mlflow stack installed above. Only the `sky` entrypoint is
# exposed on PATH (never the venv's bin dir, so its `python` cannot shadow the
# project's). The agent reaches it through the `sky` tool in dev/tools.yaml, so
# RunPod credentials stay in this container. The import is a build-time gate
# that the dependency stack resolved cleanly.
RUN --mount=type=cache,id=pip-gw,target=/home/agent/.cache/pip,sharing=locked \
    /usr/local/bin/python -m venv /opt/sky \
    && /opt/sky/bin/pip install "skypilot[runpod]==${SKYPILOT_VERSION}" \
    && ln -s /opt/sky/bin/sky /usr/local/bin/sky \
    && /opt/sky/bin/python -c "import sky; print('skypilot', sky.__version__)"

# Bake the MLflow inspection CLI into the image. It must NOT be executed from
# the agent-writable /workspace (via `python -m twitter.mlflow_inspect`): the
# agent shares that volume and could rewrite the module to run arbitrary code
# with this tool's creds. Running the copied script instead puts the image's
# /opt/tools (not /workspace/cwd) on sys.path; -I additionally ignores
# PYTHON* env and user site-packages.
COPY twitter/mlflow_inspect.py /opt/tools/mlflow_inspect.py

RUN useradd -m -u 1000 agent \
    && mkdir -p /home/agent/.cache/pip /home/agent/.sky /home/agent/.runpod \
    && chown -R agent:agent /home/agent /workspace /opt/sky
USER agent

# Install the Azure ML extension (as the runtime user, so `az ml` is found)
RUN az extension add --name ml --version ${ML_EXTENSION_VERSION} --yes

# Use bind mounts for now so this container has access to the repo
# COPY dev/tool-gateway/server.py dev/tool-gateway/broker.py dev/tool-gateway/redact.py ./
# COPY dev/tools.yaml /app/tools.yaml

EXPOSE 8004

CMD ["python", "dev/tool-gateway/server.py"]
