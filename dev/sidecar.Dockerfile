FROM python:3.12-slim

WORKDIR /workspace

# BuildKit cache mounts keep downloaded artifacts across rebuilds (see
# dev/dev.Dockerfile for the rationale). Distinct cache ids so the sidecar never
# shares state with the dev image.

# Install basic utils (single apt step; debs cached)
RUN --mount=type=cache,id=apt-sidecar,target=/var/cache/apt,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends gh ssh curl ca-certificates git unzip \
    && rm -rf /var/lib/apt/lists/*

# Install terraform
RUN --mount=type=cache,id=downloads-sidecar,target=/var/cache/downloads,sharing=locked \
    { [ -f /var/cache/downloads/terraform.zip ] || curl -fsSL https://releases.hashicorp.com/terraform/1.16.1/terraform_1.16.1_linux_amd64.zip -o /var/cache/downloads/terraform.zip; } \
    && unzip -o /var/cache/downloads/terraform.zip -d /var/cache/downloads \
    && install -m 755 /var/cache/downloads/terraform /usr/local/bin/terraform

# Install azure-cli + mcp (wheels cached in the pip cache mount)
RUN --mount=type=cache,id=pip-sidecar,target=/root/.cache/pip,sharing=locked \
    pip install --break-system-packages azure-cli "mcp[cli]" \
    && az upgrade --yes

# Install azcopy
RUN --mount=type=cache,id=downloads-sidecar,target=/var/cache/downloads,sharing=locked \
    { [ -f /var/cache/downloads/azcopy.tar.gz ] || curl -fsSL https://aka.ms/downloadazcopy-v10-linux -o /var/cache/downloads/azcopy.tar.gz; } \
    && tar -xzf /var/cache/downloads/azcopy.tar.gz -C /var/cache/downloads \
    && install -m 755 /var/cache/downloads/azcopy_linux_amd64_*/azcopy /usr/local/bin/azcopy

# Add non-privileged user
RUN useradd -m -u 1000 agent
RUN mkdir -p /home/agent/.ssh

# Add github as known host
COPY dev/known_hosts /home/agent/.ssh/known_hosts

# Adjust permissions
RUN chown -R agent:agent /home/agent

# Run as non-privileged user
USER agent

# Install the Azure ML extension (as the runtime user, so `az ml` is found)
RUN az extension add --name ml --yes

# Start mcp server
ENTRYPOINT ["python", "dev/mcp_server.py"]
