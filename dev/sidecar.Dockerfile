FROM python:3.12.14-slim

WORKDIR /workspace

# Pinned tool versions (single source of truth for reproducibility).
ARG TERRAFORM_VERSION=1.16.1
ARG AZURE_CLI_VERSION=2.90.0
ARG MCP_VERSION=2.1.1
ARG AZCOPY_VERSION=10.32.7
ARG ML_EXTENSION_VERSION=2.44.1

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
    { [ -f /var/cache/downloads/terraform_${TERRAFORM_VERSION}.zip ] || curl -fsSL https://releases.hashicorp.com/terraform/${TERRAFORM_VERSION}/terraform_${TERRAFORM_VERSION}_linux_amd64.zip -o /var/cache/downloads/terraform_${TERRAFORM_VERSION}.zip; } \
    && unzip -o /var/cache/downloads/terraform_${TERRAFORM_VERSION}.zip -d /var/cache/downloads \
    && install -m 755 /var/cache/downloads/terraform /usr/local/bin/terraform

# Install azure-cli + mcp (wheels cached in the pip cache mount; pinned, no `az upgrade`)
RUN --mount=type=cache,id=pip-sidecar,target=/root/.cache/pip,sharing=locked \
    pip install --break-system-packages azure-cli==${AZURE_CLI_VERSION} "mcp[cli]==${MCP_VERSION}"

# Install azcopy
RUN --mount=type=cache,id=downloads-sidecar,target=/var/cache/downloads,sharing=locked \
    { [ -f /var/cache/downloads/azcopy_${AZCOPY_VERSION}.tar.gz ] || curl -fsSL https://github.com/Azure/azure-storage-azcopy/releases/download/v${AZCOPY_VERSION}/azcopy_linux_amd64_${AZCOPY_VERSION}.tar.gz -o /var/cache/downloads/azcopy_${AZCOPY_VERSION}.tar.gz; } \
    && tar -xzf /var/cache/downloads/azcopy_${AZCOPY_VERSION}.tar.gz -C /var/cache/downloads \
    && install -m 755 /var/cache/downloads/azcopy_linux_amd64_${AZCOPY_VERSION}/azcopy /usr/local/bin/azcopy

# Add non-privileged user
RUN useradd -m -u 1000 agent
RUN mkdir -p /home/agent/.ssh

# Add github as known host
COPY <<EOF /home/agent/.ssh/known_hosts
github.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl
github.com ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBEmKSENjQEezOmxkZMy7opKgwFB9nkt5YRrYMjNuG5N87uRgg6CLrbo5wAdT/y6v0mKV0U2w0WZ2YB/++Tpockg=
github.com ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQCj7ndNxQowgcQnjshcLrqPEiiphnt+VTTvDP6mHBL9j1aNUkY4Ue1gvwnGLVlOhGeYrnZaMgRK6+PKCUXaDbC7qtbW8gIkhL7aGCsOr/C56SJMy/BCZfxd1nWzAOxSDPgVsmerOBYfNqltV9/hWCqBywINIR+5dIg6JTJ72pcEpEjcYgXkE2YEFXV1JHnsKgbLWNlhScqb2UmyRkQyytRLtL+38TGxkxCflmO+5Z8CSSNY7GidjMIZ7Q4zMjA2n1nGrlTDkzwDCsw+wqFPGQA179cnfGWOWRVruj16z6XyvxvjJwbz0wQZ75XK5tKSb7FNyeIEs4TT4jk+S4dhPeAUC5y+bDYirYgM4GC7uEnztnZyaVWQ7B381AK4Qdrwt51ZqExKbQpTUNn+EjqoTwvqNj4kqx5QUCI0ThS/YkOxJCXmPUWZbhjpCg56i+2aB6CmK2JGhn57K5mj0MNdBXA4/WnwH6XoPWJzK5Nyu2zB3nAZp+S5hpQs+p1vN1/wsjk=
EOF

# Adjust permissions
RUN chown -R agent:agent /home/agent

# Run as non-privileged user
USER agent

# Install the Azure ML extension (as the runtime user, so `az ml` is found)
RUN az extension add --name ml --version ${ML_EXTENSION_VERSION} --yes

# Setup aliases
COPY <<EOF /home/agent/.bashrc
alias azlogin='az login --service-principal --username \$AZCOPY_SPA_APPLICATION_ID --password  \$AZCOPY_SPA_CLIENT_SECRET --tenant \$AZCOPY_TENANT_ID'
EOF

# Start mcp server
ENTRYPOINT ["python", "dev/mcp_server.py"]
