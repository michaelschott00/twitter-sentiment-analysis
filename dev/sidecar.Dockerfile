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
RUN az extension add --name ml --yes

# Setup aliases
COPY <<EOF /home/agent/.bashrc
alias azlogin='az login --service-principal --username \$AZCOPY_SPA_APPLICATION_ID --password  \$AZCOPY_SPA_CLIENT_SECRET --tenant \$AZCOPY_TENANT_ID'
EOF

# Start mcp server
ENTRYPOINT ["python", "dev/mcp_server.py"]
