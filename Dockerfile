FROM python:3.12-slim

WORKDIR /workspace

# Install basic utils
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates libgomp1 git && rm -rf /var/lib/apt/lists/*

# Install hermes
ENV HERMES_HOME=/home/agent/.hermes
ENV HOME=/home/agent
# ENV HERMES_TUI=1
RUN apt-get update && apt-get install -y --no-install-recommends nodejs npm && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
RUN npx playwright install-deps chromium

# Install opencode
RUN curl -fsSL https://opencode.ai/install | bash

# Install python packages
COPY requirements-shared.txt .
COPY requirements-local.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements-shared.txt \
    pip install --no-cache-dir -r requirements-local.txt

# Install agent vault
RUN curl --proto '=https' --proto-redir '=https' --tlsv1.2 -fsSL https://get.agent-vault.dev | sh

# Create unprivileged agent user
RUN useradd -m -u 1000 agent
RUN mkdir -p $HERMES_HOME \
    && chown agent:agent $HERMES_HOME \
    && mkdir -p $HOME/.config/opencode \
    && mkdir -p $HOME/.local/share/opencode \
    && mkdir -p $HOME/.local/state/opencode \
    && chown agent:agent -R $HOME
RUN apt-get update && apt-get install -y gh
USER agent

# Install opencode shell completions
RUN /home/agent/.opencode/bin/opencode completion >> /home/agent/.bashrc

# Override python entrypoint
ENTRYPOINT ["agent-vault", "run", "--", "/home/agent/.opencode/bin/opencode"]
