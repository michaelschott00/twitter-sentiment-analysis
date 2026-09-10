FROM python:3.12-slim

WORKDIR /workspace

# Install basic utils
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates libgomp1 git unzip \
    && rm -rf /var/lib/apt/lists/*

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
COPY dev/requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('stopwords')"

# Install tflint
ENV TFLINT_PLUGIN_DIR=/usr/local/share/tflint/plugins
RUN curl -sSLO https://github.com/terraform-linters/tflint/releases/latest/download/tflint_linux_amd64.zip \
  && unzip tflint_linux_amd64.zip \
  && install -c -v tflint /usr/local/bin/ \
  && mkdir -p $TFLINT_PLUGIN_DIR \
  && chmod 1777 $TFLINT_PLUGIN_DIR

# Pre-cache the tflint plugins declared in the repo config (single source of
# truth: infra/terraform/.tflint.hcl) so later runs work fast/offline.
# Staged under /tmp since the repo is not COPY'd into the image; removed after.
COPY infra/terraform/.tflint.hcl /tmp/tflint-plugins/.tflint.hcl
RUN tflint --chdir=/tmp/tflint-plugins --init \
  && rm -rf /tmp/tflint-plugins

# Install terraform
RUN curl -sSLO https://releases.hashicorp.com/terraform/1.16.1/terraform_1.16.1_linux_amd64.zip \
    && unzip terraform_1.16.1_linux_amd64.zip \
    && mv terraform /usr/local/bin

# Install azure-cli
RUN pip install --no-cache-dir azure-cli \
    && az upgrade --yes

# Install azcopy
RUN curl -sSLO https://aka.ms/downloadazcopy-v10-linux \
    && tar -xvf downloadazcopy-v10-linux \
    && mv azcopy_linux_amd64_*/azcopy /usr/local/bin

# Create unprivileged agent user
RUN useradd -m -u 1000 agent
RUN mkdir -p $HERMES_HOME \
    && chown agent:agent $HERMES_HOME \
    && mkdir -p $HOME/.config/opencode \
    && mkdir -p $HOME/.local/share/opencode \
    && mkdir -p $HOME/.local/state/opencode \
    && chown agent:agent -R $HOME
USER agent

# Install opencode shell completions
RUN /home/agent/.opencode/bin/opencode completion >> /home/agent/.bashrc

# Install the Azure ML extension (as the runtime user, so `az ml` is found)
RUN az extension add --name ml --yes

# Override python entrypoint
ENTRYPOINT ["/bin/bash"]
