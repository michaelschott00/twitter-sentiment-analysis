FROM python:3.12-slim

WORKDIR /workspace

# BuildKit cache mounts (--mount=type=cache) keep downloaded artifacts across
# rebuilds, so an invalidated layer does not force re-downloading:
#   id=apt-dev         /var/cache/apt            deb packages (.deb files)
#   id=downloads-dev   /var/cache/downloads      pinned curl archives + installers
#   id=npm-dev         /home/agent/.npm          npm/npx cache (playwright, hermes)
#   id=uv-dev          /home/agent/.cache/uv     uv download cache (hermes venv)
#   id=playwright-dev  /home/agent/.cache/ms-playwright  browser downloads
#   id=hermes-dev      /home/agent/.hermes       hermes-managed uv + node
#   id=pip-dev         /home/agent/.cache/pip    pip wheels (must match PIP_CACHE_DIR,
#                                                 set below because HOME=/home/agent)
#   id=tflint-plugins-dev  /var/cache/tflint-plugins  tflint plugin downloads
#   id=nltk-dev        /var/cache/nltk           nltk data downloads

# Install basic utils
RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates libgomp1 git unzip \
    && rm -rf /var/lib/apt/lists/*

# Install hermes
ENV HERMES_HOME=/home/agent/.hermes
ENV HOME=/home/agent
# ENV HERMES_TUI=1
RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends nodejs npm && rm -rf /var/lib/apt/lists/*
RUN --mount=type=cache,id=hermes-dev,target=/home/agent/.hermes,sharing=locked \
    --mount=type=cache,id=npm-dev,target=/home/agent/.npm,sharing=locked \
    --mount=type=cache,id=uv-dev,target=/home/agent/.cache/uv,sharing=locked \
    --mount=type=cache,id=playwright-dev,target=/home/agent/.cache/ms-playwright,sharing=locked \
    --mount=type=cache,id=downloads-dev,target=/var/cache/downloads,sharing=locked \
    { [ -f /var/cache/downloads/hermes-install.sh ] || curl -fsSL https://hermes-agent.nousresearch.com/install.sh -o /var/cache/downloads/hermes-install.sh; } \
    && bash /var/cache/downloads/hermes-install.sh

# Install opencode (cache the final binary; the installer otherwise re-downloads
# the release tarball on every invalidated rebuild)
RUN --mount=type=cache,id=downloads-dev,target=/var/cache/downloads,sharing=locked \
    { [ -x /var/cache/downloads/opencode ] \
        || { curl -fsSL https://opencode.ai/install | bash \
             && install -m 755 /home/agent/.opencode/bin/opencode /var/cache/downloads/opencode; }; } \
    && install -d /home/agent/.opencode/bin \
    && install -m 755 /var/cache/downloads/opencode /home/agent/.opencode/bin/opencode \
    && ln -sf /home/agent/.opencode/bin/opencode /usr/local/bin/opencode

# Install playwright system deps (debs and the npx package come from cache)
RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=npm-dev,target=/home/agent/.npm,sharing=locked \
    npx playwright install-deps chromium

# Install python packages
ENV PIP_CACHE_DIR=/home/agent/.cache/pip
COPY dev/requirements.txt .
RUN --mount=type=cache,id=pip-dev,target=/home/agent/.cache/pip,sharing=locked \
    pip install --break-system-packages -r requirements.txt

# Download NLTK data (downloaded into the cache, then copied into the image so
# the agent user can find it at runtime; the cache keeps rebuilds offline)
RUN --mount=type=cache,id=nltk-dev,target=/var/cache/nltk,sharing=locked \
    python -c "import nltk; nltk.download('stopwords', download_dir='/var/cache/nltk')" \
    && mkdir -p /usr/local/share/nltk_data \
    && cp -a /var/cache/nltk/. /usr/local/share/nltk_data/
ENV NLTK_DATA=/usr/local/share/nltk_data

# Install tflint
ENV TFLINT_PLUGIN_DIR=/usr/local/share/tflint/plugins
RUN --mount=type=cache,id=downloads-dev,target=/var/cache/downloads,sharing=locked \
    { [ -f /var/cache/downloads/tflint.zip ] || curl -fsSL https://github.com/terraform-linters/tflint/releases/latest/download/tflint_linux_amd64.zip -o /var/cache/downloads/tflint.zip; } \
    && unzip -o /var/cache/downloads/tflint.zip -d /var/cache/downloads \
    && install -m 755 /var/cache/downloads/tflint /usr/local/bin/tflint \
    && mkdir -p $TFLINT_PLUGIN_DIR \
    && chmod 1777 $TFLINT_PLUGIN_DIR

# Pre-cache the tflint plugins declared in the repo config (single source of
# truth: infra/terraform/.tflint.hcl) so later runs work fast/offline.
# Plugins are downloaded into the cache mount, then copied into the image.
# Staged under /tmp since the repo is not COPY'd into the image; removed after.
COPY infra/terraform/.tflint.hcl /tmp/tflint-plugins/.tflint.hcl
RUN --mount=type=cache,id=tflint-plugins-dev,target=/var/cache/tflint-plugins,sharing=locked \
    TFLINT_PLUGIN_DIR=/var/cache/tflint-plugins tflint --chdir=/tmp/tflint-plugins --init \
    && mkdir -p $TFLINT_PLUGIN_DIR \
    && cp -a /var/cache/tflint-plugins/. $TFLINT_PLUGIN_DIR/ \
    && rm -rf /tmp/tflint-plugins

# Install terraform
RUN --mount=type=cache,id=downloads-dev,target=/var/cache/downloads,sharing=locked \
    { [ -f /var/cache/downloads/terraform.zip ] || curl -fsSL https://releases.hashicorp.com/terraform/1.16.1/terraform_1.16.1_linux_amd64.zip -o /var/cache/downloads/terraform.zip; } \
    && unzip -o /var/cache/downloads/terraform.zip -d /var/cache/downloads \
    && install -m 755 /var/cache/downloads/terraform /usr/local/bin/terraform

# Install azure-cli (wheels cached in the pip cache mount)
RUN --mount=type=cache,id=pip-dev,target=/home/agent/.cache/pip,sharing=locked \
    pip install --break-system-packages azure-cli \
    && az upgrade --yes

# Install azcopy
RUN --mount=type=cache,id=downloads-dev,target=/var/cache/downloads,sharing=locked \
    { [ -f /var/cache/downloads/azcopy.tar.gz ] || curl -fsSL https://aka.ms/downloadazcopy-v10-linux -o /var/cache/downloads/azcopy.tar.gz; } \
    && tar -xzf /var/cache/downloads/azcopy.tar.gz -C /var/cache/downloads \
    && install -m 755 /var/cache/downloads/azcopy_linux_amd64_*/azcopy /usr/local/bin/azcopy

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
