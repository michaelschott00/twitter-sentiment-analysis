FROM python:3.12.14-slim

WORKDIR /workspace

ENV HOME=/home/agent

# Pinned tool versions (single source of truth for reproducibility).
# Bump these ARGs to upgrade; cache filenames below include the version so a
# version bump cannot reuse a stale BuildKit cache artifact.
ARG OPENCODE_VERSION=1.18.30
ARG TFLINT_VERSION=0.64.0
ARG TERRAFORM_VERSION=1.16.1
ARG AZURE_CLI_VERSION=2.90.0
ARG AZCOPY_VERSION=10.32.7
ARG ML_EXTENSION_VERSION=2.44.1

# BuildKit cache mounts (--mount=type=cache) keep downloaded artifacts across
# rebuilds, so an invalidated layer does not force re-downloading:
#   id=apt-dev         /var/cache/apt            deb packages (.deb files)
#   id=downloads-dev   /var/cache/downloads      pinned curl archives + installers
#   id=pip-dev         $HOME/.cache/pip    pip wheels (must match PIP_CACHE_DIR,
#                                                 set below because HOME=$HOME)
#   id=tflint-plugins-dev  /var/cache/tflint-plugins  tflint plugin downloads
#   id=nltk-dev        /var/cache/nltk           nltk data downloads

# Install basic utils (+ build-essential so source distributions like the
# regular `fasttext` package, which contains C++ extensions, can be compiled
# during `pip install`)
RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates libgomp1 git unzip build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install opencode (cache the final binary; the installer otherwise re-downloads
# the release tarball on every invalidated rebuild)
RUN --mount=type=cache,id=downloads-dev,target=/var/cache/downloads,sharing=locked \
    { [ -x /var/cache/downloads/opencode-${OPENCODE_VERSION} ] \
        || { curl -fsSL https://opencode.ai/install | bash -s -- --version ${OPENCODE_VERSION} --no-modify-path \
             && install -m 755 $HOME/.opencode/bin/opencode /var/cache/downloads/opencode-${OPENCODE_VERSION}; }; } \
    && install -d $HOME/.opencode/bin \
    && install -m 755 /var/cache/downloads/opencode-${OPENCODE_VERSION} $HOME/.opencode/bin/opencode \
    && ln -sf $HOME/.opencode/bin/opencode /usr/local/bin/opencode \
    && mkdir -p $HOME/.config/opencode $HOME/.local/share/opencode $HOME/.local/state/opencode \
    && $HOME/.opencode/bin/opencode completion >> $HOME/.bashrc

# Install python packages from pyproject.toml (single source of truth).
# The CPU torch index is needed so the `torch` extra resolves to CPU wheels.
ENV PIP_CACHE_DIR=$HOME/.cache/pip
COPY pyproject.toml README.md ./
COPY twitter ./twitter
RUN --mount=type=cache,id=pip-dev,target=$HOME/.cache/pip,sharing=locked \
    pip install --break-system-packages --extra-index-url https://download.pytorch.org/whl/cpu -e ".[dev,torch,lgbm,llm]"

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
    { [ -f /var/cache/downloads/tflint-${TFLINT_VERSION}.zip ] || curl -fsSL https://github.com/terraform-linters/tflint/releases/download/v${TFLINT_VERSION}/tflint_linux_amd64.zip -o /var/cache/downloads/tflint-${TFLINT_VERSION}.zip; } \
    && unzip -o /var/cache/downloads/tflint-${TFLINT_VERSION}.zip -d /var/cache/downloads \
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
    { [ -f /var/cache/downloads/terraform_${TERRAFORM_VERSION}.zip ] || curl -fsSL https://releases.hashicorp.com/terraform/${TERRAFORM_VERSION}/terraform_${TERRAFORM_VERSION}_linux_amd64.zip -o /var/cache/downloads/terraform_${TERRAFORM_VERSION}.zip; } \
    && unzip -o /var/cache/downloads/terraform_${TERRAFORM_VERSION}.zip -d /var/cache/downloads \
    && install -m 755 /var/cache/downloads/terraform /usr/local/bin/terraform

# Install azure-cli (wheels cached in the pip cache mount; pinned, no `az upgrade`)
RUN --mount=type=cache,id=pip-dev,target=$HOME/.cache/pip,sharing=locked \
    pip install --break-system-packages azure-cli==${AZURE_CLI_VERSION}

# Install azcopy
RUN --mount=type=cache,id=downloads-dev,target=/var/cache/downloads,sharing=locked \
    { [ -f /var/cache/downloads/azcopy_${AZCOPY_VERSION}.tar.gz ] || curl -fsSL https://github.com/Azure/azure-storage-azcopy/releases/download/v${AZCOPY_VERSION}/azcopy_linux_amd64_${AZCOPY_VERSION}.tar.gz -o /var/cache/downloads/azcopy_${AZCOPY_VERSION}.tar.gz; } \
    && tar -xzf /var/cache/downloads/azcopy_${AZCOPY_VERSION}.tar.gz -C /var/cache/downloads \
    && install -m 755 /var/cache/downloads/azcopy_linux_amd64_${AZCOPY_VERSION}/azcopy /usr/local/bin/azcopy

# Create unprivileged agent user
RUN useradd -m -u 1000 agent
RUN chown -R agent:agent $HOME
USER agent

# Install the Azure ML extension (as the runtime user, so `az ml` is found)
RUN az extension add --name ml --version ${ML_EXTENSION_VERSION} --yes

# Override python entrypoint
ENTRYPOINT ["/bin/bash"]
