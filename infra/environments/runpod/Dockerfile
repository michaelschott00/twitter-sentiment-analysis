# GPU training image. Built on RunPod's lean CUDA base, which ships no torch,
# so the whole torch stack is installed from pyproject.toml (the `torch` extra)
# and this project controls every version. A `runpod/pytorch` base instead
# ships its own CUDA build that pip won't replace — a `==x.y.z` pin matches
# `x.y.z+cuNNN` (PEP 440 ignores the local label) — so PyPI's cu128 wheels end
# up layered on a cu129 torch stack and the CUDA ABIs clash at import.
#
# This image ships *dependencies only*; the sources (twitter/, configs/) are
# cloned onto the pod at start by run.sh. That keeps the expensive dependency
# layer keyed on pyproject.toml alone, so a source edit never rebuilds it.
#
# Build context: repo root (run.sh/fetch_splits.py live under
# infra/environments/runpod/). uv is preinstalled in the base image.
FROM runpod/base:1.3.1-cuda1281-ubuntu2404

# Azure ML's MLflow plugin flushes queued artifact uploads with a hard 300s
# timeout. The `registered_model` bundle wraps a multi-GB checkpoint, and on a
# slow pod link (or when several runs upload at once) the flush can exceed that
# limit ("Failed to flush task queue within 300.0 seconds"), failing model
# registration even though the run and metrics are already logged. Raise it for
# every run; twitter.mlflow_model also retries registration.
ENV AZUREML_ARTIFACTS_DEFAULT_TIMEOUT=900

# Dependencies live in a uv-managed venv, kept OFF the default PATH so the
# base image's Jupyter/SSH (/start.sh) keep using the system interpreter;
# run.sh calls /opt/venv/bin/python explicitly.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV UV_PYTHON_DOWNLOADS=never
ENV UV_LINK_MODE=copy

# Install runtime dependencies straight from pyproject.toml. --no-install-project
# installs the project's dependencies (core + `torch` extra) but NOT the package
# itself, so this layer needs no source files and is only invalidated when
# pyproject.toml changes. The uv cache mount reuses downloaded wheels across
# such rebuilds.
COPY ./pyproject.toml /tmp/build/
RUN --mount=type=cache,target=/workspace/.cache/uv \
    uv sync --no-install-project --extra torch --directory /tmp/build \
    && rm -rf /tmp/build

# Build-time gate: fail the image build (not a GPU pod at $0.50+/hr) if the
# dependency stack is broken — torch/CUDA ABI drift and omegaconf's antlr pin
# both surface here. The cloned `twitter.*` import chain is checked at runtime
# by run.sh instead, since the sources are not in the image.
RUN /opt/venv/bin/python -c "import torch, lightning, transformers, torchmetrics, jsonargparse, omegaconf, mlflow, azure.ai.ml; print('training deps ok')"

# Pod bootstrap: clones the repo and runs the requested LightningCLI command
# (defaults to `python -m twitter.main --help`), downloading the twitter-splits
# data asset first. See the script header for the full contract.
COPY ./infra/environments/runpod/run.sh /usr/local/bin/run.sh
COPY ./infra/environments/runpod/fetch_splits.py /usr/local/bin/fetch_splits.py
RUN chmod +x /usr/local/bin/run.sh

WORKDIR /project

# No CMD: with no docker args, run.sh runs `python -m twitter.main --help`
# (AUTO_STOP=0) instead of a CMD, which the entrypoint would have prepended a
# second `python -m twitter.main` to.
ENTRYPOINT ["/usr/local/bin/run.sh"]
