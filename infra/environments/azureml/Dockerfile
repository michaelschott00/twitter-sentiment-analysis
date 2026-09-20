# AML CPU-only training/inference image for the LightGBM baseline.
# GPU transformer training moved to RunPod, so no PyTorch / Lightning deps here.
#
# Deps come from pyproject.toml (single source of truth) — see the `lgbm` extra.
#
# Build context: repo root (see twitter-ml-env.yaml), so the packaging files
# and the ./twitter sources can be copied in.
FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04

WORKDIR /project

# See infra/environments/runpod/Dockerfile: Azure ML's MLflow plugin otherwise
# aborts a slow artifact upload after 300s, which fails model registration for
# CPU jobs (e.g. the LightGBM baseline) too.
ENV AZUREML_ARTIFACTS_DEFAULT_TIMEOUT=900

COPY ./pyproject.toml ./README.md ./
COPY ./twitter ./twitter
RUN pip install --no-cache-dir ".[lgbm]"
