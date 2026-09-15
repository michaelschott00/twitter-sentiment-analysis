# AML CPU-only training/inference image for the LightGBM baseline.
# GPU transformer training moved to RunPod, so no PyTorch / Lightning deps here.
#
# Deps come from pyproject.toml (single source of truth) — see the `lgbm` extra.
#
# Build context: repo root (see twitter-ml-env.yaml), so the packaging files
# and the ./twitter sources can be copied in.
FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04

WORKDIR /project

COPY ./pyproject.toml ./README.md ./
COPY ./twitter ./twitter
RUN pip install --no-cache-dir ".[lgbm]"
