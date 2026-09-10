# AML CPU-only training/inference image for the LightGBM baseline.
# GPU transformer training moved to RunPod, so no PyTorch / Lightning deps here.
#
# Build context: this directory (self-contained on purpose — package pins
# mirror requirements-shared.txt / requirements-cloud.txt at the repo root,
# so `az ml environment create` works regardless of the build context).
FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04

RUN pip install --no-cache-dir \
    "pandas" \
    "scikit-learn" \
    "numpy" \
    "click" \
    "pyyaml" \
    "tqdm" \
    "lightgbm" \
    "imbalanced-learn" \
    "mlflow>=2.9,<2.14" \
    "azure-ai-ml" \
    "azureml-mlflow"
