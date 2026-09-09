# AML training/inference image. This is a NEW file — it does NOT replace the
# local dev Dockerfile at the repo root (see plan section 7.1).
#
# Build context: this directory (self-contained on purpose — package pins
# mirror requirements-shared.txt / requirements-cloud.txt at the repo root,
# so `az ml environment create` works regardless of the build context).
FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04

RUN pip install --no-cache-dir \
    "torch" \
    "lightning" \
    "transformers" \
    "torchmetrics" \
    "pandas" \
    "scikit-learn" \
    "numpy" \
    "jsonargparse[signatures]" \
    "omegaconf" \
    "sacremoses" \
    "emoji" \
    "nltk" \
    "click" \
    "pyyaml" \
    "tqdm" \
    "lightgbm" \
    "imbalanced-learn" \
    "mlflow>=2.9,<2.14" \
    "azure-ai-ml" \
    "azureml-mlflow"

# Hugging Face cache env (writable scratch inside AML jobs)
ENV HF_HOME=/tmp/hf_cache
ENV TRANSFORMERS_OFFLINE=0
