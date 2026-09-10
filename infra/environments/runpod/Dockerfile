# GPU training image. torch comes from the base image; LightGBM-only
# deps (lightgbm, imbalanced-learn) are not needed here.
#
# Build context: repo root (so ./twitter and ./configs can be baked in).
FROM runpod/pytorch:1.1.0-rc.154-cu1290-torch290-ubuntu2404

WORKDIR /project

ENV DEBIAN_FRONTEND=noninteractive

RUN pip install --no-cache-dir \
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
    "mlflow>=2.9,<2.14" \
    "azure-ai-ml" \
    "azureml-mlflow"

COPY ./twitter ./twitter
COPY ./configs ./configs

ENTRYPOINT ["/bin/bash"]
