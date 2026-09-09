"""Scoring script for the tw-sentiment managed online endpoint (non-prod demo).

Wraps AutoTokenizer + TransformerClassifier (see plan section 9.2).
The deployment mounts the registered model (a Lightning checkpoint logged as
an MLflow artifact) at AZUREML_MODEL_DIR.
"""

import glob
import json
import os

import torch
from transformers import AutoTokenizer

from twitter.models import TransformerEncoder
from twitter.modules import SingleTaskModule

LABELS = ["negative", "neutral", "positive"]


def init():
    global model, tokenizer
    model_dir = os.environ["AZUREML_MODEL_DIR"]
    encoder_name = os.environ.get(
        "ENCODER_NAME", "sentence-transformers/bert-base-nli-mean-tokens"
    )

    ckpts = sorted(glob.glob(os.path.join(model_dir, "**", "*.ckpt"), recursive=True))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found under {model_dir}")

    encoder = TransformerEncoder(name=encoder_name)
    model = SingleTaskModule.load_from_checkpoint(ckpts[0], encoder=encoder)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(encoder_name, model_max_length=512)


def run(raw_data):
    data = json.loads(raw_data)
    texts = data["text"] if isinstance(data["text"], list) else [data["text"]]
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(inputs["input_ids"], inputs["attention_mask"])
        preds = logits.argmax(dim=-1).tolist()
    results = [{"prediction": int(p), "label": LABELS[int(p)]} for p in preds]
    return results[0] if not isinstance(data["text"], list) else results
