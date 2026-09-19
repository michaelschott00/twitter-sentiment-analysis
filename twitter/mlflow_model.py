"""MLflow model packaging + registration for transformer checkpoints.

Lightning's ``MLFlowLogger(log_model=True)`` only uploads raw ``.ckpt`` files
as run artifacts; it does not create an MLflow ``MLmodel`` directory, so those
artifacts cannot be registered in the (Azure ML) model registry. This module
wraps a Lightning checkpoint in a minimal ``mlflow.pyfunc`` MLflow model so
training runs can register deployable model versions.

The checkpoint is stored as an MLflow artifact of the model and loaded lazily
by :class:`SentimentModel`, so loading a registered version only reconstructs
the module when ``predict`` is called.
"""

import json
import os
import shutil

import mlflow
import mlflow.pyfunc
import numpy as np

# Checkpoint artifact name inside the packaged MLflow model.
CHECKPOINT_ARTIFACT = "model.ckpt"
LABELS = ["negative", "neutral", "positive"]


class SentimentModel(mlflow.pyfunc.PythonModel):
    """Pyfunc wrapper around a Lightning checkpoint from ``twitter.modules``.

    ``context.artifacts[CHECKPOINT_ARTIFACT]`` points at the bundled checkpoint.
    ``model_meta.json`` (also bundled) records the Hugging Face encoder name
    and the Lightning module class so the correct module is reconstructed
    before ``load_from_checkpoint``.
    """

    def load_context(self, context):
        import importlib

        import torch
        from transformers import AutoTokenizer

        from twitter.models import TransformerEncoder

        checkpoint_path = context.artifacts[CHECKPOINT_ARTIFACT]
        with open(context.artifacts["model_meta.json"]) as fh:
            self.meta = json.load(fh)

        self.encoder_name = self.meta["encoder_name"]
        module_path = self.meta["module_class"]
        module_name = module_path.rsplit(".", 1)[-1]
        module_cls = getattr(
            importlib.import_module(module_path.rsplit(".", 1)[0]), module_name
        )
        encoder = TransformerEncoder(name=self.encoder_name)
        self.model = module_cls.load_from_checkpoint(checkpoint_path, encoder=encoder)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.encoder_name, model_max_length=128
        )
        self._torch = torch

    def predict(self, context, model_input, params=None):
        if isinstance(model_input, dict):
            texts = model_input["text"]
        else:
            texts = model_input
        if not isinstance(texts, list):
            texts = [texts]

        inputs = self.tokenizer(
            texts, return_tensors="pt", truncation=True, padding=True
        )
        with self._torch.no_grad():
            logits = self.model(
                {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                }
            )
        preds = logits.argmax(dim=-1).tolist()
        return np.array([LABELS[int(p)] for p in preds])


def log_and_register(
    logger,
    checkpoint_path: str,
    encoder_name: str,
    registered_model_name: str,
    module_class: str,
    artifact_path: str = "registered_model",
):
    """Package ``checkpoint_path`` as an MLflow model and register it.

    Logs to the *Lightning logger's* run (``logger.run_id``) via its client,
    then registers the resulting ``runs:/<run_id>/<artifact_path>`` URI. Uses
    the logger's client so no global active run is required, matching the
    artifact-logging approach in ``twitter.modules``. ``module_class`` is the
    fully-qualified Lightning module class needed to rebuild the model.
    """
    work_dir = os.path.dirname(checkpoint_path) or "."
    meta_path = os.path.join(work_dir, "model_meta.json")
    with open(meta_path, "w") as fh:
        json.dump({"encoder_name": encoder_name, "module_class": module_class}, fh)

    model_dir = os.path.join(work_dir, "_mlflow_model")
    # ``save_model`` refuses to write into an existing directory; re-running
    # registration (retries, resumed runs) must not fail on a stale package.
    shutil.rmtree(model_dir, ignore_errors=True)
    mlflow.pyfunc.save_model(
        path=model_dir,
        python_model=SentimentModel(),
        artifacts={
            CHECKPOINT_ARTIFACT: checkpoint_path,
            "model_meta.json": meta_path,
        },
        # The Azure ML deployment provides its own environment + scoring
        # script, so MLflow requirement inference (which falls back to
        # cloudpickle-only here anyway) adds nothing useful.
        pip_requirements=[],
    )

    client = logger.experiment
    run_id = logger.run_id
    client.log_artifacts(run_id, model_dir, artifact_path)
    return mlflow.register_model(
        f"runs:/{run_id}/{artifact_path}", registered_model_name
    )
