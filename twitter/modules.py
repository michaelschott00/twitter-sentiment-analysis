from typing import Literal

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import ConfusionMatrixDisplay
from torch import nn

from twitter.models import (
    Projector,
    SupervisedContrastiveClassifier,
    SupervisedContrastiveEncoder,
    TransformerClassifier,
    TransformerEncoder,
    TransformerRegressor,
)


class _BaseModule(pl.LightningModule):
    """Base module implementing common functionality for all models in this project.

    This includes
    - logging an input batch
    - tracking high confidence errors
    - logging metrics
    - freezing and unfreezing layers over the course of training
    """

    def __init__(
        self,
        encoder: TransformerEncoder,
        task: Literal["reg", "clf"],
        checkpoint: str = None,
        lr: float | dict[str, float] = 3e-4,
        weight_decay: float = 0.01,
        freeze_cfg: dict[str, list[int]] = None,
        unfreeze_cfg: dict[str, list[int]] = None,
    ):
        super().__init__()

        if checkpoint:
            state_dict = torch.load(
                checkpoint,
                map_location="cpu" if not torch.cuda.is_available() else "cuda",
            )
            state_dict_fixed = dict()
            for key, val in state_dict["state_dict"].items():
                state_dict_fixed[key.replace("encoder.encoder.", "")] = val
            encoder.encoder.load_state_dict(state_dict_fixed, strict=False)
        self.encoder = encoder

        self.high_confidence_errors = []

    def update_high_confidence_errors(self, input_ids, logits, labels):
        probs = torch.softmax(logits, dim=1)
        high_confidence = probs.max(dim=1).values > 0.9
        high_confidence_errors = (probs.argmax(dim=1) != labels) & high_confidence
        reviews = input_ids[high_confidence_errors].cpu()
        labels = labels[high_confidence_errors].cpu()
        preds = probs.argmax(dim=1)[high_confidence_errors].cpu()
        self.high_confidence_errors.extend(
            [
                (review, label, pred)
                for review, label, pred in zip(reviews, labels, preds)
            ]
        )

    def on_train_batch_start(self, batch, batch_idx):
        # Log some texts from the first input batch
        if self.current_epoch == 0 and batch_idx == 0:
            self.log_batch(batch, 5, "training")

    def on_train_epoch_end(self):
        freeze_cfg = getattr(self.hparams, "freeze_cfg", None)
        unfreeze_cfg = getattr(self.hparams, "unfreeze_cfg", None)
        if freeze_cfg is not None and self.current_epoch in freeze_cfg:
            self.encoder.freeze(layers=freeze_cfg[self.current_epoch])

        if unfreeze_cfg is not None and self.current_epoch in unfreeze_cfg:
            self.encoder.freeze(layers=unfreeze_cfg[self.current_epoch], unfreeze=True)

    def on_validation_batch_start(self, batch, batch_idx):
        # Log some texts from the first input batch
        if self.current_epoch == 0 and batch_idx == 0:
            self.log_batch(batch, 5, "validation")

    def log_batch(self, batch, n, stage: str):
        if (
            self.logger is None
            or not hasattr(self.logger, "experiment")
            or self.logger.experiment is None
        ):
            return
        tokenizer = getattr(self, "encoder", None)
        if tokenizer is not None:
            tokenizer = tokenizer.tokenizer
        elif hasattr(self, "model") and hasattr(self.model, "encoder"):
            tokenizer = self.model.encoder.tokenizer
        elif hasattr(self, "model") and hasattr(self.model, "tokenizer"):
            tokenizer = self.model.tokenizer
        else:
            return
        # batch is dict with input_ids
        if isinstance(batch, dict) and "input_ids" in batch:
            ids = batch["input_ids"]
        elif isinstance(batch, (list, tuple)) and len(batch) > 0:
            ids = batch[0].get("input_ids", []) if isinstance(batch[0], dict) else []
        else:
            return
        for i in range(min(len(ids), n)):
            try:
                text = tokenizer.decode(ids[i])
            except Exception:
                text = str(ids[i])
            self.logger.experiment.add_text(f"Input/{stage}", text, i)

    def log_high_confidence_errors(self):
        if (
            self.logger is None
            or not hasattr(self.logger, "experiment")
            or self.logger.experiment is None
        ):
            return
        for i, (review, label, pred) in enumerate(self.high_confidence_errors):
            text = self.encoder.tokenizer.decode(review)
            text += f"\n\nLabel: {label}\nPrediction: {pred}"
            self.logger.experiment.add_text(
                "High Confidence Errors/validation", text, i
            )

    def log_confusion_matrix(self, confmat):
        if (
            self.logger is None
            or not hasattr(self.logger, "experiment")
            or self.logger.experiment is None
        ):
            return
        fig = plt.figure()
        disp = ConfusionMatrixDisplay(confmat.compute().cpu().numpy())
        disp.plot(ax=fig.gca())
        self.logger.experiment.add_figure("Confusion Matrix/validation", fig)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        pred = self.forward(batch)
        if self.hparams.task == "clf":
            pred = pred.argmax(dim=1)
        return pred


class SingleTaskModule(_BaseModule):
    """Module for managing the training and validation for single task models (regression or classification)."""

    def __init__(
        self,
        encoder: TransformerEncoder,
        task: Literal["reg", "clf"],
        checkpoint: str = None,
        lr: float | dict[str, float] = 3e-4,
        weight_decay: float = 0.01,
        freeze_cfg: dict[str, list[int]] = None,
        unfreeze_cfg: dict[str, list[int]] = None,
        class_weights: list[float] = None,
    ):
        super().__init__(
            encoder=encoder,
            checkpoint=checkpoint,
            lr=lr,
            task=task,
            weight_decay=weight_decay,
            unfreeze_cfg=unfreeze_cfg,
            freeze_cfg=freeze_cfg,
        )

        self.save_hyperparameters(ignore="encoder")

        if self.hparams.task == "reg":
            self.model = TransformerRegressor(encoder)
        elif self.hparams.task == "clf":
            self.model = TransformerClassifier(encoder, class_weights=class_weights)

    def forward(self, batch):
        return self.model(batch["input_ids"], batch["attention_mask"])

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch)
        loss = self.model.loss_func(logits, batch["labels"])

        self.log("loss/train", loss, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch)
        loss = self.model.loss_func(logits, batch["labels"])

        self.log("loss/validation", loss, on_step=False, on_epoch=True)

        self.model.metrics.update(logits, batch["labels"])
        if self.hparams.task == "clf":
            self.model.val_confmat.update(logits, batch["labels"])
            self.update_high_confidence_errors(
                batch["input_ids"], logits, batch["labels"]
            )

        return {"loss": loss, "logits": logits, "labels": batch["labels"]}

    def on_validation_epoch_end(self):
        self.log_dict(self.model.metrics.compute(), on_step=False, on_epoch=True)
        self.model.metrics.reset()
        if self.hparams.task == "clf":
            self.log_confusion_matrix(self.model.val_confmat)
            self.model.val_confmat.reset()

    def configure_optimizers(self):
        if isinstance(self.hparams.lr, dict):
            parameters = [
                {
                    "params": self.model.encoder.parameters(),
                    "lr": self.hparams.lr["encoder"],
                },
                {"params": self.model.head.parameters(), "lr": self.hparams.lr["head"]},
            ]
        else:
            parameters = [{"params": self.parameters(), "lr": self.hparams.lr}]
        return torch.optim.AdamW(parameters, weight_decay=self.hparams.weight_decay)


class MultiTaskModule(_BaseModule):
    """Module managing training and validation for multi task models, where the loss is a weighted sum of the regression- and classification loss."""

    def __init__(
        self,
        encoder: TransformerEncoder,
        task: Literal["reg", "clf"],
        checkpoint: str = None,
        lr: float | dict[str, float] = 3e-4,
        weight_decay: float = 0.01,
        freeze_cfg: dict[str, list[int]] = None,
        unfreeze_cfg: dict[str, list[int]] = None,
        freeze: bool = False,
        loss_weight: float = 1.0,
        class_weights: list[float] = None,
    ):
        super().__init__(
            encoder=encoder,
            checkpoint=checkpoint,
            task=task,
            lr=lr,
            weight_decay=weight_decay,
            freeze_cfg=freeze_cfg,
            unfreeze_cfg=unfreeze_cfg,
        )

        self.save_hyperparameters(ignore="encoder")
        self.reg = TransformerRegressor(encoder)
        self.clf = TransformerClassifier(encoder, class_weights=class_weights)

        # store a reference for use in the super class
        if task == "reg":
            self.model = self.reg
        if task == "clf":
            self.model = self.clf

    def forward(self, batch):
        return self.model(batch["input_ids"], batch["attention_mask"])

    def loss_func(self, reg_logits, clf_logits, reg_labels, clf_labels):
        reg_loss = self.reg.loss_func(reg_logits, reg_labels)
        clf_loss = self.clf.loss_func(clf_logits, clf_labels)
        loss = (
            self.hparams.loss_weight * reg_loss
            + (1 - self.hparams.loss_weight) * clf_loss
        )
        return reg_loss, clf_loss, loss

    def training_step(self, batch, batch_idx):
        reg_labels, clf_labels = torch.split(batch["labels"], 1, dim=1)

        clf_labels = clf_labels.squeeze(
            1
        ).long()  # CrossEntropyLoss expects long labels but combining the labels into one tensor converts both to float
        reg_labels = reg_labels.squeeze(1)

        reg_logits = self.reg(batch["input_ids"], batch["attention_mask"])
        clf_logits = self.clf(batch["input_ids"], batch["attention_mask"])

        reg_loss, clf_loss, loss = self.loss_func(
            reg_logits, clf_logits, reg_labels, clf_labels
        )

        self.log("loss/regression/train", reg_loss, on_step=True, on_epoch=False)
        self.log("loss/classification/train", clf_loss, on_step=True, on_epoch=False)
        self.log("loss/train", loss, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        reg_labels, clf_labels = torch.split(batch["labels"], 1, dim=1)

        clf_labels = clf_labels.squeeze(1).long()
        reg_labels = reg_labels.squeeze(1)

        reg_logits = self.reg(batch["input_ids"], batch["attention_mask"])
        clf_logits = self.clf(batch["input_ids"], batch["attention_mask"])

        reg_loss, clf_loss, loss = self.loss_func(
            reg_logits, clf_logits, reg_labels, clf_labels
        )

        self.log("loss/regression/validation", reg_loss, on_step=False, on_epoch=True)
        self.log(
            "loss/classification/validation", clf_loss, on_step=False, on_epoch=True
        )
        self.log("loss/validation", loss, on_step=False, on_epoch=True)

        self.reg.metrics.update(reg_logits, reg_labels)
        self.clf.metrics.update(clf_logits.squeeze(), clf_labels)
        self.clf.val_confmat.update(clf_logits.squeeze(), clf_labels)

        if self.hparams.task == "clf":
            self.update_high_confidence_errors(
                batch["input_ids"], clf_logits, clf_labels
            )

        return loss

    def on_validation_epoch_end(self):
        self.log_dict(self.reg.metrics.compute())
        self.log_dict(self.clf.metrics.compute())
        self.reg.metrics.reset()
        self.clf.metrics.reset()
        if self.hparams.task == "clf":
            self.log_confusion_matrix(self.clf.val_confmat)
            self.clf.val_confmat.reset()

    def configure_optimizers(self):
        if isinstance(self.hparams.lr, dict):
            parameters = [
                {"params": self.encoder.parameters(), "lr": self.hparams.lr["encoder"]},
                {"params": self.reg.head.parameters(), "lr": self.hparams.lr["head"]},
                {"params": self.clf.head.parameters(), "lr": self.hparams.lr["head"]},
            ]
        else:
            parameters = [{"params": self.parameters(), "lr": self.hparams.lr}]
        return torch.optim.AdamW(parameters, weight_decay=self.hparams.weight_decay)


class SimCSEModule(pl.LightningModule):
    """This module implements the unsupervised contrastive learning method SimCSE (https://arxiv.org/pdf/2104.08821.pdf)."""

    def __init__(
        self, encoder: TransformerEncoder, lr: float = 3e-5, weight_decay: float = 0.01
    ):
        super().__init__()

        self.save_hyperparameters(ignore="encoder")
        self.sim = nn.CosineSimilarity(dim=2, eps=1e-6)
        self.softmax = nn.Softmax(dim=0)  # softmax goes over batch dimension
        self.encoder = encoder

    def loss_func(self, x1, x2):
        n, d = x1.shape
        x1 = x1.expand((n, n, d))
        x2 = x2.expand((n, n, d)).permute(1, 0, 2)
        sim = self.sim(x1, x2)
        return torch.diagonal(-torch.log(self.softmax(sim))).mean()

    def training_step(self, batch, batch_idx):
        x1 = self.encoder(batch["input_ids"], batch["attention_mask"])
        x2 = self.encoder(batch["input_ids"], batch["attention_mask"])
        loss = self.loss_func(x1, x2)

        self.log("loss/train", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x1 = self.encoder(batch["input_ids"], batch["attention_mask"])
        x2 = self.encoder(batch["input_ids"], batch["attention_mask"])
        loss = self.loss_func(x1, x2)

        self.log("loss/validation", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


class SupervisedConstrastivePretrainingModule(pl.LightningModule):
    """Tries using the the supervised contrastive learning method: https://arxiv.org/pdf/2004.11362.pdf in an unsupervised way for pre-training."""

    def __init__(
        self,
        encoder: TransformerEncoder,
        projector: Projector,
        lr: float = 3e-5,
        weight_decay: float = 0.01,
    ):
        super().__init__()

        self.save_hyperparameters(ignore="encoder")
        self.encoder = SupervisedContrastiveEncoder(
            encoder=encoder, projector=projector
        )

    def training_step(self, batch, batch_idx):
        z = self.encoder(batch["input_ids"], batch["attention_mask"])
        loss = self.encoder.loss_func(z, batch["labels"])
        self.log("loss/train", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        z = self.encoder(batch["input_ids"], batch["attention_mask"])
        loss = self.encoder.loss_func(z, batch["labels"])
        self.log("loss/validation", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


class SupervisedConstrastiveLearningModule(_BaseModule):
    """This module implements the supervised contrastive learning method: https://arxiv.org/pdf/2004.11362.pdf"""

    def __init__(
        self,
        encoder: TransformerEncoder,
        lr: float | dict[str, float] = 3e-4,
        weight_decay: float = 0.01,
        loss_weight: float = 0.9,
        temperature: float = 0.3,
        freeze_cfg: dict[str, list[int]] = None,
        unfreeze_cfg: dict[str, list[int]] = None,
        class_weights: list[float] = None,
    ):
        super().__init__(
            encoder=encoder,
            freeze_cfg=freeze_cfg,
            unfreeze_cfg=unfreeze_cfg,
            task="clf",
            lr=lr,
            weight_decay=weight_decay,
        )

        self.save_hyperparameters(ignore="encoder")
        self.model = SupervisedContrastiveClassifier(
            encoder=encoder, temperature=temperature, class_weights=class_weights
        )

    def loss_func(self, embeddings, logits, labels):
        loss_scl = self.hparams.loss_weight * self.model.loss_scl(
            embeddings, F.one_hot(labels, num_classes=3)
        )
        loss_ce = (1 - self.hparams.loss_weight) * self.model.loss_ce(logits, labels)
        loss = loss_ce + loss_scl
        return loss

    def training_step(self, batch, batch_idx):
        embeddings, logits = self.model(batch["input_ids"], batch["attention_mask"])
        loss = self.loss_func(embeddings, logits, batch["labels"])
        self.log("loss/train", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        embeddings, logits = self.model(batch["input_ids"], batch["attention_mask"])
        loss = self.loss_func(embeddings, logits, batch["labels"])

        self.log("loss/validation", loss, on_step=False, on_epoch=True)

        self.model.metrics.update(logits, batch["labels"])
        self.model.val_confmat.update(logits, batch["labels"])
        self.update_high_confidence_errors(batch["input_ids"], logits, batch["labels"])

        return loss

    def on_validation_epoch_end(self):
        self.log_dict(self.model.metrics.compute(), on_step=False, on_epoch=True)
        self.model.metrics.reset()
        self.log_confusion_matrix(self.model.val_confmat)
        self.log_high_confidence_errors()
        self.model.val_confmat.reset()

    def configure_optimizers(self):
        if isinstance(self.hparams.lr, dict):
            parameters = [
                {
                    "params": self.model.encoder.parameters(),
                    "lr": self.hparams.lr["encoder"],
                },
                {"params": self.model.head.parameters(), "lr": self.hparams.lr["head"]},
            ]
        else:
            parameters = [{"params": self.parameters(), "lr": self.hparams.lr}]
        return torch.optim.AdamW(parameters, weight_decay=self.hparams.weight_decay)
