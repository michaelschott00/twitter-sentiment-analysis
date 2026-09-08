from typing import Literal

import torch
import torch.nn.functional as F
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from torch import nn
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
from torchmetrics.regression import MeanSquaredError
from transformers import AutoConfig, AutoModel, AutoTokenizer

from twitter import util

################################
#                              #
#         Transformers         #
#                              #
################################


class TransformerEncoder(nn.Module):
    """Loads a huggingface transformer model and produces tweet embeddings.

    This module allows adapting the embedding extraction to the architecture.
    It also allows:
    - freezing some of the layers during training
    - re-initializing some of the layers before training
    - starting with a frozen encoder and unfreezing layers throughout training
    """

    def __init__(
        self,
        name: str,
        pooling: Literal["cls", "mean", "last"] = "cls",
        freeze: bool = False,
        reset_last: int = None,
    ):
        super().__init__()

        self.config = AutoConfig.from_pretrained(name)
        if ("roberta" in name) or ("sentence-transformers" in name):
            self.encoder = AutoModel.from_pretrained(
                name, add_pooling_layer=False, config=self.config
            )  # use cls token as embedding so don't pool!!
        else:
            self.encoder = AutoModel.from_pretrained(name, config=self.config)

        self.tokenizer = AutoTokenizer.from_pretrained(name, model_max_length=512)

        # match the pooling method to the model
        if pooling == "cls":
            self.pooling = lambda x, attention_mask: x[
                :, 0, :
            ]  # attention_mask for compatibility
        elif pooling == "last":
            self.pooling = lambda x, attention_mask: x[:, -1, :]
        elif pooling == "mean":
            self.pooling = util.MeanPooling()
        else:
            raise ValueError(f"Pooling method {pooling} not supported.")

        if reset_last:
            for layer in self.encoder.encoder.layer[-reset_last:]:
                layer.apply(self.encoder._init_weights)

        if freeze:
            self.freeze(list(range(len(self.encoder.encoder.layer))))

    def freeze(self, layers: list[int], unfreeze: bool = False):
        """Freezes the given layers."""
        for i in range(len(self.encoder.encoder.layer)):
            if i in layers:
                for param in self.encoder.encoder.layer[i].parameters():
                    param.requires_grad = unfreeze

    def forward(self, input_ids, attention_mask):
        token_embeddings = self.encoder(input_ids, attention_mask=attention_mask)[
            0
        ]  # output last hidden state
        tweet_embeddings = self.pooling(token_embeddings, attention_mask=attention_mask)
        return tweet_embeddings


class TransformerRegressor(nn.Module):
    """Adds a regression head, regression loss and regression metrics to a transformer encoder."""

    def __init__(self, encoder: TransformerEncoder):
        super().__init__()

        self.encoder = encoder

        self.head = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 1),
        )

        # store the loss function here to allow a unified training loop
        self.loss_func = nn.MSELoss()

        self.metrics = MetricCollection(
            [
                MeanSquaredError(squared=False),
            ]
        )

    def forward(self, input_ids, attention_mask):
        tweet_embeddings = self.encoder(input_ids, attention_mask=attention_mask)
        logits = self.head(tweet_embeddings)
        return logits.squeeze()


class TransformerClassifier(nn.Module):
    """Adds a classification head, classification loss and classification metrics to a transformer encoder."""

    def __init__(self, encoder: TransformerEncoder, class_weights: list[float] = None):
        super().__init__()

        self.encoder = encoder

        self.num_classes = 3

        self.head = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, self.num_classes)
        )

        # store the loss function and metrics here to allow a unified training loop
        if class_weights is not None:
            class_weights = torch.tensor(class_weights)
        self.loss_func = nn.CrossEntropyLoss(weight=class_weights)

        self.metrics = MetricCollection(
            [
                MulticlassAccuracy(num_classes=self.num_classes, average="micro"),
                MulticlassF1Score(
                    num_classes=self.num_classes,
                    average="macro",  # we have a class imbalance, so we use micro averaging
                ),
                MulticlassPrecision(num_classes=self.num_classes, average="macro"),
                MulticlassRecall(num_classes=self.num_classes, average="macro"),
            ]
        )
        self.val_confmat = MulticlassConfusionMatrix(num_classes=self.num_classes)

    def forward(self, input_ids, attention_mask):
        tweet_embeddings = self.encoder(input_ids, attention_mask=attention_mask)
        logits = self.head(tweet_embeddings)
        return logits


class Projector(nn.Module):
    """A simple projection head for contrastive learning. This projects the embeddings into a different space, where they are compared."""

    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.model = nn.Sequential(
            nn.Dropout1d(),
            nn.Linear(hidden_size, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 128),
        )

    def forward(self, x):
        return self.model(x)


class SupervisedContrastiveEncoder(nn.Module):
    """A contrastive encoder that can be used for supervised contrastive learning. It normalizes the embeddings and projects them if a projector is given."""

    def __init__(self, encoder: TransformerEncoder, projector: nn.Module = None):
        super().__init__()
        self.encoder = encoder
        self.tokenizer = self.encoder.tokenizer
        self.projector = projector
        self.loss_func = util.SCLoss()

    def forward(self, input_ids, attention_mask):
        z = F.normalize(self.encoder(input_ids, attention_mask), dim=1)

        if self.projector:
            z = F.normalize(self.projector(z), dim=1)

        return z


class SupervisedContrastiveClassifier(nn.Module):
    """Adds classification head, classification loss and classification metrics to a supervised contrastive encoder."""

    def __init__(
        self,
        encoder: TransformerEncoder,
        class_weights: list[float] = None,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.encoder = SupervisedContrastiveEncoder(encoder=encoder)
        self.tokenizer = self.encoder.tokenizer
        self.num_classes = 3
        self.head = nn.Linear(self.encoder.encoder.config.hidden_size, self.num_classes)
        self.loss_scl = util.SCLoss(temperature=temperature)
        if class_weights is not None:
            class_weights = torch.tensor(class_weights)
        self.loss_ce = nn.CrossEntropyLoss(weight=class_weights)

        # we have a class imbalance, so we use micro averaging
        self.metrics = MetricCollection(
            [
                MulticlassAccuracy(num_classes=self.num_classes, average="micro"),
                MulticlassF1Score(num_classes=self.num_classes, average="micro"),
                MulticlassPrecision(num_classes=self.num_classes, average="micro"),
                MulticlassRecall(num_classes=self.num_classes, average="micro"),
            ]
        )
        self.val_confmat = MulticlassConfusionMatrix(num_classes=self.num_classes)

    def forward(self, input_ids, attention_mask):
        embeddings = self.encoder(input_ids, attention_mask)
        logits = self.head(embeddings)
        return embeddings, logits.squeeze()


################################
#                              #
#             SVMs             #
#                              #
################################

preprocessing = ColumnTransformer(
    [
        (
            "normalize",
            StandardScaler(),
            [
                "retweet_count",
                "quote_count",
                "reply_count",
                "like_count",
                "followers_count",
                "following_count",
                "tweet_count",
                "listed_count",
            ],
        ),
        ("onehot", OneHotEncoder(), ["type", "author_id", "possibly_sensitive"]),
        ("passthrough", "passthrough", ["bert_score", "bert_sentiment"]),
    ],
    remainder="drop",
)

svc_pipeline = Pipeline(
    [("features", preprocessing), ("svm", SVC(C=1, kernel="poly", degree=3))]
)

svr_pipeline = Pipeline(
    [("features", preprocessing), ("svm", SVR(C=50, kernel="linear"))]
)
