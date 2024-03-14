from typing import Literal, Callable, Tuple, List, Dict, Union
import os

import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
from transformers import AutoTokenizer
import torch
import numpy as np
from torch import nn

from twitter.util import TweetNormalizer, WordsToSentence
from twitter.augmentation import EDA


################################
#                              #
#           Constants          #
#                              #
################################

# these are required for submitting predictions in the correct format
LABEL_CODING: Dict[str, int] = {'negative': 0, 'neutral': 1, 'positive': 2}
INVERSE_LABEL_CODING: Dict[int, str] = {v: k for k, v in LABEL_CODING.items()}


################################
#                              #
#       Competition Data       #
#                              #
################################

class _BaseDataset(Dataset):
    """A Base dataset class for the Twitter Sentiment Analysis dataset.

    Loads the dataframe, encodes categorical columns and returns rows as dicts.
    """

    LABEL_CODING = LABEL_CODING
    INVERSE_LABEL_CODING = INVERSE_LABEL_CODING

    def __init__(self,
                 root_dir: str,
                 split: Literal["train", "dev", "test_1", "test_2"]) -> None:
        super().__init__()

        # store hparams
        self.split = split
        self.root_dir = root_dir

        # load data
        self.df = pd.read_csv(
            filepath_or_buffer=os.path.join(self.root_dir, f"tweets_{split}.csv"),
            dtype={
                "type": "category",
                "author_id": "category",
                "possibly_sensitive": "category",
            }
        )

        # encode categorical columns except sentiment, because the sentiment column should be encoded
        # according the LABEL_CODING and INVERSE_LABEL_CODING
        for column in self.df.columns:
            if (self.df[column].dtype.name == "category") and not (column == "sentiment"):
                self.df[column] = self.df[column].cat.codes

        # encode sentiment column, except for test set, which does not contain it
        if "test" not in self.split:
            self.df.sentiment = self.df.sentiment.astype("category").apply(lambda x: self.LABEL_CODING[x])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, Union[int, str]], Dict[str, Union[int, float, None]]]:
        """Returns a single observation from the dataset.

        Args:
            idx (int): The index of the observation to return.
        Returns:
            Tuple[Dict[str, Union[int, str]], Dict[str, Union[int, float]]]: The features and labels of the observation. For the test set, the labels are None.
        """
        observation = self.df.iloc[idx]
        if "test" not in self.split:
            X = observation.drop(labels=["sentiment", "score_compound", "id"])
            y = observation[["sentiment", "score_compound"]]
            return dict(X), dict(y)
        else:
            X = observation.drop(labels=["id"])
            return dict(X), None


class TextDataset(_BaseDataset):
    """A Dataset only containing the text column without the metadata."""

    def __init__(self,
                 root_dir: str,
                 split: Literal["train", "dev", "test_1"],
                 preprocessing: Callable = None,
                 augmentation: Callable = None) -> None:
        super().__init__(root_dir=root_dir, split=split)
        self.preprocessing = preprocessing
        self.augmentation = augmentation

    def __getitem__(self, idx: int) -> Tuple[str, Dict[str, Union[int, float]]]:
        X, y = super().__getitem__(idx)
        X = X["text"]
        if self.augmentation:
            X = self.augmentation(X)
        if self.preprocessing:
            X = self.preprocessing(X)
        return X, y


class RegressionTextDataset(TextDataset):
    """A Dataset for the regression task, which only contains the text and the compound score."""

    def __getitem__(self, idx) -> Tuple[str, float]:
        X, y = super().__getitem__(idx)
        return X, y["score_compound"]


class ClassificationTextDataset(TextDataset):
    """A Dataset for the classification task, which only contains the text and the sentiment label."""

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        X, y = super().__getitem__(idx)
        return X, y["sentiment"]


class SCLTextDataset(TextDataset):
    """A dataset for contrastive learning, which returns two augmented views of the same text. This only supports the classification labels, because
    contrastive learning requires class labels."""

    def __init__(self, augmentation: Callable, *args, **kwargs) -> None:
        super().__init__(augmentation=augmentation, *args, **kwargs)

    def __getitem__(self, idx: int) -> Tuple[Tuple[str, str], Dict[str, int]]:
        # get two different augmented views due to randomness
        X_1, _ = super().__getitem__(idx)
        X_2, y = super().__getitem__(idx)
        return (X_1, X_2), y["sentiment"]


class PredictionTextDataset(TextDataset):
    """A Dataset only containing the text without any labels. This is used for the test set, where the labels are not available."""

    def __getitem__(self, idx: int) -> str:
        X, _ = super().__getitem__(idx)
        return X


################################
#                              #
#        External Data         #
#                              #
################################

class ExternalTextDataset(Dataset):
    """A Dataset for external data, which is not part of the original dataset. This is used, e.g. for pre-computed data augmentations."""

    LABEL_CODING = LABEL_CODING
    INVERSE_LABEL_CODING = INVERSE_LABEL_CODING

    def __init__(self,
                 path: str,
                 preprocessing: Callable = None,
                 augmentation: Callable = None,
                 text_column: str = "text") -> None:
        super().__init__()
        self.df = pd.read_csv(path)
        self.preprocessing = preprocessing
        self.augmentation = augmentation
        self.text_column = text_column

        # encode labels explicitly to keep track of it when coding back
        self.df.sentiment = self.df.sentiment.apply(lambda x: self.LABEL_CODING[x])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[str, Dict[str, Union[int, float]]]:
        observation = self.df.iloc[idx]
        X = observation[self.text_column]
        if self.augmentation is not None:
            X = self.augmentation(X)
        if self.preprocessing is not None:
            X = self.preprocessing(X)
        if "sentiment" in observation:
            y = {"sentiment": observation["sentiment"]}
        if "score_compound" in observation:
            y = {"score_compound": observation["score_compound"]}
        return X, y


class ExternalRegressionTextDataset(ExternalTextDataset):
    """A dataset for external data for the regression task, which only contains the text and the compound score."""

    def __getitem__(self, idx: int) -> Tuple[str, float]:
        X, y = super().__getitem__(idx)
        return X, y["score_compound"]


class ExternalClassificationTextDataset(ExternalTextDataset):
    """A dataset for external data for the classification task, which only contains the text and the sentiment."""

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        X, y = super().__getitem__(idx)
        return X, y["sentiment"]


################################
#                              #
#         Data Modules         #
#                              #
################################

class TwitterDataModule(pl.LightningDataModule):
    """A DataModule for the Twitter Sentiment Analysis dataset. Handles all data-related hyperameters and data loading."""

    def __init__(
        self,

        # files
        root_dir: str,

        # labels and features
        features: Literal["text", "words", "contrast"],
        labels: Literal["reg", "clf", "both", "none"],

        # model
        encoder_name: str,

        # hparams
        batch_size: int = 32,
        class_sample_weights: Tuple[float, float, float] = None,  # negative, neutral, positive

        # transforms
        normalize: bool = False,
        join_words: bool = False,

        # cleaning
        drop_mismatched_labels: bool = False,

        # computationally intensive augmentations (path to csv)
        external_data_paths: List[str] = None,

        # easy data augmentations
        eda: bool = False,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        assert not (self.hparams.features == "contrast" and self.hparams.labels != "clf"), "contrastive learning only implemented for clf labels"

        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name, model_max_length=512)

        preprocessing = []
        if normalize:
            preprocessing.append(TweetNormalizer())
        if join_words:
            preprocessing.append(WordsToSentence())
        self.preprocessing = nn.Sequential(*preprocessing) if len(preprocessing) > 0 else None

        augmentation = []
        if eda:
            augmentation.append(EDA())
        self.augmentation = nn.Sequential(*augmentation) if len(augmentation) > 0 else None

        self.collate_fn_map = {
            TextDataset: self.text_collate_fn,
            RegressionTextDataset: self.text_reg_collate_fn,
            ClassificationTextDataset: self.text_clf_collate_fn,
            SCLTextDataset: self.text_scl_collate_fn,
            PredictionTextDataset: self.text_only_collate_fn,
            ExternalTextDataset: self.text_collate_fn,
            ExternalRegressionTextDataset: self.text_reg_collate_fn,
            ExternalClassificationTextDataset: self.text_clf_collate_fn
        }

    def setup(self, stage: str):
        if self.hparams.features == "text" and self.hparams.labels == "reg":
            Trainset_cls = RegressionTextDataset
            External_cls = ExternalRegressionTextDataset
        elif self.hparams.features == "text" and self.hparams.labels == "clf":
            Trainset_cls = ClassificationTextDataset
            External_cls = ExternalClassificationTextDataset
        elif self.hparams.features == "text" and self.hparams.labels == "both":
            Trainset_cls = TextDataset
            External_cls = ExternalTextDataset
        elif self.hparams.features == "text" and self.hparams.labels == "none":
            Trainset_cls = PredictionTextDataset
            External_cls = ExternalTextDataset
        elif self.hparams.features == "contrast":
            Trainset_cls = SCLTextDataset
            External_cls = ExternalTextDataset
        else:
            raise NotImplementedError

        Devset_cls = Trainset_cls

        # Load train and dev data
        if stage == "fit":
            self.twitter_train = Trainset_cls(root_dir=self.hparams.root_dir,
                                              split="train",
                                              preprocessing=self.preprocessing,
                                              augmentation=self.augmentation)

            if self.hparams.external_data_paths:
                external_datasets = []
                for path in self.hparams.external_data_paths:
                    external_datasets.append(External_cls(
                        path=path,
                        augmentation=self.augmentation,
                        preprocessing=self.preprocessing,
                    ))
                self.twitter_train = torch.utils.data.ConcatDataset([self.twitter_train] + external_datasets)

            self.twitter_dev = Devset_cls(root_dir=self.hparams.root_dir,
                                          split="dev",
                                          preprocessing=self.preprocessing)

        if stage == "validate":
            self.twitter_dev = Devset_cls(root_dir=self.hparams.root_dir,
                                          split="dev",
                                          preprocessing=self.preprocessing)

        # Load test data
        if stage == "predict":
            self.twitter_test_1 = PredictionTextDataset(root_dir=self.hparams.root_dir,
                                                         split="test_1",
                                                         preprocessing=self.preprocessing)
            self.twitter_test_2 = PredictionTextDataset(root_dir=self.hparams.root_dir,
                                                         split="test_2",
                                                         preprocessing=self.preprocessing)

    def prepare_texts(self, texts):
        return self.tokenizer(texts, truncation=True, padding="longest", return_tensors="pt")

    def text_collate_fn(self, batch):
        texts, labels = zip(*batch)
        encodings = self.prepare_texts(texts)
        encodings["labels"] = torch.tensor([(x['score_compound'], x['sentiment']) for x in labels], dtype=torch.float32)
        return encodings

    def text_reg_collate_fn(self, batch):
        texts, labels = zip(*batch)
        encodings = self.prepare_texts(texts)
        encodings["labels"] = torch.tensor(labels, dtype=torch.float32)
        return encodings

    def text_clf_collate_fn(self, batch):
        texts, labels = zip(*batch)
        encodings = self.prepare_texts(texts)
        encodings["labels"] = torch.tensor(labels)
        return encodings

    def text_scl_collate_fn(self, batch):
        texts, labels = zip(*batch)
        texts_1, texts_2 = zip(*texts)
        texts = texts_1 + texts_2
        labels = labels + labels
        encodings = self.prepare_texts(texts)
        encodings["labels"] = torch.tensor(labels)
        return encodings

    def text_only_collate_fn(self, batch):
        texts = batch
        encodings = self.prepare_texts(texts)
        return encodings

    def train_dataloader(self):
        sampler = None
        if self.hparams.class_sample_weights is not None:
            labels = np.array([label for _, label in self.twitter_train])
            if self.hparams.labels == "clf":
                samples_weight = torch.from_numpy(np.array([self.hparams.class_sample_weights[t] for t in labels]))
            elif self.hparams.labels == "both":
                samples_weight = torch.from_numpy(np.array([self.hparams.class_sample_weights[int(t[1])] for t in labels]))
            sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

        return DataLoader(self.twitter_train,
                          batch_size=self.hparams.batch_size,
                          shuffle=sampler is None,
                          sampler=sampler,
                          collate_fn=self.collate_fn_map[self.twitter_train.__class__])

    def val_dataloader(self):
        return DataLoader(self.twitter_dev,
                          batch_size=self.hparams.batch_size,
                          shuffle=False,
                          collate_fn=self.collate_fn_map[self.twitter_dev.__class__])

    def predict_dataloader(self):
        dataloader_1 = DataLoader(self.twitter_test_1,
                                  batch_size=self.hparams.batch_size,
                                  shuffle=False,
                                  collate_fn=self.collate_fn_map[self.twitter_test_1.__class__])
        dataloader_2 = DataLoader(self.twitter_test_2,
                                  batch_size=self.hparams.batch_size,
                                  shuffle=False,
                                  collate_fn=self.collate_fn_map[self.twitter_test_2.__class__])
        return [dataloader_1, dataloader_2]
        # return [dataloader_2]
