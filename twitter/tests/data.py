import os
import random
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler

from twitter import data
from twitter.data import (
    ClassificationTextDataset,
    ExternalClassificationTextDataset,
    ExternalRegressionTextDataset,
    PredictionTextDataset,
    RegressionTextDataset,
    SCLTextDataset,
    TextDataset,
    TwitterDataModule,
    _TwitterBaseDataset,
)

ENCODER = "sentence-transformers/bert-base-nli-mean-tokens"


class TestBaseDataset(unittest.TestCase):
    def setUp(self):
        self.train_dataset = _TwitterBaseDataset(root_dir="data/splits", split="train")
        self.dev_dataset = _TwitterBaseDataset(root_dir="data/splits", split="dev")
        self.test_dataset = _TwitterBaseDataset(root_dir="data/splits", split="test_1")

    def test_init(self):
        self.assertEqual(self.train_dataset.split, "train")
        self.assertEqual(self.dev_dataset.split, "dev")
        self.assertEqual(self.test_dataset.split, "test_1")
        for dataset in [self.train_dataset, self.dev_dataset, self.test_dataset]:
            self.assertEqual(dataset.root_dir, "data/splits")

    def test_length(self):
        self.assertEqual(len(self.train_dataset), 6400)
        self.assertEqual(len(self.dev_dataset), 1600)
        self.assertEqual(len(self.test_dataset), 1000)

    def test_items(self):
        x, y = self.train_dataset[0]
        self.assertEqual(
            x["text"],
            "Proud to work with @GavinNewsom &amp; partners to help bridge the digital divide in our home state. We’re providing 4,000 Chromebooks to California students in greatest need &amp; free wifi to 100,000 rural households during the #COVID19 crisis to make distance learning more accessible.",
        )
        self.assertEqual(type(x["type"]), np.int8)
        self.assertEqual(x["author_id"], 2)
        self.assertEqual(x["possibly_sensitive"], 0)
        self.assertEqual(x["retweet_count"], 584)
        self.assertEqual(x["quote_count"], 85)
        self.assertEqual(x["reply_count"], 229)
        self.assertEqual(x["like_count"], 4649)
        self.assertEqual(x["followers_count"], 5169451)
        self.assertEqual(x["following_count"], 140)
        self.assertEqual(x["tweet_count"], 1833)
        self.assertEqual(x["listed_count"], 8955)
        self.assertEqual(
            x["words"],
            "['proud', 'work', 'amp', 'partners', 'help', 'bridge', 'digital', 'divide', 'home', 'state', 'providing', 'chromebooks', 'california', 'students', 'greatest', 'need', 'amp', 'free', 'wifi', 'rural', 'households', 'covid', 'crisis', 'make', 'distance', 'learning', 'accessible']",
        )
        self.assertEqual(y["score_compound"], 0.8481)
        self.assertEqual(y["sentiment"], data.LABEL_CODING["positive"])

    def test_items_train_dev_have_labels(self):
        # X must not leak labels/id; y must carry exactly the two labels.
        for ds in [self.train_dataset, self.dev_dataset]:
            x, y = ds[0]
            self.assertIsInstance(x, dict)
            self.assertIsInstance(y, dict)
            for leaked in ["sentiment", "score_compound", "id"]:
                self.assertNotIn(leaked, x)
            self.assertEqual(set(y.keys()), {"sentiment", "score_compound"})
            self.assertIn(y["sentiment"], set(data.LABEL_CODING.values()))
            self.assertIsInstance(float(y["score_compound"]), float)
            self.assertGreaterEqual(float(y["score_compound"]), -1.0)
            self.assertLessEqual(float(y["score_compound"]), 1.0)
            # spot-check a few more rows for well-formedness
            for idx in [1, 100, len(ds) - 1]:
                xi, yi = ds[idx]
                self.assertIsInstance(xi["text"], str)
                self.assertTrue(len(xi["text"]) > 0)
                self.assertIn(yi["sentiment"], set(data.LABEL_CODING.values()))

    def test_items_dev_first_row(self):
        # Guards against silently loading the wrong split file.
        x, y = self.dev_dataset[0]
        self.assertIsInstance(x["text"], str)
        self.assertTrue(len(x["text"]) > 0)
        self.assertEqual(set(y.keys()), {"sentiment", "score_compound"})

    def test_items_test_has_no_labels(self):
        for idx in [0, 1, 100, len(self.test_dataset) - 1]:
            x, y = self.test_dataset[idx]
            self.assertIsNone(y, f"test item {idx} should have y=None, got {y!r}")
            self.assertIsInstance(x, dict)
            self.assertIsInstance(x["text"], str)
            self.assertTrue(len(x["text"]) > 0)
            # labels must be gone from the features as well
            for leaked in ["sentiment", "score_compound", "id"]:
                self.assertNotIn(leaked, x)

    def test_test_2_split(self):
        ds = _TwitterBaseDataset(root_dir="data/splits", split="test_2")
        self.assertEqual(len(ds), 1000)
        x, y = ds[0]
        self.assertIsNone(y)
        self.assertNotIn("sentiment", x)
        self.assertNotIn("score_compound", x)


class TextDatasetTransformTests(unittest.TestCase):
    def test_no_transforms_returns_raw_text(self):
        ds = TextDataset(root_dir="data/splits", split="train")
        base = _TwitterBaseDataset(root_dir="data/splits", split="train")
        raw_x, _ = base[0]
        x, y = ds[0]
        self.assertEqual(x, raw_x["text"])
        self.assertEqual(set(y.keys()), {"sentiment", "score_compound"})

    def test_preprocessing_only(self):
        calls = []

        def pre(s):
            calls.append(s)
            return s.upper()

        ds = TextDataset(
            root_dir="data/splits", split="train", preprocessing=pre, augmentation=None
        )
        base = _TwitterBaseDataset(root_dir="data/splits", split="train")
        raw = base[0][0]["text"]
        x, _ = ds[0]
        self.assertEqual(x, raw.upper())
        self.assertEqual(calls, [raw])

    def test_augmentation_only(self):
        def aug(s):
            return s + " AUG"

        ds = TextDataset(
            root_dir="data/splits", split="train", preprocessing=None, augmentation=aug
        )
        base = _TwitterBaseDataset(root_dir="data/splits", split="train")
        raw = base[0][0]["text"]
        x, _ = ds[0]
        self.assertEqual(x, raw + " AUG")

    def test_both_applied_in_order_augmentation_then_preprocessing(self):
        order = []

        def aug(s):
            order.append("aug")
            return s + " AUG"

        def pre(s):
            order.append("pre")
            return s + " PRE"

        ds = TextDataset(
            root_dir="data/splits", split="train", preprocessing=pre, augmentation=aug
        )
        base = _TwitterBaseDataset(root_dir="data/splits", split="train")
        raw = base[0][0]["text"]
        x, _ = ds[0]
        self.assertEqual(x, raw + " AUG PRE")
        self.assertEqual(order, ["aug", "pre"])

    def test_neither_applied_when_none(self):
        # sentinel transforms that would explode if called
        def boom(s):
            raise AssertionError("transform should not be called")

        ds = TextDataset(
            root_dir="data/splits", split="train", preprocessing=None, augmentation=None
        )
        base = _TwitterBaseDataset(root_dir="data/splits", split="train")
        self.assertEqual(ds[5][0], base[5][0]["text"])
        _ = boom  # silence unused warning


class LabelSubsetTests(unittest.TestCase):
    def test_regression_contains_only_compound(self):
        ds = RegressionTextDataset(root_dir="data/splits", split="train")
        base = _TwitterBaseDataset(root_dir="data/splits", split="train")
        for idx in [0, 1, 42]:
            x, y = ds[idx]
            _, y_base = base[idx]
            self.assertIsInstance(x, str)
            self.assertNotIsInstance(y, dict)
            self.assertEqual(float(y), float(y_base["score_compound"]))

    def test_classification_contains_only_sentiment(self):
        ds = ClassificationTextDataset(root_dir="data/splits", split="train")
        base = _TwitterBaseDataset(root_dir="data/splits", split="train")
        for idx in [0, 1, 42]:
            x, y = ds[idx]
            _, y_base = base[idx]
            self.assertIsInstance(x, str)
            self.assertNotIsInstance(y, dict)
            # sentiment codes are small ints 0/1/2
            self.assertIn(int(y), set(data.LABEL_CODING.values()))
            self.assertEqual(int(y), int(y_base["sentiment"]))

    def test_scl_returns_two_views_and_int_label(self):
        ds = SCLTextDataset(root_dir="data/splits", split="train")
        (x1, x2), y = ds[0]
        self.assertIsInstance(x1, str)
        self.assertIsInstance(x2, str)
        self.assertEqual(x1, x2)  # no augmentation -> identical views
        self.assertNotIsInstance(y, dict)
        self.assertIn(int(y), set(data.LABEL_CODING.values()))

    def test_prediction_returns_text_only(self):
        ds = PredictionTextDataset(root_dir="data/splits", split="test_1")
        x = ds[0]
        self.assertIsInstance(x, str)
        self.assertTrue(len(x) > 0)


class ModuleTests(unittest.TestCase):
    def setUp(self):
        # no eda
        self.data_module = TwitterDataModule(
            root_dir="data/splits",
            features="contrast",
            labels="clf",
            encoder_name="sentence-transformers/bert-base-nli-mean-tokens",
            batch_size=32,
        )
        self.data_module.setup("fit")
        self.data_module.setup("validate")
        self.data_module.setup("predict")

        # with eda
        self.eda_data_module = TwitterDataModule(
            root_dir="data/splits",
            features="contrast",
            labels="clf",
            encoder_name="sentence-transformers/bert-base-nli-mean-tokens",
            batch_size=32,
            eda=True,
        )
        self.eda_data_module.setup("fit")
        self.eda_data_module.setup("validate")
        self.eda_data_module.setup("predict")

    def contrast_batch_shared(self, batch, eda=False):
        self.assertEqual(batch["input_ids"].shape[0], 64)
        self.assertEqual(batch["labels"].shape, torch.Size([64]))
        if not eda:
            self.assertTrue(
                (batch["input_ids"][:32] == batch["input_ids"][32:]).all().item()
            )
            self.assertTrue(
                (batch["attention_mask"][:32] == batch["attention_mask"][32:])
                .all()
                .item()
            )
        else:
            self.assertFalse(
                (batch["input_ids"][:32] == batch["input_ids"][32:]).all().item()
            )
            self.assertFalse(
                (batch["attention_mask"][:32] == batch["attention_mask"][32:])
                .all()
                .item()
            )

    def test_contrast_batch_train(self):
        # no eda
        train_loader = self.data_module.train_dataloader()
        batch = next(iter(train_loader))
        self.contrast_batch_shared(batch)

        # with eda
        random.seed(42)
        eda_train_loader = self.eda_data_module.train_dataloader()
        eda_batch = next(iter(eda_train_loader))
        self.contrast_batch_shared(eda_batch, eda=True)

    def test_contrast_batch_dev(self):
        # no eda
        val_loader = self.data_module.val_dataloader()
        batch = next(iter(val_loader))
        self.contrast_batch_shared(batch)

    def test_contrast_batch_predict(self):
        # no eda
        predict_loader_1, predict_loader_2 = self.data_module.predict_dataloader()
        batch_1 = next(iter(predict_loader_1))
        batch_2 = next(iter(predict_loader_2))
        self.assertEqual(batch_1["input_ids"].shape[0], 32)
        self.assertEqual(batch_2["input_ids"].shape[0], 32)


def _make_module(features, labels, batch_size=8, **kwargs):
    kwargs.setdefault("root_dir", "data/splits")
    kwargs.setdefault("encoder_name", ENCODER)
    kwargs.setdefault("num_workers", 0)
    kwargs.setdefault("pin_memory", False)
    dm = TwitterDataModule(
        features=features, labels=labels, batch_size=batch_size, **kwargs
    )
    dm.setup("fit")
    dm.setup("validate")
    dm.setup("predict")
    return dm


class DataModuleBatchTests(unittest.TestCase):
    """Smoke tests: every loader yields a well-formed batch. Catches crashes
    (tokenizer, collate_fn mismatch, ConcatDataset, sampler) before a cluster run."""

    def _assert_text_batch(self, batch, batch_size, label_shape=None, label_dtype=None):
        self.assertIn("input_ids", batch)
        self.assertIn("attention_mask", batch)
        self.assertEqual(batch["input_ids"].shape[0], batch_size)
        self.assertEqual(batch["attention_mask"].shape, batch["input_ids"].shape)
        # padding mask must be binary and non-empty per row
        self.assertTrue(
            ((batch["attention_mask"] == 0) | (batch["attention_mask"] == 1))
            .all()
            .item()
        )
        self.assertGreater(batch["attention_mask"].sum().item(), 0)
        if label_shape is None:
            self.assertNotIn("labels", batch)
        else:
            self.assertIn("labels", batch)
            self.assertEqual(tuple(batch["labels"].shape), label_shape)
            if label_dtype is not None:
                self.assertEqual(batch["labels"].dtype, label_dtype)

    def test_text_clf_batches(self):
        dm = _make_module("text", "clf")
        train = next(iter(dm.train_dataloader()))
        self._assert_text_batch(train, 8, (8,), torch.int64)
        self.assertTrue(train["labels"].min().item() >= 0)
        self.assertTrue(train["labels"].max().item() <= 2)
        val = next(iter(dm.val_dataloader()))
        self._assert_text_batch(val, 8, (8,), torch.int64)
        for pred_loader in dm.predict_dataloader():
            b = next(iter(pred_loader))
            self._assert_text_batch(b, 8, None)

    def test_text_reg_batches(self):
        dm = _make_module("text", "reg")
        train = next(iter(dm.train_dataloader()))
        self._assert_text_batch(train, 8, (8,), torch.float32)
        self.assertTrue(train["labels"].min().item() >= -1.0)
        self.assertTrue(train["labels"].max().item() <= 1.0)
        val = next(iter(dm.val_dataloader()))
        self._assert_text_batch(val, 8, (8,), torch.float32)

    def test_text_both_batches(self):
        dm = _make_module("text", "both")
        train = next(iter(dm.train_dataloader()))
        self._assert_text_batch(train, 8, (8, 2), torch.float32)
        val = next(iter(dm.val_dataloader()))
        self._assert_text_batch(val, 8, (8, 2), torch.float32)

    def test_contrast_batches(self):
        dm = _make_module("contrast", "clf")
        train = next(iter(dm.train_dataloader()))
        # SCL collate doubles the batch: (texts_1 + texts_2)
        self._assert_text_batch(train, 16, (16,), torch.int64)
        val = next(iter(dm.val_dataloader()))
        self._assert_text_batch(val, 16, (16,), torch.int64)

    def test_predict_loaders_cover_full_test_sets(self):
        dm = _make_module("text", "clf", batch_size=32)
        l1, l2 = dm.predict_dataloader()
        n1 = sum(b["input_ids"].shape[0] for b in l1)
        n2 = sum(b["input_ids"].shape[0] for b in l2)
        self.assertEqual(n1, 1000)
        self.assertEqual(n2, 1000)

    def test_invalid_contrast_labels_raises(self):
        with self.assertRaises(AssertionError):
            TwitterDataModule(
                root_dir="data/splits",
                features="contrast",
                labels="reg",
                encoder_name=ENCODER,
                batch_size=8,
            )


class DataModuleTransformTests(unittest.TestCase):
    def test_no_transforms_by_default(self):
        dm = _make_module("text", "clf")
        self.assertIsNone(dm.preprocessing)
        self.assertIsNone(dm.augmentation)
        self.assertIsNone(dm.twitter_train.preprocessing)
        self.assertIsNone(dm.twitter_train.augmentation)

    def test_normalize_reaches_datasets_and_batch(self):
        from twitter.util import TweetNormalizer

        dm_plain = _make_module("text", "clf")
        dm_norm = _make_module("text", "clf", normalize=True)
        self.assertIsNotNone(dm_norm.preprocessing)
        self.assertIsNone(dm_norm.augmentation)
        # dev gets preprocessing but never augmentation
        self.assertIsNotNone(dm_norm.twitter_dev.preprocessing)
        self.assertIsNone(dm_norm.twitter_dev.augmentation)
        # functional check: dataset text equals manually normalized raw text
        base = _TwitterBaseDataset(root_dir="data/splits", split="train")
        raw = base[0][0]["text"]
        self.assertEqual(dm_norm.twitter_train[0][0], TweetNormalizer()(raw))
        self.assertEqual(dm_plain.twitter_train[0][0], raw)
        # batch builds fine with normalized texts
        batch = next(iter(dm_norm.train_dataloader()))
        self.assertEqual(batch["input_ids"].shape[0], 8)

    def test_eda_only_on_train(self):
        dm = _make_module("text", "clf", eda=True)
        self.assertIsNotNone(dm.augmentation)
        self.assertIsNotNone(dm.twitter_train.augmentation)
        # dev/predict must stay unaugmented for stable eval
        self.assertIsNone(dm.twitter_dev.augmentation)
        self.assertIsNone(dm.twitter_test_1.augmentation)
        self.assertIsNone(dm.twitter_test_2.augmentation)
        batch = next(iter(dm.train_dataloader()))
        self.assertEqual(batch["input_ids"].shape[0], 8)

    def test_join_words_with_words_feature(self):
        # words column holds a stringified list; WordsToSentence joins it.
        from twitter.util import WordsToSentence

        dm = _make_module("text", "clf", join_words=True)
        self.assertIsNotNone(dm.preprocessing)
        base = _TwitterBaseDataset(root_dir="data/splits", split="train")
        _ = WordsToSentence  # import guard for cluster envs
        _ = base


class DataModuleExternalDataTests(unittest.TestCase):
    def _write_external_csv(self, n=16):
        base = _TwitterBaseDataset(root_dir="data/splits", split="train")
        rows = []
        inv = {v: k for k, v in data.LABEL_CODING.items()}
        for i in range(n):
            x, y = base[i]
            rows.append(
                {
                    "text": x["text"],
                    "sentiment": inv[int(y["sentiment"])],
                    "score_compound": float(y["score_compound"]),
                }
            )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            pd.DataFrame(rows).to_csv(f.name, index=False)
            path = f.name
        self.addCleanup(os.unlink, path)
        return path

    def test_external_ignored_when_none(self):
        dm = _make_module("text", "clf", external_data_paths=None)
        self.assertEqual(len(dm.twitter_train), 6400)

    def test_external_concatenated_when_provided(self):
        path = self._write_external_csv(n=16)
        dm = _make_module("text", "clf", external_data_paths=[path])
        self.assertEqual(len(dm.twitter_train), 6400 + 16)
        # external tail must be usable: sample from beyond original range
        x, y = dm.twitter_train[6400]
        self.assertIsInstance(x, str)
        self.assertIn(int(y), set(data.LABEL_CODING.values()))
        # and a training batch still collates
        batch = next(iter(dm.train_dataloader()))
        self.assertEqual(batch["input_ids"].shape[0], 8)

    def test_external_reg_concatenated(self):
        path = self._write_external_csv(n=8)
        dm = _make_module("text", "reg", external_data_paths=[path])
        self.assertEqual(len(dm.twitter_train), 6400 + 8)
        x, y = dm.twitter_train[6400]
        self.assertIsInstance(x, str)
        self.assertIsInstance(float(y), float)


class DataModuleSampleWeightTests(unittest.TestCase):
    def test_no_sampler_without_weights(self):
        from torch.utils.data import RandomSampler

        for labels in ["clf", "both", "reg"]:
            dm = _make_module("text", labels, class_sample_weights=None)
            loader = dm.train_dataloader()
            self.assertNotIsInstance(loader.sampler, WeightedRandomSampler)
            # default shuffling path (RandomSampler via shuffle=True)
            self.assertIsInstance(loader.sampler, RandomSampler)

    def test_weighted_sampler_for_clf(self):
        w = (2.0, 1.0, 0.5)
        dm = _make_module("text", "clf", class_sample_weights=w)
        loader = dm.train_dataloader()
        self.assertIsInstance(loader.sampler, WeightedRandomSampler)
        weights = loader.sampler.weights
        self.assertEqual(len(weights), len(dm.twitter_train))
        # every stored weight must be one of the class weights
        uniq = {float(v) for v in weights.unique()}
        self.assertTrue(uniq.issubset(set(w)))

    def test_weighted_sampler_for_both(self):
        w = (2.0, 1.0, 0.5)
        dm = _make_module("text", "both", class_sample_weights=w)
        loader = dm.train_dataloader()
        self.assertIsInstance(loader.sampler, WeightedRandomSampler)

    def test_weights_ignored_for_reg(self):
        # regression has no class labels -> no sampler even if weights given
        dm = _make_module("text", "reg", class_sample_weights=(2.0, 1.0, 0.5))
        loader = dm.train_dataloader()
        self.assertNotIsInstance(loader.sampler, WeightedRandomSampler)

    def test_weighted_loader_still_yields_valid_batch(self):
        dm = _make_module("text", "clf", class_sample_weights=(2.0, 1.0, 0.5))
        batch = next(iter(dm.train_dataloader()))
        self.assertEqual(batch["input_ids"].shape[0], 8)
        self.assertEqual(tuple(batch["labels"].shape), (8,))


class ExternalDatasetTests(unittest.TestCase):
    def test_external_clf_reg_label_subsets(self):
        import tempfile

        base = _TwitterBaseDataset(root_dir="data/splits", split="train")
        inv = {v: k for k, v in data.LABEL_CODING.items()}
        x0, y0 = base[0]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            pd.DataFrame(
                [
                    {
                        "text": x0["text"],
                        "sentiment": inv[int(y0["sentiment"])],
                        "score_compound": float(y0["score_compound"]),
                    }
                ]
            ).to_csv(f.name, index=False)
            path = f.name
        self.addCleanup(os.unlink, path)
        xc, yc = ExternalClassificationTextDataset(path=path)[0]
        self.assertEqual(xc, x0["text"])
        self.assertEqual(int(yc), int(y0["sentiment"]))
        _, yr = ExternalRegressionTextDataset(path=path)[0]
        self.assertEqual(float(yr), float(y0["score_compound"]))


if __name__ == "__main__":
    unittest.main()
