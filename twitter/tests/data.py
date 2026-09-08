import random
import unittest

import numpy as np
import torch

from twitter import data
from twitter.data import TwitterDataModule, _TwitterBaseDataset


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


if __name__ == "__main__":
    unittest.main()
