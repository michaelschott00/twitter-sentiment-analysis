import unittest

import numpy as np
import torch

from twitter.util import SCLoss


class UtilTests(unittest.TestCase):
    """https://github.com/google-research/google-research/blob/master/supcon/losses_test.py"""

    def setUp(self):
        self.loss_func = SCLoss()
        self.loss_func_temp = SCLoss(temperature=0.1)
        self.loss_func_no_reduce = SCLoss(reduction="none")

    def test_loss_value_with_labels(self):
        sqrt2 = np.sqrt(2.)
        sqrt6 = np.sqrt(6.)
        features = torch.tensor([[0, 0, 1],
                                 [sqrt6 / 3., -sqrt2 / 3., -1. / 3],
                                 [0, (2. * sqrt2) / 3., -1 / 3.],
                                 [-sqrt6 / 3., -sqrt2 / 3., -1. / 3]])
        labels = torch.eye(2, dtype=torch.int32)
        labels = torch.cat([labels, labels], dim=0)
        loss = self.loss_func(features, labels=labels)
        self.assertFalse(np.isnan(loss.numpy()).any())
        expected_loss = 1.098612  # np.log(3.)
        self.assertAlmostEqual(np.mean(loss.numpy()), expected_loss, places=6)

    def test_loss_value_with_labels_and_positives(self):
        features = torch.tensor([[0, 0, 1],
                                 [0, 1, 0],
                                 [1, 0, 0],
                                 [0, 0, 1],
                                 [0, 1, 0],
                                 [1, 0, 0]], dtype=torch.float)
        labels = torch.eye(3, dtype=torch.int32)
        labels[1] = labels[0]
        labels = torch.cat([labels, labels], dim=0)
        # Make the label of sample 1 and 2 the same (= label 0)
        loss = self.loss_func_no_reduce(features, labels).numpy()
        print(loss)
        self.assertFalse(np.isnan(loss).any())
        expected_loss = [
            1.571499,  # (3. * np.log(np.e + 4) - 1) / 3.
            1.571499,  # (3. * np.log(np.e + 4) - 1) / 3.
            0.904832,  # np.log(np.e + 4) - 1
        ]
        self.assertAlmostEqual(loss[0], expected_loss[0], places=6)
        self.assertAlmostEqual(loss[1], expected_loss[1], places=6)
        self.assertAlmostEqual(loss[2], expected_loss[2], places=6)

    def testLossValueWithTemp(self):
        sqrt2 = np.sqrt(2.)
        sqrt6 = np.sqrt(6.)
        features = torch.tensor([[0, 0, 1],
                                 [sqrt6 / 3., -sqrt2 / 3., -1. / 3],
                                 [0, (2. * sqrt2) / 3., -1 / 3.],
                                 [-sqrt6 / 3., -sqrt2 / 3., -1. / 3]])
        labels = torch.eye(2, dtype=torch.int32)
        labels = torch.cat([labels, labels], dim=0)
        loss = self.loss_func_temp(features, labels)
        self.assertFalse(np.isnan(loss.numpy()).any())
        expected_loss = 0.1098612  # 0.1 * np.log(3.)
        self.assertAlmostEqual(np.mean(loss.numpy()), expected_loss, places=5)


if __name__ == '__main__':
    unittest.main()
