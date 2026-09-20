"""Guards for the task configs used by the GPU training entry points.

Cheap, dependency-free checks on ``configs/tasks/*.yaml`` so a config regression
(e.g. dropping ``save_weights_only``) fails locally instead of on an expensive
RunPod run.
"""

import unittest

import yaml

TASK_CONFIGS = [
    "configs/tasks/classification.yaml",
    "configs/tasks/multitask.yaml",
    "configs/tasks/regression.yaml",
]


def _model_checkpoints(path):
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    return [
        cb
        for cb in cfg["trainer"]["callbacks"] or []
        if cb["class_path"].endswith("ModelCheckpoint")
    ]


class TaskConfigTests(unittest.TestCase):
    def test_model_checkpoints_save_weights_only(self):
        """A full checkpoint is ~3x larger (it also stores the Adam moments),
        and that bundle timed out the Azure ML artifact upload during model
        registration. ``save_weights_only`` keeps hyperparameters, so
        ``load_from_checkpoint`` still rebuilds the module."""
        for path in TASK_CONFIGS:
            with self.subTest(config=path):
                checkpoints = _model_checkpoints(path)
                self.assertTrue(checkpoints, f"no ModelCheckpoint in {path}")
                for cb in checkpoints:
                    self.assertIs(
                        cb["init_args"].get("save_weights_only"),
                        True,
                        f"{path} must set save_weights_only: true",
                    )
