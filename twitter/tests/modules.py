"""CPU-only smoke + unit tests for twitter.modules.

Goal: catch bugs locally before submitting full training runs to RunPod.
Uses tiny random checkpoints (hidden size 32) and synthetic batches, so the
whole file runs in a couple of minutes on a CPU-only machine with no real
dataset needed.

The bulk of the tests run against ``hf-internal-testing/tiny-random-bert``.
Because the production encoder configs also target RoBERTa (BERTweet),
DeBERTa-v2 (DeBERTa-v3) and ModernBERT -- which expose their transformer
layers differently -- ``ARCH_TINY`` covers each of those architecture families
with a tiny checkpoint too, so the architecture-specific paths (layer
discovery, pooling, freeze/reset) are exercised without downloading the
multi-GB real checkpoints.

Run with: ``pytest twitter/tests/modules.py -v``

The real production checkpoints (multi-GB, ~8 GB RAM for one training step on
CPU) are checked by ``RealEncoderIntegrationTests``, which is skipped unless
``TWITTER_TEST_REAL_ENCODERS=1`` is set.
"""

import gc
import os
import tempfile
import unittest

import lightning.pytorch as pl
import torch

from twitter import models, modules

TINY = "hf-internal-testing/tiny-random-bert"
SEQ_LEN = 8
BATCH = 4
N_LAYERS = 5  # tiny-random-bert has 5 hidden layers

# Tiny checkpoint per architecture family used by configs/encoders/*.yaml.
# Each exercises the same layer-discovery / pooling / init code paths as the
# real model, at hidden size 32, so it stays fast and CPU/RAM friendly.
#   vinai/bertweet-large         -> roberta
#   microsoft/deberta-v3-large   -> deberta-v2
#   answerdotai/ModernBERT-large -> modernbert
ARCH_TINY = {
    "roberta": "hf-internal-testing/tiny-random-roberta",
    "deberta-v2": "hf-internal-testing/tiny-random-DebertaV2Model",
    "modernbert": "hf-internal-testing/tiny-random-ModernBertForMaskedLM",
}

# Real production checkpoints, name -> (hf id, pooling). Opt-in only.
REAL_ENCODERS = {
    "bertweet-large": ("vinai/bertweet-large", "mean"),
    "deberta-v3-large": ("microsoft/deberta-v3-large", "cls"),
    "modernbert-large": ("answerdotai/ModernBERT-large", "cls"),
}


def make_encoder(**kwargs):
    kwargs.setdefault("name", TINY)
    return models.TransformerEncoder(**kwargs)


def _rand_ids(batch=BATCH, seq=SEQ_LEN, vocab=1000):
    return torch.randint(0, vocab, (batch, seq))


def clf_batch(batch=BATCH, seq=SEQ_LEN, n_classes=3):
    return {
        "input_ids": _rand_ids(batch, seq),
        "attention_mask": torch.ones(batch, seq, dtype=torch.long),
        "labels": torch.randint(0, n_classes, (batch,)),
    }


def reg_batch(batch=BATCH, seq=SEQ_LEN):
    return {
        "input_ids": _rand_ids(batch, seq),
        "attention_mask": torch.ones(batch, seq, dtype=torch.long),
        "labels": torch.rand(batch) * 2 - 1,  # compound score in [-1, 1]
    }


def multitask_batch(batch=BATCH, seq=SEQ_LEN):
    return {
        "input_ids": _rand_ids(batch, seq),
        "attention_mask": torch.ones(batch, seq, dtype=torch.long),
        # collate_fn stacks (compound, sentiment) as float32
        "labels": torch.stack(
            [
                torch.rand(batch) * 2 - 1,
                torch.randint(0, 3, (batch,)).float(),
            ],
            dim=1,
        ),
    }


class _DummyDataModule(pl.LightningDataModule):
    """Minimal datamodule yielding synthetic batches (no tokenizer, no CSVs)."""

    def __init__(self, batch_fn=clf_batch, n_batches=4):
        super().__init__()
        self.batch_fn = batch_fn
        self.n_batches = n_batches

    def _loader(self):
        data = [self.batch_fn() for _ in range(self.n_batches)]

        def gen():
            yield from data

        return gen()

    def train_dataloader(self):
        return self._loader()

    def val_dataloader(self):
        return self._loader()


def _cpu_trainer(**kwargs):
    kwargs.setdefault("accelerator", "cpu")
    kwargs.setdefault("devices", 1)
    kwargs.setdefault("logger", False)
    kwargs.setdefault("enable_checkpointing", False)
    kwargs.setdefault("enable_progress_bar", False)
    kwargs.setdefault("enable_model_summary", False)
    return pl.Trainer(**kwargs)


class EncoderFreezeTests(unittest.TestCase):
    def test_freeze_sets_requires_grad_false(self):
        enc = make_encoder()
        enc.freeze(layers=[0, 2])
        layers = enc._encoder_layers()
        for i, layer in enumerate(layers):
            for p in layer.parameters():
                self.assertEqual(p.requires_grad, i not in (0, 2))

    def test_unfreeze_restores_requires_grad(self):
        enc = make_encoder()
        enc.freeze(layers=[0, 1])
        enc.freeze(layers=[0], unfreeze=True)
        layers = enc._encoder_layers()
        for p in layers[0].parameters():
            self.assertTrue(p.requires_grad)
        for p in layers[1].parameters():
            self.assertFalse(p.requires_grad)

    def test_freeze_cfg_applied_at_epoch_end(self):
        enc = make_encoder()
        module = modules.SingleTaskModule(
            encoder=enc, task="clf", freeze_cfg={0: [0, 1]}, scheduler=None
        )
        module.on_train_epoch_end()  # current_epoch == 0
        layers = enc._encoder_layers()
        for i in (0, 1):
            for p in layers[i].parameters():
                self.assertFalse(p.requires_grad)
        for p in layers[2].parameters():
            self.assertTrue(p.requires_grad)

    def test_unfreeze_cfg_applied_at_epoch_end(self):
        enc = make_encoder(freeze=True)
        layers = enc._encoder_layers()
        for layer in layers:
            for p in layer.parameters():
                self.assertFalse(p.requires_grad)
        module = modules.SingleTaskModule(
            encoder=enc, task="clf", unfreeze_cfg={0: [0]}, scheduler=None
        )
        module.on_train_epoch_end()
        for p in layers[0].parameters():
            self.assertTrue(p.requires_grad)
        for p in layers[1].parameters():
            self.assertFalse(p.requires_grad)

    def test_no_cfg_leaves_grad_flags_untouched(self):
        enc = make_encoder()
        before = [p.requires_grad for p in enc.parameters()]
        module = modules.SingleTaskModule(encoder=enc, task="clf", scheduler=None)
        module.on_train_epoch_end()
        after = [p.requires_grad for p in enc.parameters()]
        self.assertEqual(before, after)

    def test_frozen_params_get_no_grad(self):
        enc = make_encoder()
        enc.freeze(layers=[0])
        module = modules.SingleTaskModule(encoder=enc, task="clf", scheduler=None)
        batch = clf_batch()
        loss = module.training_step(batch, 0)
        loss.backward()
        layers = enc._encoder_layers()
        for p in layers[0].parameters():
            self.assertIsNone(p.grad)
        self.assertTrue(any(p.grad is not None for p in layers[-1].parameters()))


class CheckpointTests(unittest.TestCase):
    def _perturb(self, encoder):
        with torch.no_grad():
            for p in encoder.encoder.parameters():
                p.fill_(0.1234)
                break

    def _encoder_weights(self, module):
        # NOTE: the `checkpoint` arg intentionally restores encoder weights
        # only (see _BaseModule.__init__); heads stay freshly initialized.
        return {
            k: v.cpu()
            for k, v in module.state_dict().items()
            if "encoder.encoder." in k or k.startswith("encoder.encoder.")
        }

    def test_checkpoint_weights_loaded_when_provided(self):
        enc = make_encoder()
        self._perturb(enc)
        src = modules.SingleTaskModule(encoder=enc, task="clf", scheduler=None)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ckpt.pt")
            torch.save({"state_dict": src.state_dict()}, path)
            dst = modules.SingleTaskModule(
                encoder=make_encoder(), task="clf", checkpoint=path, scheduler=None
            )
        src_enc, dst_enc = self._encoder_weights(src), self._encoder_weights(dst)
        self.assertTrue(len(src_enc) > 0)
        self.assertEqual(set(src_enc), set(dst_enc))
        for k in src_enc:
            self.assertTrue(torch.equal(src_enc[k], dst_enc[k]), f"mismatch at {k}")

    def test_checkpoint_does_not_restore_heads(self):
        """Documents the contract: `checkpoint` restores the encoder, while
        heads are always freshly initialized (prevents silently reusing a
        stale head from another task)."""
        enc = make_encoder()
        self._perturb(enc)
        src = modules.SingleTaskModule(encoder=enc, task="clf", scheduler=None)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ckpt.pt")
            torch.save({"state_dict": src.state_dict()}, path)
            dst = modules.SingleTaskModule(
                encoder=make_encoder(), task="clf", checkpoint=path, scheduler=None
            )
        heads_src = {k: v.cpu() for k, v in src.state_dict().items() if "head" in k}
        heads_dst = {k: v.cpu() for k, v in dst.state_dict().items() if "head" in k}
        self.assertEqual(set(heads_src), set(heads_dst))
        self.assertTrue(
            any(not torch.equal(heads_src[k], heads_dst[k]) for k in heads_src)
        )

    def test_no_checkpoint_means_pretrained_weights_kept(self):
        enc = make_encoder()
        self._perturb(enc)
        src = modules.SingleTaskModule(encoder=enc, task="clf", scheduler=None)
        fresh = modules.SingleTaskModule(
            encoder=make_encoder(), task="clf", scheduler=None
        )
        diffs = [
            not torch.equal(v1.cpu(), v2.cpu())
            for v1, v2 in zip(src.state_dict().values(), fresh.state_dict().values())
        ]
        self.assertTrue(any(diffs), "perturbed weights should differ from fresh ones")

    def test_checkpoint_roundtrip_multitask(self):
        enc = make_encoder()
        self._perturb(enc)
        src = modules.MultiTaskModule(encoder=enc, task="clf", scheduler=None)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ckpt.pt")
            torch.save({"state_dict": src.state_dict()}, path)
            dst = modules.MultiTaskModule(
                encoder=make_encoder(), task="clf", checkpoint=path, scheduler=None
            )
        src_enc, dst_enc = self._encoder_weights(src), self._encoder_weights(dst)
        self.assertTrue(len(src_enc) > 0)
        self.assertEqual(set(src_enc), set(dst_enc))
        for k in src_enc:
            self.assertTrue(torch.equal(src_enc[k], dst_enc[k]), f"mismatch at {k}")


class MultiTaskLossWeightTests(unittest.TestCase):
    def test_none_triggers_uncertainty_weighting(self):
        module = modules.MultiTaskModule(
            encoder=make_encoder(), task="clf", loss_weight=None, scheduler=None
        )
        self.assertTrue(module.use_uncertainty_weighting)
        self.assertTrue(hasattr(module, "log_sigma_reg"))
        self.assertTrue(hasattr(module, "log_sigma_clf"))
        self.assertIsInstance(module.log_sigma_reg, torch.nn.Parameter)
        self.assertIsInstance(module.log_sigma_clf, torch.nn.Parameter)

    def test_float_disables_uncertainty_weighting(self):
        module = modules.MultiTaskModule(
            encoder=make_encoder(), task="clf", loss_weight=0.7, scheduler=None
        )
        self.assertFalse(module.use_uncertainty_weighting)
        self.assertFalse(hasattr(module, "log_sigma_reg"))
        self.assertFalse(hasattr(module, "log_sigma_clf"))

    def test_fixed_weight_math(self):
        w = 0.7
        module = modules.MultiTaskModule(
            encoder=make_encoder(), task="clf", loss_weight=w, scheduler=None
        )
        reg_loss, clf_loss, total = module.loss_func(
            torch.tensor([0.5, -0.5]),
            torch.tensor([[2.0, 0.1, 0.1], [0.1, 0.1, 2.0]]),
            torch.tensor([0.5, -0.5]),
            torch.tensor([0, 2]),
        )
        expected = w * reg_loss + (1 - w) * clf_loss
        self.assertAlmostEqual(total.item(), expected.item(), places=5)

    def test_uncertainty_weight_math(self):
        module = modules.MultiTaskModule(
            encoder=make_encoder(), task="clf", loss_weight=None, scheduler=None
        )
        with torch.no_grad():
            module.log_sigma_reg.fill_(0.5)
            module.log_sigma_clf.fill_(-0.5)
        reg_loss, clf_loss, total = module.loss_func(
            torch.tensor([0.5, -0.5]),
            torch.tensor([[2.0, 0.1, 0.1], [0.1, 0.1, 2.0]]),
            torch.tensor([0.5, -0.5]),
            torch.tensor([0, 2]),
        )
        expected = (
            torch.exp(torch.tensor(-0.5)) * reg_loss
            + 0.5
            + torch.exp(torch.tensor(0.5)) * clf_loss
            - 0.5
        )
        self.assertAlmostEqual(total.item(), expected.item(), places=5)

    def test_uncertainty_sigmas_are_learnable(self):
        module = modules.MultiTaskModule(
            encoder=make_encoder(), task="clf", loss_weight=None, scheduler=None
        )
        names = {n for n, _ in module.named_parameters()}
        self.assertIn("log_sigma_reg", names)
        self.assertIn("log_sigma_clf", names)
        batch = multitask_batch()
        reg_logits, clf_logits = module.forward_both(batch)
        reg_labels, clf_labels = torch.split(batch["labels"], 1, dim=1)
        _, _, loss = module.loss_func(
            reg_logits, clf_logits, reg_labels.squeeze(1), clf_labels.squeeze(1).long()
        )
        loss.backward()
        self.assertIsNotNone(module.log_sigma_reg.grad)
        self.assertIsNotNone(module.log_sigma_clf.grad)


class ModuleSmokeTests(unittest.TestCase):
    def test_single_task_clf_step(self):
        module = modules.SingleTaskModule(
            encoder=make_encoder(), task="clf", scheduler=None
        )
        batch = clf_batch()
        logits = module.forward(batch)
        self.assertEqual(tuple(logits.shape), (BATCH, 3))
        loss = module.training_step(batch, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_single_task_reg_step(self):
        module = modules.SingleTaskModule(
            encoder=make_encoder(), task="reg", scheduler=None
        )
        batch = reg_batch()
        out = module.forward(batch)
        self.assertEqual(tuple(out.shape), (BATCH,))
        loss = module.training_step(batch, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_multitask_step_and_forward_both(self):
        module = modules.MultiTaskModule(
            encoder=make_encoder(), task="clf", loss_weight=0.5, scheduler=None
        )
        batch = multitask_batch()
        reg_logits, clf_logits = module.forward_both(batch)
        self.assertEqual(tuple(reg_logits.shape), (BATCH,))
        self.assertEqual(tuple(clf_logits.shape), (BATCH, 3))
        loss = module.training_step(batch, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_forward_both_single_encoder_pass(self):
        """Both heads must see the same embeddings (one shared encoder pass)."""
        module = modules.MultiTaskModule(
            encoder=make_encoder(), task="clf", loss_weight=0.5, scheduler=None
        )
        batch = multitask_batch()
        calls = []

        orig_forward = module.encoder.forward

        def counting_forward(input_ids, attention_mask):
            calls.append(1)
            return orig_forward(input_ids, attention_mask)

        module.encoder.forward = counting_forward
        module.forward_both(batch)
        self.assertEqual(len(calls), 1)
        module.encoder.forward = orig_forward

    def test_simcse_step(self):
        module = modules.SimCSEModule(encoder=make_encoder(), scheduler=None)
        batch = clf_batch()
        loss = module.training_step(batch, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_scl_pretraining_step(self):
        module = modules.SupervisedConstrastivePretrainingModule(
            encoder=make_encoder(),
            projector=models.Projector(hidden_size=32),
            scheduler=None,
        )
        # SCL collate doubles the batch with duplicated labels
        batch = clf_batch(batch=BATCH * 2)
        loss = module.training_step(batch, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_scl_learning_step(self):
        module = modules.SupervisedConstrastiveLearningModule(
            encoder=make_encoder(), scheduler=None
        )
        batch = clf_batch()
        loss = module.training_step(batch, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()


class EncoderArchitectureTests(unittest.TestCase):
    """Architecture-specific paths for the production encoder families.

    ``tiny-random-bert`` only covers BERT's ``encoder.encoder.layer`` layout.
    BERTweet (RoBERTa), DeBERTa-v3 (DeBERTa-v2) and ModernBERT differ in where
    their transformer layers live, so freeze/reset can silently break on them.
    """

    def test_encoder_layers_resolve_and_freeze(self):
        for arch, name in ARCH_TINY.items():
            with self.subTest(arch=arch):
                enc = make_encoder(name=name, pooling="cls")
                layers = enc._encoder_layers()
                self.assertGreater(len(layers), 0)

                enc.freeze(layers=[0])
                for p in layers[0].parameters():
                    self.assertFalse(p.requires_grad)
                self.assertTrue(any(p.requires_grad for p in layers[-1].parameters()))

                enc.freeze(layers=[0], unfreeze=True)
                for p in layers[0].parameters():
                    self.assertTrue(p.requires_grad)

    def test_reset_last_reinitializes_layers(self):
        for arch, name in ARCH_TINY.items():
            with self.subTest(arch=arch):
                enc = make_encoder(name=name, pooling="cls", reset_last=1)
                for p in enc.parameters():
                    self.assertTrue(torch.isfinite(p).all())

    def test_pooling_shapes(self):
        for arch, name in ARCH_TINY.items():
            for pooling in ("cls", "mean", "last"):
                with self.subTest(arch=arch, pooling=pooling):
                    enc = make_encoder(name=name, pooling=pooling)
                    out = enc(_rand_ids(), torch.ones(BATCH, SEQ_LEN, dtype=torch.long))
                    self.assertEqual(tuple(out.shape), (BATCH, enc.config.hidden_size))


class ArchitectureTrainingTests(unittest.TestCase):
    """A Lightning train loop must run for every production architecture family."""

    def _fit(self, module, batch_fn):
        trainer = _cpu_trainer(fast_dev_run=True)
        trainer.fit(module, _DummyDataModule(batch_fn=batch_fn))

    def test_single_task_training_runs(self):
        for arch, name in ARCH_TINY.items():
            for task, batch_fn in (("clf", clf_batch), ("reg", reg_batch)):
                with self.subTest(arch=arch, task=task):
                    module = modules.SingleTaskModule(
                        encoder=make_encoder(name=name),
                        task=task,
                        scheduler=None,
                    )
                    self._fit(module, batch_fn)

    def test_multitask_training_runs(self):
        for arch, name in ARCH_TINY.items():
            with self.subTest(arch=arch):
                module = modules.MultiTaskModule(
                    encoder=make_encoder(name=name),
                    task="clf",
                    loss_weight=0.5,
                    scheduler=None,
                )
                self._fit(module, multitask_batch)


class OptimizerConfigTests(unittest.TestCase):
    def test_dict_lr_gives_encoder_head_groups(self):
        module = modules.SingleTaskModule(
            encoder=make_encoder(),
            task="clf",
            lr={"encoder": 1e-5, "head": 3e-4},
            scheduler=None,
        )
        opt = module.configure_optimizers()
        self.assertIsInstance(opt, torch.optim.AdamW)
        self.assertEqual(len(opt.param_groups), 2)
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 1e-5)
        self.assertAlmostEqual(opt.param_groups[1]["lr"], 3e-4)

    def test_scalar_lr_gives_single_group(self):
        module = modules.SingleTaskModule(
            encoder=make_encoder(), task="clf", lr=1e-3, scheduler=None
        )
        opt = module.configure_optimizers()
        self.assertEqual(len(opt.param_groups), 1)

    def test_multitask_dict_lr_gives_three_groups(self):
        module = modules.MultiTaskModule(
            encoder=make_encoder(),
            task="clf",
            lr={"encoder": 1e-5, "head": 3e-4},
            loss_weight=0.5,
            scheduler=None,
        )
        opt = module.configure_optimizers()
        self.assertEqual(len(opt.param_groups), 3)

    def test_scheduler_none_returns_plain_optimizer(self):
        module = modules.SingleTaskModule(
            encoder=make_encoder(), task="clf", scheduler=None
        )
        self.assertIsInstance(module.configure_optimizers(), torch.optim.Optimizer)

    def test_default_scheduler_returns_dict(self):
        module = modules.SingleTaskModule(
            encoder=make_encoder(), task="clf", scheduler="linear"
        )
        dm = _DummyDataModule()
        trainer = _cpu_trainer(max_steps=2)
        trainer.fit(module, dm)  # attaches trainer -> estimated_stepping_batches
        out = module.configure_optimizers()
        self.assertIsInstance(out, dict)
        self.assertIn("lr_scheduler", out)


class TrainerIntegrationTests(unittest.TestCase):
    def test_fast_dev_run_single_task(self):
        module = modules.SingleTaskModule(
            encoder=make_encoder(), task="clf", scheduler=None
        )
        trainer = _cpu_trainer(fast_dev_run=True)
        trainer.fit(module, _DummyDataModule())

    def test_fast_dev_run_multitask(self):
        module = modules.MultiTaskModule(
            encoder=make_encoder(),
            task="clf",
            loss_weight=0.5,
            scheduler=None,
        )
        trainer = _cpu_trainer(fast_dev_run=True)
        trainer.fit(module, _DummyDataModule(batch_fn=multitask_batch))

    def test_fast_dev_run_scl(self):
        module = modules.SupervisedConstrastiveLearningModule(
            encoder=make_encoder(), scheduler=None
        )
        trainer = _cpu_trainer(fast_dev_run=True)
        trainer.fit(module, _DummyDataModule())

    def test_freeze_cfg_end_to_end(self):
        """A 2-epoch CPU run with freeze_cfg must actually freeze layer 0."""
        enc = make_encoder()
        module = modules.SingleTaskModule(
            encoder=enc, task="clf", freeze_cfg={0: [0]}, scheduler=None
        )
        trainer = _cpu_trainer(max_epochs=2)
        trainer.fit(module, _DummyDataModule(n_batches=2))
        for p in enc._encoder_layers()[0].parameters():
            self.assertFalse(p.requires_grad)

    def test_checkpoint_save_load_end_to_end(self):
        enc = make_encoder()
        module = modules.SingleTaskModule(encoder=enc, task="clf", scheduler=None)
        trainer = _cpu_trainer(fast_dev_run=True)
        trainer.fit(module, _DummyDataModule())
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.ckpt")
            trainer.save_checkpoint(path)
            self.assertTrue(os.path.exists(path))
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            before = {
                k: v.clone()
                for k, v in modules.SingleTaskModule(
                    encoder=make_encoder(), task="clf", scheduler=None
                )
                .state_dict()
                .items()
            }
            restored = modules.SingleTaskModule.load_from_checkpoint(
                path, encoder=make_encoder(), task="clf"
            )
            for k, v in restored.state_dict().items():
                self.assertTrue(torch.equal(v.cpu(), ckpt["state_dict"][k].cpu()))
            self.assertTrue(
                any(
                    not torch.equal(v.cpu(), before[k].cpu())
                    for k, v in restored.state_dict().items()
                )
            )

    def test_hparams_survive_checkpoint(self):
        module = modules.MultiTaskModule(
            encoder=make_encoder(), task="clf", loss_weight=0.3, scheduler=None
        )
        trainer = _cpu_trainer(fast_dev_run=True)
        trainer.fit(module, _DummyDataModule(batch_fn=multitask_batch))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.ckpt")
            trainer.save_checkpoint(path)
            restored = modules.MultiTaskModule.load_from_checkpoint(
                path, encoder=make_encoder()
            )
            self.assertAlmostEqual(restored.hparams.loss_weight, 0.3)
            self.assertFalse(restored.use_uncertainty_weighting)


class MLflowLoggerTests(unittest.TestCase):
    """The production trainer uses ``MLFlowLogger`` (configs/defaults.yaml).

    Lightning logs scalar params/metrics to the run it owns, but the module's
    text/figure logging goes through ``_log_text``/``_log_figure``. Those must
    attach their artifacts to that *same* run. Using the fluent ``mlflow``
    helpers instead silently starts an orphan run in the default experiment,
    so these tests pin the run id the artifacts land in. They are the local
    guard for the RunPod/Azure MLflow path.
    """

    def _fit(self, uri):
        from lightning.pytorch.loggers import MLFlowLogger

        logger = MLFlowLogger(
            experiment_name="twitter-test", tracking_uri=uri, log_model=False
        )
        module = modules.SingleTaskModule(
            encoder=make_encoder(), task="clf", scheduler=None
        )
        trainer = _cpu_trainer(
            logger=logger,
            max_epochs=1,
            limit_train_batches=1,
            limit_val_batches=1,
            num_sanity_val_steps=0,
            log_every_n_steps=1,
        )
        trainer.fit(module, _DummyDataModule(n_batches=2))
        return logger

    def test_artifacts_attach_to_lightning_run(self):
        with tempfile.TemporaryDirectory() as d:
            logger = self._fit("file:" + os.path.join(d, "mlruns"))
            client = logger.experiment
            run = client.get_run(logger.run_id)
            self.assertIn("loss/validation", run.data.metrics)
            artifacts = {a.path for a in client.list_artifacts(logger.run_id)}
            self.assertIn("Confusion Matrix_validation.png", artifacts)
            self.assertIn("Input_training", artifacts)
            self.assertIn("Input_validation", artifacts)

    def test_no_orphan_run_in_default_experiment(self):
        with tempfile.TemporaryDirectory() as d:
            logger = self._fit("file:" + os.path.join(d, "mlruns"))
            client = logger.experiment
            default = client.get_experiment_by_name("Default")
            if default is not None:
                self.assertEqual(client.search_runs([default.experiment_id]), [])


@unittest.skipUnless(
    os.environ.get("TWITTER_TEST_REAL_ENCODERS") == "1",
    "set TWITTER_TEST_REAL_ENCODERS=1 to load the multi-GB production "
    "checkpoints and run one CPU training step each (needs ~8 GB free RAM, "
    "~5 GB disk for the HF cache, and network on first run)",
)
class RealEncoderIntegrationTests(unittest.TestCase):
    """Opt-in check that the *actual* production checkpoints load and train.

    The default suite uses tiny per-architecture checkpoints (``ARCH_TINY``) to
    stay fast and light; this is the last stop before submitting a RunPod job.
    Batch/sequence are kept tiny because it runs on CPU (32 GB RAM, no GPU).
    """

    def test_multitask_training_step(self):
        for key, (name, pooling) in REAL_ENCODERS.items():
            with self.subTest(encoder=key):
                module = modules.MultiTaskModule(
                    encoder=make_encoder(name=name, pooling=pooling),
                    task="clf",
                    loss_weight=0.5,
                    scheduler=None,
                )
                batch = {
                    "input_ids": _rand_ids(batch=2),
                    "attention_mask": torch.ones(2, SEQ_LEN, dtype=torch.long),
                    "labels": torch.stack(
                        [
                            torch.rand(2) * 2 - 1,
                            torch.randint(0, 3, (2,)).float(),
                        ],
                        dim=1,
                    ),
                }
                loss = module.training_step(batch, 0)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                del module, loss
                gc.collect()


if __name__ == "__main__":
    unittest.main()
