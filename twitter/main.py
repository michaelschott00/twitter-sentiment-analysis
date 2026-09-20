import os

from lightning.pytorch.cli import LightningCLI, SaveConfigCallback


class _LogDirFallbackConfigCallback(SaveConfigCallback):
    """Persist the resolved CLI config next to non-local loggers.

    ``trainer.log_dir`` is ``None`` whenever the only logger has no filesystem
    ``save_dir`` -- notably Lightning's ``MLFlowLogger`` with an ``azureml://``
    (or any non-``file:``) tracking URI, which is exactly what the RunPod/AML
    runs use. ``SaveConfigCallback.setup`` asserts ``log_dir is not None`` and
    crashes before training starts. For those runs the config is logged to the
    MLflow run instead (mirroring ``twitter.modules`` artifact logging) so the
    run stays reproducible without a local directory to write into.
    """

    def setup(self, trainer, pl_module, stage):
        if trainer.log_dir is not None or not self.save_to_log_dir:
            return super().setup(trainer, pl_module, stage)

        if getattr(self, "already_saved", False):
            return
        if trainer.is_global_zero:
            logger = trainer.loggers[0] if trainer.loggers else None
            experiment = getattr(logger, "experiment", None)
            run_id = getattr(logger, "run_id", None)
            if experiment is None or run_id is None:
                return
            import tempfile

            with tempfile.TemporaryDirectory() as tmp_dir:
                config_path = os.path.join(tmp_dir, self.config_filename)
                self.parser.save(
                    self.config,
                    config_path,
                    skip_none=False,
                    overwrite=self.overwrite,
                    multifile=self.multifile,
                )
                experiment.log_artifact(run_id, config_path)
            self.already_saved = True

        self.already_saved = trainer.strategy.broadcast(self.already_saved)


class MyLightningCLI(LightningCLI):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("save_config_callback", _LogDirFallbackConfigCallback)
        super().__init__(*args, **kwargs)

    def add_arguments_to_parser(self, parser):
        parser.link_arguments(
            "model.init_args.encoder.init_args.name", "data.init_args.encoder_name"
        )


def cli_main():
    cli = MyLightningCLI(
        parser_kwargs={
            "fit": {
                "default_config_files": [
                    "configs/defaults.yaml",
                    "configs/data.yaml",
                ]
            },
            "parser_mode": "omegaconf",
        }
    )


if __name__ == "__main__":
    cli_main()
