from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from twitter.modules import (
    SingleTaskModule,
    MultiTaskModule
)
from twitter.models import (
    TransformerEncoder,
    TransformerRegressor,
    TransformerClassifier
)
from twitter.data import TwitterDataModule


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("model.init_args.encoder.init_args.name", "data.init_args.encoder_name")


def cli_main():
    cli = MyLightningCLI(parser_kwargs={
        "fit": {"default_config_files": [
            "configs/defaults.yaml",
            "configs/data.yaml",
        ]},
        "parser_mode": "omegaconf"
    })


if __name__ == '__main__':
    cli_main()
