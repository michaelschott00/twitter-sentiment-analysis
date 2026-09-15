"""Phase 2 validation: TwitterDataModule against the Azure-uploaded split layout.

The blob containers (`raw`/`splits`) mirror local `data/` file-for-file, and the
AML Data Assets (`twitter-raw:1`, `twitter-splits:1`) point at the `ds_raw` /
`ds_splits` datastores. An Azure ML job mounts the asset at
`${{inputs.splits}}` and passes it as `data.init_args.root_dir`, so validating
locally with `root_dir=data/splits` proves the same code path works on the
mounted path.

Run with: python infra/scripts/validate_data.py [--root-dir data/splits]
"""

import click

from twitter.data import TwitterDataModule

EXPECTED_FILES = {
    "tweets_train.csv",
    "tweets_dev.csv",
    "tweets_test_1.csv",
    "tweets_test_2.csv",
}
EXPECTED_COUNTS = {"train": 6400, "dev": 1600, "test_1": 1000, "test_2": 1000}


@click.command()
@click.option(
    "--root-dir",
    default="data/splits",
    help="Split directory (local mirror of the azureml-mounted asset).",
)
@click.option(
    "--encoder-name",
    default="sentence-transformers/all-MiniLM-L6-v2",
    help="HF encoder used only for its tokenizer.",
)
@click.option("--batch-size", default=8, type=int, help="Batch size for smoketest.")
def main(root_dir: str, encoder_name: str, batch_size: int) -> None:
    import os

    files = set(os.listdir(root_dir))
    assert EXPECTED_FILES <= files, (
        f"missing splits in {root_dir}: {EXPECTED_FILES - files}"
    )
    click.echo(f"files ok: {sorted(files)}")

    for labels in ("clf", "reg", "both"):
        dm = TwitterDataModule(
            root_dir=root_dir,
            features="text",
            labels=labels,  # type: ignore[arg-type]
            encoder_name=encoder_name,
            batch_size=batch_size,
        )
        dm.setup("fit")
        assert len(dm.twitter_train) == EXPECTED_COUNTS["train"], (
            labels,
            len(dm.twitter_train),
        )
        assert len(dm.twitter_dev) == EXPECTED_COUNTS["dev"], (
            labels,
            len(dm.twitter_dev),
        )
        batch = next(iter(dm.train_dataloader()))
        assert "input_ids" in batch and "labels" in batch
        click.echo(
            f"{labels}: train={len(dm.twitter_train)} dev={len(dm.twitter_dev)} batch={tuple(batch['input_ids'].shape)} ok"
        )

    dm = TwitterDataModule(
        root_dir=root_dir,
        features="text",
        labels="none",  # type: ignore[arg-type]
        encoder_name=encoder_name,
        batch_size=batch_size,
    )
    dm.setup("predict")
    assert len(dm.twitter_test_1) == EXPECTED_COUNTS["test_1"]
    assert len(dm.twitter_test_2) == EXPECTED_COUNTS["test_2"]
    click.echo(
        f"predict: test_1={len(dm.twitter_test_1)} test_2={len(dm.twitter_test_2)} ok"
    )
    click.echo("PHASE2 VALIDATION OK")


if __name__ == "__main__":
    main()
