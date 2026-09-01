"""Split into train-, development- and testset in a reproducible way."""
import os

import click
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


@click.command()
@click.option("--config", default="configs/split.yaml", help="Path to split config.yaml")
@click.option("--root", default="data", help="Root data directory")
@click.option("--raw", default="raw", help="Raw data subdirectory")
@click.option("--splits", default="splits", help="Splits subdirectory")
@click.option("--seed", default=None, type=int, help="Override seed from config")
@click.option("--test-size", default=None, type=float, help="Override test_size from config")
def main(config, root, raw, splits, seed, test_size):
    # load config with defaults if missing
    if os.path.exists(config):
        with open(config) as f:
            cfg = yaml.safe_load(f)
        seed = seed if seed is not None else cfg.get("split", {}).get("seed", 42)
        test_size = test_size if test_size is not None else cfg.get("split", {}).get("test_size", 0.2)
    else:
        seed = seed if seed is not None else 42
        test_size = test_size if test_size is not None else 0.2

    splits_dir = os.path.join(root, splits)
    os.makedirs(splits_dir, exist_ok=True)

    raw_path = os.path.join(root, raw, "tweets_train.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data not found at {raw_path}. Place tweets_train.csv there.")

    df = pd.read_csv(raw_path)
    df_train, df_dev = train_test_split(
        df, test_size=test_size, random_state=seed, shuffle=True, stratify=df["sentiment"]
    )

    df_train.to_csv(os.path.join(splits_dir, "tweets_train.csv"), index=False)
    df_dev.to_csv(os.path.join(splits_dir, "tweets_dev.csv"), index=False)
    click.echo(f"Created splits: {len(df_train)} train / {len(df_dev)} dev (seed={seed}, test_size={test_size})")


if __name__ == "__main__":
    main()
