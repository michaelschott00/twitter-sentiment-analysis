"""Run FastText language ID over data/splits and report per-language counts."""

import os
import urllib.request

import click
import pandas as pd
from tqdm import tqdm

MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
MODEL_FILENAME = "lid.176.bin"
SPLITS = ("train", "dev", "test_1", "test_2")


def _model_path(model_dir: str) -> str:
    return os.path.join(os.path.expanduser(model_dir), MODEL_FILENAME)


def _ensure_model(model_dir: str) -> str:
    path = _model_path(model_dir)
    if os.path.exists(path):
        return path
    os.makedirs(os.path.expanduser(model_dir), exist_ok=True)
    click.echo(f"Downloading FastText model to {path} ...")
    urllib.request.urlretrieve(MODEL_URL, path)
    click.echo("Download complete.")
    return path


def _clean(text) -> str | None:
    if not isinstance(text, str):
        return None
    text = " ".join(text.split())
    return text or None


@click.command()
@click.option("--splits-dir", default="data/splits", help="Directory with tweets_*.csv")
@click.option(
    "--save-dir",
    default="data/splits",
    help="Directory for *_lang.csv and language_counts.csv outputs",
)
@click.option(
    "--model-dir",
    default="~/.cache/fasttext",
    help="Directory to cache lid.176.bin (downloaded once if missing)",
)
@click.option("--text-column", default="text", help="Column holding tweet text")
@click.option(
    "--min-confidence", default=0.0, type=float, help="Below this conf. -> 'uncertain'"
)
def main(splits_dir, save_dir, model_dir, text_column, min_confidence):
    import fasttext
    import numpy as np

    # fasttext's predict() calls np.array(probs, copy=False), which raises
    # with NumPy 2.x. Map copy=False -> asarray (no behavior change on 1.x).
    _orig_array = np.array

    def _compat_array(*args, **kwargs):
        if kwargs.pop("copy", None) is False:
            return np.asarray(*args, **kwargs)
        return _orig_array(*args, **kwargs)

    np.array = _compat_array

    model_path = _ensure_model(model_dir)
    model = fasttext.load_model(model_path)
    os.makedirs(save_dir, exist_ok=True)

    all_frames = []
    for split in SPLITS:
        src = os.path.join(splits_dir, f"tweets_{split}.csv")
        if not os.path.exists(src):
            click.echo(f"Skipping missing {src}")
            continue
        df = pd.read_csv(src)
        if text_column not in df.columns:
            raise click.ClickException(f"{text_column!r} not in {src}")

        langs, confs = [], []
        for text in tqdm(df[text_column], desc=f"classifying {split}"):
            cleaned = _clean(text)
            if cleaned is None:
                langs.append("unknown")
                confs.append(0.0)
                continue
            labels, probs = model.predict(cleaned, k=1)
            lang = labels[0].removeprefix("__label__")
            conf = float(probs[0])
            if conf < min_confidence:
                lang = "uncertain"
            langs.append(lang)
            confs.append(conf)

        df["pred_lang"] = langs
        df["pred_conf"] = confs
        out = os.path.join(save_dir, f"tweets_{split}_lang.csv")
        df.to_csv(out, index=False)

        counts = (
            df.groupby("pred_lang").size().reset_index(name="count").assign(split=split)
        )
        counts["pct"] = counts["count"] / len(df) * 100
        all_frames.append(counts)

        click.echo(
            f"\n--- {split} (n={len(df)}, langs={counts['pred_lang'].nunique()}) ---"
        )
        click.echo(counts.sort_values("count", ascending=False).to_string(index=False))

    if not all_frames:
        raise click.ClickException(f"No splits found in {splits_dir}")
    summary = pd.concat(all_frames, ignore_index=True)
    summary_path = os.path.join(save_dir, "language_counts.csv")
    summary.to_csv(summary_path, index=False)
    click.echo(f"\nWrote per-split *_lang.csv + {summary_path}")


if __name__ == "__main__":
    main()
