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


# FastText lid.176 language codes -> full English names.
LANG_NAMES = {
    "af": "Afrikaans",
    "als": "Albanian",
    "am": "Amharic",
    "an": "Aragonese",
    "ar": "Arabic",
    "arz": "Egyptian Arabic",
    "as": "Assamese",
    "ast": "Asturian",
    "av": "Avaric",
    "ay": "Aymara",
    "az": "Azerbaijani",
    "azb": "South Azerbaijani",
    "ba": "Bashkir",
    "bar": "Bavarian",
    "bcl": "Central Bikol",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "bh": "Bihari",
    "bn": "Bengali",
    "bo": "Tibetan",
    "bpy": "Bishnupriya",
    "br": "Breton",
    "bs": "Bosnian",
    "bxr": "Buryat",
    "ca": "Catalan",
    "cbk": "Chavacano",
    "ce": "Chechen",
    "ceb": "Cebuano",
    "ckb": "Central Kurdish",
    "co": "Corsican",
    "cs": "Czech",
    "cv": "Chuvash",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "diq": "Zazaki",
    "dsb": "Lower Sorbian",
    "dv": "Divehi",
    "el": "Greek",
    "eml": "Emiliano-Romagnolo",
    "en": "English",
    "eo": "Esperanto",
    "es": "Spanish",
    "et": "Estonian",
    "eu": "Basque",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "frr": "Northern Frisian",
    "fy": "Western Frisian",
    "ga": "Irish",
    "gd": "Scottish Gaelic",
    "gl": "Galician",
    "gn": "Guarani",
    "gom": "Goan Konkani",
    "gu": "Gujarati",
    "gv": "Manx",
    "he": "Hebrew",
    "hi": "Hindi",
    "hif": "Fiji Hindi",
    "hr": "Croatian",
    "hsb": "Upper Sorbian",
    "ht": "Haitian",
    "hu": "Hungarian",
    "hy": "Armenian",
    "ia": "Interlingua",
    "id": "Indonesian",
    "ie": "Interlingue",
    "ilo": "Iloko",
    "io": "Ido",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "jbo": "Lojban",
    "jv": "Javanese",
    "ka": "Georgian",
    "kk": "Kazakh",
    "km": "Khmer",
    "kn": "Kannada",
    "ko": "Korean",
    "krc": "Karachay-Balkar",
    "ku": "Kurdish",
    "kv": "Komi",
    "kw": "Cornish",
    "ky": "Kyrgyz",
    "la": "Latin",
    "lb": "Luxembourgish",
    "lez": "Lezgian",
    "li": "Limburgish",
    "lmo": "Lombard",
    "lo": "Lao",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mai": "Maithili",
    "mg": "Malagasy",
    "mhr": "Eastern Mari",
    "min": "Minangkabau",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mn": "Mongolian",
    "mr": "Marathi",
    "mrj": "Western Mari",
    "ms": "Malay",
    "mt": "Maltese",
    "mwl": "Mirandese",
    "my": "Burmese",
    "myv": "Erzya",
    "mzn": "Mazanderani",
    "nah": "Nahuatl",
    "nap": "Neapolitan",
    "nds": "Low German",
    "ne": "Nepali",
    "new": "Newari",
    "nl": "Dutch",
    "nn": "Norwegian Nynorsk",
    "no": "Norwegian",
    "oc": "Occitan",
    "or": "Oriya",
    "os": "Ossetian",
    "pa": "Punjabi",
    "pam": "Pampanga",
    "pfl": "Palatine German",
    "pl": "Polish",
    "pms": "Piedmontese",
    "pnb": "Western Punjabi",
    "ps": "Pashto",
    "pt": "Portuguese",
    "qu": "Quechua",
    "rm": "Romansh",
    "ro": "Romanian",
    "ru": "Russian",
    "rue": "Rusyn",
    "sa": "Sanskrit",
    "sah": "Yakut",
    "sc": "Sardinian",
    "scn": "Sicilian",
    "sco": "Scots",
    "sd": "Sindhi",
    "sh": "Serbo-Croatian",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovenian",
    "so": "Somali",
    "sq": "Albanian",
    "sr": "Serbian",
    "su": "Sundanese",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "tg": "Tajik",
    "th": "Thai",
    "tk": "Turkmen",
    "tl": "Tagalog",
    "tr": "Turkish",
    "tt": "Tatar",
    "tyv": "Tuvinian",
    "ug": "Uyghur",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "vec": "Venetian",
    "vi": "Vietnamese",
    "vo": "Volapük",
    "wa": "Walloon",
    "war": "Waray",
    "wuu": "Wu Chinese",
    "xal": "Kalmyk",
    "xmf": "Mingrelian",
    "yi": "Yiddish",
    "yo": "Yoruba",
    "yue": "Cantonese",
    "zh": "Chinese",
    "unknown": "Unknown",
    "uncertain": "Uncertain",
}


def _lang_name(code: str) -> str:
    return LANG_NAMES.get(code, code.capitalize())


def _split_name(split: str) -> str:
    return split.replace("_", " ").title()


@click.command()
@click.option("--splits-dir", default="data/splits", help="Directory with tweets_*.csv")
@click.option(
    "--save-dir",
    default=None,
    help="Directory for language_counts.csv output (if unset, only print results)",
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
@click.option(
    "--plot-path",
    default=None,
    type=click.Path(dir_okay=False),
    help="Save language distribution bar plot as PNG here (if unset, only show it)",
)
def main(splits_dir, save_dir, model_dir, text_column, min_confidence, plot_path):
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
    if save_dir:
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

        langs = []
        for text in tqdm(df[text_column], desc=f"classifying {split}"):
            cleaned = _clean(text)
            if cleaned is None:
                langs.append("unknown")
                continue
            labels, probs = model.predict(cleaned, k=1)
            lang = labels[0].removeprefix("__label__")
            conf = float(probs[0])
            if conf < min_confidence:
                lang = "uncertain"
            langs.append(lang)

        df["pred_lang"] = langs

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
    if save_dir:
        summary_path = os.path.join(save_dir, "language_counts.csv")
        summary.to_csv(summary_path, index=False)
        click.echo(f"\nWrote {summary_path}")

    import matplotlib.pyplot as plt
    import seaborn as sns

    order = (
        summary.groupby("pred_lang")["count"].sum().sort_values(ascending=False).index
    )
    order_names = [_lang_name(code) for code in order]
    plot_df = summary.assign(
        Language=summary["pred_lang"].map(_lang_name),
        Split=summary["split"].map(_split_name),
    )
    binary = (
        plot_df.assign(
            Language=plot_df["Language"].map(
                lambda lang: "English" if lang == "English" else "Non-English"
            )
        )
        .groupby(["Split", "Language"], as_index=False)[["count", "pct"]]
        .sum()
    )
    fig, (ax_lang, ax_binary) = plt.subplots(
        1, 2, figsize=(max(8, len(order) * 0.6) + 6, 6)
    )
    sns.barplot(
        data=plot_df, x="Language", y="pct", hue="Split", order=order_names, ax=ax_lang
    )
    ax_lang.tick_params(axis="x", rotation=45)
    ax_lang.set_xlabel("Language")
    ax_lang.set_ylabel("Percentage (%)")
    ax_lang.set_title("Language Distribution per Split")
    sns.barplot(
        data=binary,
        x="Language",
        y="pct",
        hue="Split",
        order=["English", "Non-English"],
        ax=ax_binary,
    )
    ax_binary.set_xlabel("Language Group")
    ax_binary.set_ylabel("Percentage (%)")
    ax_binary.set_title("English vs. Non-English per Split")
    fig.tight_layout()
    if plot_path:
        parent = os.path.dirname(os.path.abspath(plot_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        plt.savefig(plot_path, dpi=150)
        click.echo(f"Wrote {plot_path}")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
