"""LightGBM non-deep baseline for Twitter sentiment analysis.

Uses TF-IDF features (no metadata) with tunable ngram_range.
Applies SMOTE to TF-IDF for classification to address class imbalance.
Evaluates on holdout dev set: macro F1 for classification, RMSE for regression.
"""

import os

import click
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, mean_squared_error

try:
    from sklearn.metrics import root_mean_squared_error

    _has_rmse = True
except ImportError:
    _has_rmse = False

import lightgbm as lgb

from twitter.labels import LABEL_CODING


def _load_data(train_path: str, dev_path: str):
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not os.path.exists(dev_path):
        raise FileNotFoundError(f"Dev file not found: {dev_path}")
    df_train = pd.read_csv(train_path)
    df_dev = pd.read_csv(dev_path)
    return df_train, df_dev


def _encode_labels(series: pd.Series) -> np.ndarray:
    # handle both string labels and already-encoded ints
    if series.dtype == object or series.dtype.name == "category":
        # try mapping, fallback to astype
        mapped = series.map(LABEL_CODING)
        # if mapping produced NaN for numeric strings, try direct
        if mapped.isna().any():
            # try converting to int directly
            return series.astype(int).to_numpy()
        return mapped.to_numpy(dtype=int)
    return series.to_numpy(dtype=int)


def _compute_rmse(y_true, y_pred) -> float:
    if _has_rmse:
        return root_mean_squared_error(y_true, y_pred)
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


@click.command()
@click.option(
    "--train-path", default="data/splits/tweets_train.csv", help="Path to train CSV"
)
@click.option(
    "--dev-path", default="data/splits/tweets_dev.csv", help="Path to dev CSV"
)
@click.option("--ngram-min", default=1, type=int, help="Min n for TF-IDF ngram_range")
@click.option("--ngram-max", default=2, type=int, help="Max n for TF-IDF ngram_range")
@click.option(
    "--max-features",
    default=10000,
    type=int,
    help="Max TF-IDF features (0 for no limit)",
)
@click.option(
    "--use-smote/--no-smote",
    default=True,
    help="Apply SMOTE to TF-IDF for classification",
)
@click.option("--smote-k", default=5, type=int, help="k_neighbors for SMOTE")
@click.option("--random-state", default=42, type=int, help="Random seed")
@click.option(
    "--task",
    default="both",
    type=click.Choice(["clf", "reg", "both"], case_sensitive=False),
    help="Task to train: clf, reg or both",
)
@click.option(
    "--output-dir", default=None, type=str, help="Directory to save model artifacts"
)
@click.option("--n-estimators", default=100, type=int, help="LightGBM n_estimators")
@click.option("--learning-rate", default=0.1, type=float, help="LightGBM learning_rate")
@click.option("--num-leaves", default=31, type=int, help="LightGBM num_leaves")
def main(
    train_path,
    dev_path,
    ngram_min,
    ngram_max,
    max_features,
    use_smote,
    smote_k,
    random_state,
    task,
    output_dir,
    n_estimators,
    learning_rate,
    num_leaves,
):
    """Train LightGBM baseline on TF-IDF features and evaluate on holdout dev set."""
    if ngram_min < 1 or ngram_max < 1:
        raise click.BadParameter("ngram_min and ngram_max must be >=1")
    if ngram_min > ngram_max:
        raise click.BadParameter("ngram_min must be <= ngram_max")

    ngram_range = (ngram_min, ngram_max)
    max_features_val = None if max_features == 0 else max_features

    click.echo(f"Loading data from {train_path} and {dev_path}")
    df_train, df_dev = _load_data(train_path, dev_path)

    # ensure text column exists
    if "text" not in df_train.columns:
        raise ValueError("Train CSV missing 'text' column")
    if "text" not in df_dev.columns:
        raise ValueError("Dev CSV missing 'text' column")

    X_train_text = df_train["text"].astype(str).fillna("")
    X_dev_text = df_dev["text"].astype(str).fillna("")

    # TF-IDF vectorizer (text only, no metadata)
    click.echo(
        f"Fitting TfidfVectorizer ngram_range={ngram_range} max_features={max_features_val}"
    )
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features_val,
        lowercase=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train_text)
    X_dev_vec = vectorizer.transform(X_dev_text)
    click.echo(f"TF-IDF shapes: train {X_train_vec.shape}, dev {X_dev_vec.shape}")

    results = {}

    # Classification
    if task in ("clf", "both"):
        if "sentiment" not in df_train.columns:
            click.echo("Skipping clf: sentiment column missing")
        else:
            y_train_clf = _encode_labels(df_train["sentiment"])
            y_dev_clf = _encode_labels(df_dev["sentiment"])
            click.echo(f"Class distribution train: {np.bincount(y_train_clf)}")
            click.echo(f"Class distribution dev: {np.bincount(y_dev_clf)}")

            X_clf_train = X_train_vec
            y_clf_train = y_train_clf

            if use_smote:
                click.echo(f"Applying SMOTE k={smote_k} to address imbalance")
                # SMOTE does not support sparse; densify
                # for memory, check size
                n_samples, n_features = X_clf_train.shape
                dense_size = n_samples * n_features
                # warn if > 200M entries (~1.6GB for float64)
                if dense_size > 200_000_000:
                    click.echo(
                        f"Warning: dense TF-IDF would be {dense_size} entries, may be heavy. "
                        "Consider reducing max_features."
                    )
                # convert sparse to dense
                if hasattr(X_clf_train, "toarray"):
                    X_dense = X_clf_train.toarray()
                else:
                    X_dense = np.asarray(X_clf_train)

                # adjust k_neighbors if minority class too small
                _, counts = np.unique(y_clf_train, return_counts=True)
                min_count = counts.min()
                k = min(smote_k, max(1, min_count - 1))
                if k != smote_k:
                    click.echo(
                        f"Adjusting SMOTE k_neighbors from {smote_k} to {k} (min_count={min_count})"
                    )
                smote = SMOTE(random_state=random_state, k_neighbors=k)
                X_clf_train, y_clf_train = smote.fit_resample(X_dense, y_clf_train)
                click.echo(
                    f"After SMOTE: {X_clf_train.shape}, dist {np.bincount(y_clf_train)}"
                )
                # keep dev as sparse; LightGBM can handle dense train + sparse dev via consistent transform?
                # For dev we keep sparse but convert to dense for prediction consistency if train was dense?
                # LightGBM handles both, but we convert dev to dense array for consistent input type
                # Actually LightGBM handles sparse and dense separately; dense train + sparse dev is okay
                # but to be safe, keep dev as dense array of same dtype if needed
                # We leave X_dev_vec as is (sparse) and let predict handle it
            else:
                click.echo("SMOTE disabled")

            clf = lgb.LGBMClassifier(
                random_state=random_state,
                verbosity=-1,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
            )
            click.echo("Training LGBMClassifier...")
            clf.fit(X_clf_train, y_clf_train)

            y_pred = clf.predict(X_dev_vec)
            macro_f1 = f1_score(y_dev_clf, y_pred, average="macro")
            click.echo(f"Classification Macro F1 (dev): {macro_f1:.4f}")
            results["macro_f1"] = macro_f1

            # also show per-class report optionally
            from sklearn.metrics import classification_report

            click.echo(
                classification_report(y_dev_clf, y_pred, digits=4, zero_division=0)
            )

            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                joblib.dump(clf, os.path.join(output_dir, "lgbm_classifier.joblib"))
                joblib.dump(
                    vectorizer, os.path.join(output_dir, "tfidf_vectorizer.joblib")
                )
                click.echo(f"Saved classifier and vectorizer to {output_dir}")

    # Regression
    if task in ("reg", "both"):
        if "score_compound" not in df_train.columns:
            click.echo("Skipping reg: score_compound column missing")
        else:
            y_train_reg = df_train["score_compound"].to_numpy(dtype=float)
            y_dev_reg = df_dev["score_compound"].to_numpy(dtype=float)

            reg = lgb.LGBMRegressor(
                random_state=random_state,
                verbosity=-1,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
            )
            click.echo("Training LGBMRegressor...")
            reg.fit(X_train_vec, y_train_reg)

            y_pred_reg = reg.predict(X_dev_vec)
            rmse = _compute_rmse(y_dev_reg, y_pred_reg)
            click.echo(f"Regression RMSE (dev): {rmse:.4f}")
            results["rmse"] = rmse

            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                # if clf already saved vectorizer, don't overwrite unnecessarily but okay
                if task == "reg":
                    joblib.dump(
                        vectorizer, os.path.join(output_dir, "tfidf_vectorizer.joblib")
                    )
                joblib.dump(reg, os.path.join(output_dir, "lgbm_regressor.joblib"))
                click.echo(f"Saved regressor to {output_dir}")

    # final summary
    click.echo(f"Done. Results: {results}")
    return results


if __name__ == "__main__":
    main()
