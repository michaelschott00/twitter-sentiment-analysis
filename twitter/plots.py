"""Compare MLflow runs by their main eval metrics (F1 for clf, RMSE for reg).

Queries runs via the MLflow SDK (see
https://learn.microsoft.com/en-us/azure/machine-learning/how-to-track-experiments-mlflow?view=azureml-api-2#get-metrics-parameters-artifacts-and-models),
lets the user pick multiple runs, then plots a bar chart comparison.

Metric names logged in this repo:
- Lightning transformer runs (``twitter/modules.py`` + ``twitter/models.py``):
  ``MulticlassF1Score`` (macro F1, classification) and ``MeanSquaredError``
  (torchmetrics with ``squared=False``, i.e. RMSE, regression).
- Baselines (``twitter/baselines/*_baseline.py``): ``macro_f1`` / ``rmse``.

Usage:
    python -m twitter.plots list-runs --experiment-name twitter-sentiment
    python -m twitter.plots compare --experiment-name twitter-sentiment
    python -m twitter.plots compare --experiment-name twitter-sentiment \\
        --run-ids abc123,def456 --output comparison.png
    python -m twitter.plots compare --experiment-name twitter-sentiment \\
        --table --output comparison.csv
"""

from __future__ import annotations

import os
import re

import click

# Candidate metric keys (in priority order) for each task.
F1_CANDIDATES = [
    "MulticlassF1Score",
    "macro_f1",
    "MacroF1",
    "F1",
    "f1_score",
    "f1",
]
RMSE_CANDIDATES = [
    "MeanSquaredError",
    "rmse",
    "RMSE",
]

# Presentable display names keyed by the MLflow run name (case-insensitive).
MODEL_DISPLAY_NAMES = {
    "clf-bertweet-large": "BERTweet-large",
    "clf-deberta-v3-large": "DeBERTa-v3-large",
    "clf-modernbert-large": "ModernBERT-large",
    "clf-sentence-bert-base": "Sentence-BERT-base",
    "lightgbm-baseline-cpu": "LightGBM (TF-IDF)",
    "llm-baseline": "GPT-5.6 Luna",
}


def _setup_tracking(tracking_uri: str | None) -> None:
    import mlflow

    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)


def _resolve_experiment_ids(
    experiment_names: tuple[str, ...] | None, experiment_ids: tuple[str, ...] | None
) -> list[str]:
    """Return experiment ids, prompting for a selection if none are given."""
    import mlflow

    ids: list[str] = list(experiment_ids or [])
    for name in experiment_names or []:
        exp = mlflow.get_experiment_by_name(name)
        if exp is None:
            raise click.ClickException(f"Experiment not found: {name!r}")
        ids.append(exp.experiment_id)
    if ids:
        return ids
    return _prompt_experiment_ids()


def _fetch_runs(
    experiment_ids: list[str] | None,
    filter_string: str,
    max_results: int,
    order_by: tuple[str, ...] | None,
):
    """Fetch runs via mlflow.search_runs(output_format='list')."""
    import mlflow

    kwargs: dict = {
        "filter_string": filter_string or "",
        "max_results": max_results,
        "output_format": "list",
    }
    if experiment_ids:
        kwargs["experiment_ids"] = experiment_ids
    if order_by:
        kwargs["order_by"] = list(order_by)
    return mlflow.search_runs(**kwargs)


def _is_failed(run) -> bool:
    return (run.info.status or "").upper() == "FAILED"


def _run_label(run) -> str:
    name = (run.data.tags or {}).get("mlflow.runName") or run.info.run_name
    display = MODEL_DISPLAY_NAMES.get((name or "").lower(), name)
    return display or run.info.run_id[:8]


def _find_metric(
    run_metrics: dict, candidates: list[str]
) -> tuple[str | None, float | None]:
    """Find first matching metric key (case-insensitive), return (key, value)."""
    lowered = {k.lower(): (k, v) for k, v in run_metrics.items()}
    for cand in candidates:
        if cand.lower() in lowered:
            key, val = lowered[cand.lower()]
            try:
                return key, float(val)
            except (TypeError, ValueError):
                return key, None
    return None, None


def _best_metric(
    run, candidates: list[str], higher_is_better: bool
) -> tuple[str | None, float | None]:
    """Best logged value for the first matching metric key.

    Early stopping means the best value may have been logged before the final
    one, so read the full metric history and pick max (F1) / min (RMSE).
    """
    import mlflow

    key, last = _find_metric(run.data.metrics or {}, candidates)
    if key is None:
        return None, None
    try:
        history = mlflow.tracking.MlflowClient().get_metric_history(
            run.info.run_id, key
        )
    except Exception:  # noqa: BLE001
        return key, last
    values = [m.value for m in history if m.value is not None]
    if not values:
        return key, last
    return key, max(values) if higher_is_better else min(values)


def _parse_selection(selection: str, n: int) -> list[int]:
    """Parse '0,2-4,all' style selection into sorted run indices."""
    selection = selection.strip().lower()
    if selection in {"all", "*"}:
        return list(range(n))
    indices: set[int] = set()
    for part in selection.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if m:
            start, end = int(m.group(1)), int(m.group(2))
            if start > end:
                start, end = end, start
            indices.update(range(start, end + 1))
        elif part.isdigit():
            indices.add(int(part))
        else:
            raise click.ClickException(
                f"Invalid selection {part!r}. Use comma-separated indices "
                f"(e.g. '0,2-3') or 'all'."
            )
    bad = sorted(i for i in indices if i < 0 or i >= n)
    if bad:
        raise click.ClickException(f"Selection out of range [0, {n - 1}]: {bad}")
    if not indices:
        raise click.ClickException("Empty selection.")
    return sorted(indices)


def _print_experiments_table(experiments) -> None:
    header = f"{'idx':>4}  {'experiment':<40}  experiment_id"
    click.echo(header)
    click.echo("-" * len(header))
    for i, exp in enumerate(experiments):
        click.echo(f"{i:>4}  {(exp.name or ''):<40}  {exp.experiment_id}")


def _prompt_experiment_ids() -> list[str]:
    """List experiments and prompt the user to select one or more."""
    import mlflow

    experiments = mlflow.search_experiments()
    if not experiments:
        raise click.ClickException("No experiments found.")
    _print_experiments_table(experiments)
    answer = click.prompt(
        "Select experiments (comma-separated indices, ranges, or 'all')",
        default="all",
    )
    indices = _parse_selection(answer, len(experiments))
    return [experiments[i].experiment_id for i in indices]


def _print_runs_table(runs) -> None:
    header = f"{'idx':>4}  {'run':<36}  {'status':<10}  metrics"
    click.echo(header)
    click.echo("-" * len(header))
    for i, run in enumerate(runs):
        f1_key, f1_val = _best_metric(run, F1_CANDIDATES, higher_is_better=True)
        rmse_key, rmse_val = _best_metric(run, RMSE_CANDIDATES, higher_is_better=False)
        summary = []
        if f1_val is not None:
            summary.append(f"{f1_key}={f1_val:.4f}")
        if rmse_val is not None:
            summary.append(f"{rmse_key}={rmse_val:.4f}")
        click.echo(
            f"{i:>4}  {_run_label(run):<36}  {run.info.status:<10}  "
            f"{', '.join(summary) or '(no F1/RMSE metrics)'}"
        )


def _print_results_table(labels, f1_vals, rmse_vals) -> None:
    cols = []
    if any(v is not None for v in f1_vals):
        cols.append(("F1", f1_vals))
    if any(v is not None for v in rmse_vals):
        cols.append(("RMSE", rmse_vals))
    header = f"{'model':<32}" + "".join(f"  {name:>8}" for name, _ in cols)
    header = header.rstrip()
    click.echo(header)
    click.echo("-" * len(header))
    for i, label in enumerate(labels):
        row = f"{label:<32}"
        for _, vals in cols:
            val = vals[i]
            cell = f"{val:.4f}" if val is not None else "n/a"
            row += f"  {cell:>8}"
        click.echo(row.rstrip())


def _save_results_csv(labels, f1_vals, rmse_vals, output: str) -> None:
    import csv

    include_f1 = any(v is not None for v in f1_vals)
    include_rmse = any(v is not None for v in rmse_vals)
    fields = ["model"]
    if include_f1:
        fields.append("f1")
    if include_rmse:
        fields.append("rmse")

    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for i, label in enumerate(labels):
            row = [label]
            if include_f1:
                row.append(f1_vals[i])
            if include_rmse:
                row.append(rmse_vals[i])
            writer.writerow(row)
    click.echo(f"Saved results table to {output}")


def _plot_comparison(
    labels: list[str], f1_vals, rmse_vals, output: str | None, show: bool
) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as e:
        raise click.ClickException(
            "matplotlib and seaborn are required for plotting "
            "(pip install -e '.[dev]')."
        ) from e

    sns.set_style("whitegrid")
    palette = sns.color_palette("colorblind", n_colors=max(len(labels), 1))
    hatches = ["", "//", "\\\\", "xx", "..", "++", "oo", "**", "OO", "--"]
    colors = [palette[i % len(palette)] for i in range(len(labels))]

    panels = []
    if any(v is not None for v in f1_vals):
        panels.append("f1")
    if any(v is not None for v in rmse_vals):
        panels.append("rmse")
    if not panels:
        raise click.ClickException(
            "None of the selected runs logged a known F1/RMSE metric "
            f"(looked for {F1_CANDIDATES} / {RMSE_CANDIDATES})."
        )

    fig, axes = plt.subplots(
        1, len(panels), figsize=(6 * len(panels), 5), squeeze=False
    )
    axes = axes[0]
    for ax, panel in zip(axes, panels):
        if panel == "f1":
            vals = [v if v is not None else float("nan") for v in f1_vals]
            ax.set_title("F1-Score (higher is better)")
            ax.set_ylabel("Macro F1")
        else:
            vals = [v if v is not None else float("nan") for v in rmse_vals]
            ax.set_title("RMSE (lower is better)")
            ax.set_ylabel("RMSE")
        sns.barplot(
            x=labels,
            y=vals,
            hue=labels,
            palette=colors,
            ax=ax,
            errorbar=None,
            legend=False,
            dodge=False,
            edgecolor="white",
        )
        for patch in ax.patches:
            idx = round(patch.get_x() + patch.get_width() / 2)
            if 0 <= idx < len(hatches):
                patch.set_hatch(hatches[idx % len(hatches)])
                patch.set_edgecolor("white")
        ax.tick_params(axis="x", rotation=30)
        for x, v in zip(labels, vals):
            import math

            if v is not None and not math.isnan(v):
                ax.text(x, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    if output:
        fig.savefig(output, dpi=150)
        click.echo(f"Saved bar chart to {output}")
    if show or not output:
        plt.show()
    else:
        plt.close(fig)


@click.group()
def cli() -> None:
    """Query MLflow runs and compare eval metrics."""


def _tracking_options(f):
    f = click.option(
        "--tracking-uri",
        default=None,
        help="MLflow tracking URI (default: $MLFLOW_TRACKING_URI or ./mlruns).",
    )(f)
    f = click.option(
        "--experiment-name",
        "experiment_names",
        multiple=True,
        default=None,
        help="Experiment name(s) to search. Repeatable.",
    )(f)
    f = click.option(
        "--experiment-id",
        "experiment_ids",
        multiple=True,
        default=None,
        help="Experiment ID(s) to search. Repeatable.",
    )(f)
    f = click.option(
        "--filter-string",
        default="",
        help="MLflow filter string, e.g. \"tags.task = 'clf'\".",
    )(f)
    f = click.option(
        "--max-results",
        default=50,
        show_default=True,
        help="Max runs to list.",
    )(f)
    return f


@cli.command("list-runs")
@_tracking_options
def list_runs(
    tracking_uri, experiment_names, experiment_ids, filter_string, max_results
):
    """List MLflow runs (one model per run) with their eval metrics."""
    try:
        import mlflow  # noqa: F401
    except ImportError as e:
        raise click.ClickException(
            "mlflow is required (pip install -e '.[torch]' or '.[lgbm]' / '.[llm]')."
        ) from e
    _setup_tracking(tracking_uri)
    ids = _resolve_experiment_ids(experiment_names, experiment_ids)
    runs = _fetch_runs(ids, filter_string, max_results, None)
    if not runs:
        click.echo("No runs found.")
        return
    _print_runs_table(runs)


@cli.command("compare")
@_tracking_options
@click.option(
    "--run-ids",
    default=None,
    help="Comma-separated run IDs to compare (skips interactive selection).",
)
@click.option(
    "--select",
    default=None,
    help="Non-interactive index selection, e.g. '0,2-3' or 'all'.",
)
@click.option(
    "--metric",
    type=click.Choice(["auto", "f1", "rmse"]),
    default="auto",
    show_default=True,
    help="Which metric panel(s) to plot.",
)
@click.option(
    "--output",
    default=None,
    help="Path to save the output (PNG, or CSV with --table). "
    "Nothing is saved unless this is set.",
)
@click.option(
    "--table",
    is_flag=True,
    default=False,
    help="Print a results table (CSV if --output is set) instead of plotting.",
)
@click.option("--show/--no-show", default=False, help="Display the plot interactively.")
def compare(
    tracking_uri,
    experiment_names,
    experiment_ids,
    filter_string,
    max_results,
    run_ids,
    select,
    metric,
    output,
    table,
    show,
):
    """List runs, select several, and plot F1/RMSE bars for comparison."""
    try:
        import mlflow  # noqa: F401
    except ImportError as e:
        raise click.ClickException(
            "mlflow is required (pip install -e '.[torch]' or '.[lgbm]' / '.[llm]')."
        ) from e
    _setup_tracking(tracking_uri)
    ids = _resolve_experiment_ids(experiment_names, experiment_ids)

    if run_ids:
        wanted = [r.strip() for r in run_ids.split(",") if r.strip()]
        runs = _fetch_runs(
            ids,
            f"attributes.run_id IN ({', '.join(repr(r) for r in wanted)})",
            max(max_results, len(wanted)),
            None,
        )
        found = {r.info.run_id: r for r in runs}
        missing = [r for r in wanted if r not in found]
        if missing:
            raise click.ClickException(f"Runs not found: {missing}")
        runs = [found[r] for r in wanted]
    else:
        runs = _fetch_runs(
            ids, filter_string, max_results, ["attributes.start_time DESC"]
        )
        if not runs:
            click.echo("No runs found.")
            return
        runs = [r for r in runs if not _is_failed(r)]
        if not runs:
            click.echo("No non-failed runs found.")
            return
        _print_runs_table(runs)
        if select:
            indices = _parse_selection(select, len(runs))
        else:
            answer = click.prompt(
                "Select runs (comma-separated indices, ranges, or 'all')",
                default="all",
            )
            indices = _parse_selection(answer, len(runs))
        runs = [runs[i] for i in indices]

    labels, f1_vals, rmse_vals = [], [], []
    for run in runs:
        _, f1 = _best_metric(run, F1_CANDIDATES, higher_is_better=True)
        _, rmse = _best_metric(run, RMSE_CANDIDATES, higher_is_better=False)
        labels.append(_run_label(run))
        f1_vals.append(f1)
        rmse_vals.append(rmse)
        f1_str = f"{f1:.4f}" if f1 is not None else "n/a"
        rmse_str = f"{rmse:.4f}" if rmse is not None else "n/a"
        click.echo(f"{_run_label(run)}: F1={f1_str} RMSE={rmse_str}")

    if metric == "f1":
        rmse_vals = [None] * len(rmse_vals)
    elif metric == "rmse":
        f1_vals = [None] * len(f1_vals)

    if table:
        _print_results_table(labels, f1_vals, rmse_vals)
        if output:
            _save_results_csv(labels, f1_vals, rmse_vals, output)
        return

    _plot_comparison(labels, f1_vals, rmse_vals, output, show)


if __name__ == "__main__":
    cli()
