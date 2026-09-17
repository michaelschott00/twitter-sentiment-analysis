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


def _setup_tracking(tracking_uri: str | None) -> None:
    import mlflow

    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)


def _resolve_experiment_ids(
    experiment_names: tuple[str, ...] | None, experiment_ids: tuple[str, ...] | None
) -> list[str] | None:
    """Return experiment ids, or None to mean 'use active/default experiment'."""
    import mlflow

    ids: list[str] = list(experiment_ids or [])
    for name in experiment_names or []:
        exp = mlflow.get_experiment_by_name(name)
        if exp is None:
            raise click.ClickException(f"Experiment not found: {name!r}")
        ids.append(exp.experiment_id)
    return ids or None


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


def _run_label(run) -> str:
    name = (run.data.tags or {}).get("mlflow.runName") or run.info.run_name
    if name:
        return f"{name} ({run.info.run_id[:8]})"
    return run.info.run_id[:8]


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


def _print_runs_table(runs) -> None:
    header = f"{'idx':>4}  {'run':<36}  {'status':<10}  metrics"
    click.echo(header)
    click.echo("-" * len(header))
    for i, run in enumerate(runs):
        metrics = run.data.metrics or {}
        f1_key, f1_val = _find_metric(metrics, F1_CANDIDATES)
        rmse_key, rmse_val = _find_metric(metrics, RMSE_CANDIDATES)
        summary = []
        if f1_val is not None:
            summary.append(f"{f1_key}={f1_val:.4f}")
        if rmse_val is not None:
            summary.append(f"{rmse_key}={rmse_val:.4f}")
        click.echo(
            f"{i:>4}  {_run_label(run):<36}  {run.info.status:<10}  "
            f"{', '.join(summary) or '(no F1/RMSE metrics)'}"
        )


def _plot_comparison(
    labels: list[str], f1_vals, rmse_vals, output: str | None, show: bool
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise click.ClickException(
            "matplotlib is required for plotting (pip install -e '.[torch]')."
        ) from e

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
            ax.bar(labels, vals, color="steelblue")
            ax.set_title("F1-Score (higher is better)")
            ax.set_ylabel("Macro F1")
        else:
            vals = [v if v is not None else float("nan") for v in rmse_vals]
            ax.bar(labels, vals, color="darkorange")
            ax.set_title("RMSE (lower is better)")
            ax.set_ylabel("RMSE")
        ax.set_xlabel("Run")
        ax.tick_params(axis="x", rotation=30)
        for x, v in zip(labels, vals):
            import math

            if v is not None and not math.isnan(v):
                ax.text(x, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("MLflow run comparison")
    fig.tight_layout()
    if output:
        fig.savefig(output, dpi=150)
        click.echo(f"Saved bar chart to {output}")
    if show:
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
    default="run_comparison.png",
    show_default=True,
    help="Path to save the bar chart PNG.",
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
        metrics = run.data.metrics or {}
        _, f1 = _find_metric(metrics, F1_CANDIDATES)
        _, rmse = _find_metric(metrics, RMSE_CANDIDATES)
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

    _plot_comparison(labels, f1_vals, rmse_vals, output, show)


if __name__ == "__main__":
    cli()
