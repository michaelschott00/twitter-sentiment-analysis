"""Inspect MLflow experiments, runs and metric history (non-interactive).

Usage:
    python -m twitter.mlflow_inspect list-experiments
    python -m twitter.mlflow_inspect list-runs --experiment twitter-sentiment
    python -m twitter.mlflow_inspect list-metrics --run <run_id>
    python -m twitter.mlflow_inspect history --run <run_id>
    python -m twitter.mlflow_inspect history --run <run_id> --metric MulticlassF1Score
"""

from __future__ import annotations

import datetime as dt
import os

import click


def _setup_tracking(tracking_uri: str | None) -> None:
    import mlflow

    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)


def _fmt_time(ms: int | None) -> str:
    if not ms:
        return ""
    ts = dt.datetime.fromtimestamp(ms / 1000, tz=dt.UTC)
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def _experiment_id(experiment: str) -> str:
    """Resolve an experiment name or id to an experiment id."""
    import mlflow

    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        try:
            exp = mlflow.get_experiment(experiment)
        except Exception:  # noqa: BLE001
            raise click.ClickException(
                f"Experiment not found: {experiment!r}"
            ) from None
    return exp.experiment_id


def _get_run(run_id: str):
    import mlflow

    try:
        return mlflow.get_run(run_id)
    except Exception:  # noqa: BLE001
        raise click.ClickException(f"Run not found: {run_id!r}") from None


def _run_name(run) -> str:
    return (run.data.tags or {}).get("mlflow.runName") or run.info.run_name or ""


@click.group()
@click.option(
    "--tracking-uri",
    default=None,
    help="MLflow tracking URI (default: $MLFLOW_TRACKING_URI or ./mlruns).",
)
def cli(tracking_uri: str | None) -> None:
    """Inspect MLflow experiments, runs and metric history."""
    _setup_tracking(tracking_uri)


@cli.command("list-experiments")
def list_experiments() -> None:
    """List experiments with their ids."""
    import mlflow

    experiments = mlflow.search_experiments()
    if not experiments:
        click.echo("No experiments found.")
        return
    header = f"{'experiment_id':<24}  {'name':<40}  lifecycle_stage"
    click.echo(header)
    click.echo("-" * len(header))
    for exp in experiments:
        click.echo(
            f"{exp.experiment_id:<24}  {(exp.name or ''):<40}  {exp.lifecycle_stage}"
        )


@cli.command("list-runs")
@click.option("--experiment", required=True, help="Experiment name or id.")
def list_runs(experiment: str) -> None:
    """List runs for an experiment."""
    import mlflow

    exp_id = _experiment_id(experiment)
    runs = mlflow.search_runs(
        experiment_ids=[exp_id],
        output_format="list",
        order_by=["attributes.start_time DESC"],
    )
    if not runs:
        click.echo("No runs found.")
        return
    header = f"{'run_id':<36}  {'run_name':<32}  {'status':<10}  start_time"
    click.echo(header)
    click.echo("-" * len(header))
    for run in runs:
        click.echo(
            f"{run.info.run_id:<36}  {_run_name(run):<32}  "
            f"{run.info.status:<10}  {_fmt_time(run.info.start_time)}"
        )


@cli.command("list-metrics")
@click.option("--run", "run_id", required=True, help="Run id.")
def list_metrics(run_id: str) -> None:
    """List metrics logged for a run (last value per metric)."""
    run = _get_run(run_id)
    metrics = run.data.metrics or {}
    if not metrics:
        click.echo("No metrics logged for this run.")
        return
    header = f"{'metric':<40}  {'last_value':>16}"
    click.echo(header)
    click.echo("-" * len(header))
    for key in sorted(metrics):
        click.echo(f"{key:<40}  {metrics[key]:>16.6g}")


@cli.command("history")
@click.option("--run", "run_id", required=True, help="Run id.")
@click.option(
    "--metric",
    default=None,
    help="Metric name (case-insensitive). Omit to show every metric.",
)
@click.option(
    "--experiment",
    default=None,
    help="Experiment name or id; if set, the run must belong to it.",
)
def history(run_id: str, metric: str | None, experiment: str | None) -> None:
    """Print the metric history (step, value, timestamp) for a run."""
    import mlflow

    run = _get_run(run_id)
    if experiment is not None:
        exp_id = _experiment_id(experiment)
        if run.info.experiment_id != exp_id:
            raise click.ClickException(
                f"Run {run_id!r} does not belong to experiment {experiment!r}."
            )

    metrics = run.data.metrics or {}
    if metric is not None:
        lowered = {k.lower(): k for k in metrics}
        if metric.lower() not in lowered:
            available = ", ".join(sorted(metrics)) or "(none)"
            raise click.ClickException(
                f"Metric {metric!r} not logged for run {run_id!r}. "
                f"Available: {available}"
            )
        keys = [lowered[metric.lower()]]
    else:
        keys = sorted(metrics)
    if not keys:
        click.echo("No metrics logged for this run.")
        return

    click.echo(f"run: {run_id} ({_run_name(run)})")
    client = mlflow.tracking.MlflowClient()
    for key in keys:
        points = client.get_metric_history(run_id, key)
        click.echo(f"\n{key} ({len(points)} points)")
        header = f"{'step':>8}  {'value':>18}  timestamp (UTC)"
        click.echo(header)
        click.echo("-" * len(header))
        for point in points:
            click.echo(
                f"{point.step:>8}  {point.value:>18.6g}  {_fmt_time(point.timestamp)}"
            )


if __name__ == "__main__":
    cli()
