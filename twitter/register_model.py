"""Register an already-finished run's model in the MLflow model registry.

Training registers the best checkpoint automatically (see
``twitter.modules._BaseModule.on_train_end``), but Azure ML's artifact-upload
flush timeout can fail that step even though the run and its
``registered_model`` artifact are intact. This command retries just the
registration against the existing run, without retraining:

    python -m twitter.register_model --run-id <run_id>

The registry name is derived from the run's logged ``task`` and
``encoder_name`` params (``twitter-{task}-{encoder}``) unless ``--name`` is
given. Tracking uses ``MLFLOW_TRACKING_URI`` unless ``--tracking-uri`` is set.
"""

import os
import time

import click
import mlflow
from mlflow.tracking import MlflowClient

from twitter.mlflow_model import registry_name


def _derive_name(client: MlflowClient, run_id: str) -> str:
    params = client.get_run(run_id).data.params
    encoder_name = params.get("encoder_name")
    if not encoder_name:
        raise click.ClickException(
            f"run {run_id} has no 'encoder_name' param; pass --name explicitly"
        )
    return registry_name(encoder_name, params.get("task") or "clf")


@click.command()
@click.option("--run-id", required=True, help="MLflow run id that holds the artifact")
@click.option(
    "--artifact-path",
    default="registered_model",
    show_default=True,
    help="Run-relative artifact path to register",
)
@click.option(
    "--name",
    default=None,
    help="Registry name (default: derived from the run's task/encoder params)",
)
@click.option(
    "--tracking-uri",
    default=None,
    help="MLflow tracking URI (default: $MLFLOW_TRACKING_URI)",
)
@click.option("--max-attempts", default=3, show_default=True, type=int)
@click.option("--retry-delay", default=10.0, show_default=True, type=float)
def main(run_id, artifact_path, name, tracking_uri, max_attempts, retry_delay):
    """Register runs:/<run-id>/<artifact-path> in the MLflow model registry."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    elif not os.environ.get("MLFLOW_TRACKING_URI"):
        raise click.ClickException("set MLFLOW_TRACKING_URI or pass --tracking-uri")

    client = MlflowClient()
    if name is None:
        name = _derive_name(client, run_id)

    uri = f"runs:/{run_id}/{artifact_path}"
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            version = mlflow.register_model(uri, name)
            click.echo(f"Registered {name} version {version.version} from {uri}")
            return
        except Exception as e:  # noqa: BLE001
            last_error = e
            if attempt < max_attempts:
                click.echo(f"Registration attempt {attempt} failed: {e}; retrying")
                time.sleep(retry_delay * attempt)
    raise click.ClickException(f"could not register {uri} as {name}: {last_error}")


if __name__ == "__main__":
    main()
