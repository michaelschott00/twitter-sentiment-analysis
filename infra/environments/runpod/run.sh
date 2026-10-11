#!/bin/bash
set -euo pipefail

# Training interpreter lives in the uv-managed venv baked into the image (see
# Dockerfile). It is deliberately NOT on the default PATH so the base image's
# Jupyter/SSH services keep using the system interpreter.
PY=/opt/venv/bin/python

# Start RunPod base image services (Jupyter/SSH) in background
if [ -f /start.sh ]; then
  /start.sh &
  # Wait for services to start
  sleep 2
fi

# The image ships dependencies only; the sources (twitter/, configs/) are cloned
# onto the pod at start so a source change does not require an image rebuild --
# only a dependency change (pyproject.toml) does. Override the defaults to pin a
# branch/tag, use a fork, or clone a private mirror with GITHUB_TOKEN.
REPO_URL="${REPO_URL:-https://github.com/michaelschott00/twitter-sentiment-analysis.git}"
REPO_REF="${REPO_REF:-main}"
REPO_DIR="${REPO_DIR:-/project}"
if [ -n "${GITHUB_TOKEN:-}" ]; then
  REPO_URL="https://x-access-token:${GITHUB_TOKEN}@${REPO_URL#https://}"
fi
if [ -d "$REPO_DIR/.git" ]; then
  echo "run.sh: updating $REPO_DIR (ref=$REPO_REF)" >&2
  git -C "$REPO_DIR" fetch --depth 1 origin "$REPO_REF"
  git -C "$REPO_DIR" checkout -q FETCH_HEAD
else
  echo "run.sh: cloning $REPO_URL (ref=$REPO_REF) -> $REPO_DIR" >&2
  git clone --depth 1 --branch "$REPO_REF" "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
# Source-level import gate: catches a broken import chain (torch/CUDA ABI drift,
# omegaconf's antlr pin) before spending GPU time. The dependency-level gate in
# the Dockerfile runs at build time; this one validates the cloned sources.
"$PY" -c "import twitter.data, twitter.models, twitter.modules, twitter.main; print('training imports ok')" >&2

# Default to showing LightningCLI help so the container does not exit
# immediately when no command is given. Override with any Lightning run,
# e.g.: python -m twitter.main --config configs/tasks/classification.yaml \
#   --config configs/encoders/bertweet_large.yaml
# REAL_RUN distinguishes a real training invocation from the `--help`/debug
# default. Data download + the MLflow reachability check run for every real
# run, independent of AUTO_STOP: opting out of the self-stop
# (`RUNPOD_AUTO_STOP=0`) must not also silently skip data preparation and
# crash training with a missing-CSV error.
REAL_RUN=1
if [ "$#" -eq 0 ]; then
  # The entrypoint already prepends `python -m twitter.main`, so only the
  # CLI flags belong here.
  set -- --help
  REAL_RUN=0
fi
AUTO_STOP=$REAL_RUN
case "${RUNPOD_AUTO_STOP:-}" in
  0|false|False|FALSE|no|No|NO)
    AUTO_STOP=0
    ;;
esac

# Stop the pod on *any* exit (success, training error, or a failure in the
# data-download/MLflow checks) when AUTO_STOP is on. Doing this in an EXIT trap
# is essential because `set -e` aborts the script at the failing command, so a
# trailing `runpodctl stop` after the training command would be skipped on error
# and the GPU pod would keep billing.
stop_pod() {
  exit_code=$?
  if [ "${AUTO_STOP:-0}" -eq 1 ]; then
    if [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null 2>&1; then
      echo "run.sh: stopping pod $RUNPOD_POD_ID (exit=$exit_code)" >&2
      runpodctl stop pod "$RUNPOD_POD_ID" || echo "run.sh: runpodctl stop pod failed" >&2
    else
      echo "run.sh: skipping pod stop (RUNPOD_POD_ID unset or runpodctl missing)" >&2
    fi
  fi
  exit "$exit_code"
}
trap stop_pod EXIT

# Download twitter-splits (tweets_{train,dev,test_1,test_2}.csv) so
# TwitterDataModule(root_dir=data/splits) works. Only for real runs,
# so `--help` / debug shells don't require Azure credentials.
DEST="${SPLITS_DIR:-data/splits}"
DATA_NAME="${DATA_NAME:-twitter-splits}"
DATA_VERSION="${DATA_VERSION:-1}"
if [ "$REAL_RUN" -eq 1 ]; then
  case "${SKIP_DATA_DOWNLOAD:-0}" in
    1|true|True|TRUE|yes|Yes|YES)
      echo "run.sh: skipping splits download (SKIP_DATA_DOWNLOAD set)" >&2
      ;;
    *)
      if [ -f "$DEST/tweets_train.csv" ] && [ -f "$DEST/tweets_dev.csv" ] && [ -f "$DEST/tweets_test_1.csv" ] && [ -f "$DEST/tweets_test_2.csv" ]; then
        echo "run.sh: splits already present in $DEST, skipping download" >&2
      else
        : "${AZURE_SUBSCRIPTION_ID:?run.sh: set AZURE_SUBSCRIPTION_ID}"
        : "${AZURE_RESOURCE_GROUP:?run.sh: set AZURE_RESOURCE_GROUP}"
        : "${AZUREML_WORKSPACE:?run.sh: set AZUREML_WORKSPACE}"
        mkdir -p "$DEST"
        # Resolve the datastore-backed `twitter-splits` asset with the Azure ML
        # SDK and pull the blobs with azure-storage-blob. Auth is
        # DefaultAzureCredential, which picks up the injected service-principal
        # env vars (AZURE_CLIENT_ID/SECRET/TENANT_ID) automatically.
        "$PY" /usr/local/bin/fetch_splits.py \
          --subscription-id "$AZURE_SUBSCRIPTION_ID" \
          --resource-group "$AZURE_RESOURCE_GROUP" \
          --workspace "$AZUREML_WORKSPACE" \
          --data-name "$DATA_NAME" \
          --data-version "$DATA_VERSION" \
          --dest "$DEST"
        ls -lh "$DEST"
      fi
      ;;
  esac

  # Fail fast if MLflow cannot reach the configured tracking backend. Without
  # this, lightning's MLFlowLogger silently falls back to a local file store
  # (file:./mlruns) and the pod is stopped before anyone notices the run never
  # reached Azure. Only real runs are checked; `--help`/debug shells may
  # legitimately run without MLflow.
  case "${SKIP_MLFLOW_CHECK:-0}" in
    1|true|True|TRUE|yes|Yes|YES)
      echo "run.sh: skipping MLflow reachability check (SKIP_MLFLOW_CHECK set)" >&2
      ;;
    *)
      : "${MLFLOW_TRACKING_URI:?run.sh: set MLFLOW_TRACKING_URI (azureml://<region>.api.azureml.ms/mlflow/v1.0/<workspace-id>)}"
      if [ "${MLFLOW_TRACKING_URI#azureml://}" != "$MLFLOW_TRACKING_URI" ]; then
        : "${AZURE_TENANT_ID:?run.sh: set AZURE_TENANT_ID for azureml:// MLflow}"
        : "${AZURE_CLIENT_ID:?run.sh: set AZURE_CLIENT_ID for azureml:// MLflow}"
        : "${AZURE_CLIENT_SECRET:?run.sh: set AZURE_CLIENT_SECRET for azureml:// MLflow}"
      fi
      if ! "$PY" - <<'PY'
import os
import sys

import mlflow

uri = os.environ["MLFLOW_TRACKING_URI"]
name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "twitter-sentiment")
mlflow.set_tracking_uri(uri)
experiment = mlflow.set_experiment(name)
print(f"run.sh: MLflow ok ({uri}, experiment={experiment.name})", file=sys.stderr)
PY
      then
        echo "run.sh: MLflow reachability check failed for MLFLOW_TRACKING_URI='$MLFLOW_TRACKING_URI'." >&2
        echo "run.sh: refusing to train into a local ./mlruns fallback; fix credentials/URI or set SKIP_MLFLOW_CHECK=1." >&2
        exit 1
      fi
      ;;
  esac
fi

# Run main command. The EXIT trap above stops the pod afterwards, including
# when this command (or any earlier step) fails.
"$PY" -m twitter.main "$@"
