# GPU training image. Built on RunPod's lean CUDA base, which ships no torch,
# so the whole torch stack is installed from pyproject.toml (the `torch` extra)
# and this project controls every version. A `runpod/pytorch` base instead
# ships its own CUDA build that pip won't replace — a `==x.y.z` pin matches
# `x.y.z+cuNNN` (PEP 440 ignores the local label) — so PyPI's cu128 wheels end
# up layered on a cu129 torch stack and the CUDA ABIs clash at import.
#
# Build context: repo root (so ./twitter and ./configs can be baked in).
FROM runpod/base:1.3.1-cuda1281-ubuntu2404

WORKDIR /project

ENV DEBIAN_FRONTEND=noninteractive

COPY ./pyproject.toml ./README.md ./
COPY ./twitter ./twitter
# NOTE: --ignore-installed blinker is required: Ubuntu 24.04 ships
# python3-blinker 1.7.0 via apt (no RECORD file, uninstallable by pip),
# while Flask (via mlflow) needs blinker>=1.9. Ignoring the apt copy lets
# pip install the newer wheel alongside instead of trying to uninstall it.
RUN pip install --no-cache-dir --ignore-installed blinker ".[torch]"
# azure-cli requires antlr4~=4.13 while omegaconf==2.3.1 requires
# antlr==4.9.* — installing both into one env upgrades antlr and breaks
# `import omegaconf` (surfaces in the training CLI as `error: 'omegaconf'`).
# So the CLI lives in an isolated venv; only the `az` entrypoint is linked
# into PATH (never the venv's bin dir — its `python` must not shadow the
# training interpreter).
RUN python -m venv /opt/azcli && \
    /opt/azcli/bin/pip install --no-cache-dir azure-cli && \
    ln -s /opt/azcli/bin/az /usr/local/bin/az && \
    az extension add --name ml --yes
# Build-time gate: fail the image build (not a GPU pod at $0.50+/hr) if the
# training CLI's import chain is broken. `twitter.main` pulls in
# jsonargparse/omegaconf (the antlr4 version conflict) and
# `twitter.data`/`models`/`modules` pull in torch, torchmetrics, lightning
# and transformers (CUDA-ABI drift).
RUN python -c "import twitter.data, twitter.models, twitter.modules, twitter.main; print('training imports ok')"

COPY ./configs ./configs

# RunPod base images expect /start.sh (Jupyter/SSH) to be running.
# Inline run.sh via heredoc so no extra file is needed. It starts
# RunPod services in the background, downloads the twitter-splits data
# asset via `az ml data download`, then runs whatever command is
# passed in (defaults to `python -m twitter.main --help`), so the image
# can be used for arbitrary LightningCLI runs, e.g.:
#   python -m twitter.main --config configs/tasks/classification.yaml \
#       --config configs/encoders/bertweet_large.yaml
# After a real (non-default) run finishes, the pod stops itself via
# `runpodctl stop pod $RUNPOD_POD_ID` since RunPod does not tear the
# pod down automatically when the script exits. Set RUNPOD_AUTO_STOP=0
# to opt out. Set SKIP_DATA_DOWNLOAD=1 to skip the splits download
# (e.g. when data is already baked into the volume).
RUN <<'EOF' cat > /usr/local/bin/run.sh
#!/bin/bash
set -euo pipefail
# Start RunPod base image services (Jupyter/SSH) in background
if [ -f /start.sh ]; then
  /start.sh &
  # Wait for services to start
  sleep 2
fi

# Default to showing LightningCLI help so the container does not exit
# immediately when no command is given. Override with any Lightning run,
# e.g.: python -m twitter.main --config configs/tasks/classification.yaml \
#   --config configs/encoders/bertweet_large.yaml
AUTO_STOP=1
if [ "$#" -eq 0 ]; then
  # The entrypoint already prepends `python -m twitter.main`, so only the
  # CLI flags belong here.
  set -- --help
  AUTO_STOP=0
fi
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
if [ "$AUTO_STOP" -eq 1 ]; then
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
        if [ -n "${AZURE_CLIENT_ID:-}" ] && [ -n "${AZURE_CLIENT_SECRET:-}" ] && [ -n "${AZURE_TENANT_ID:-}" ]; then
          az login --service-principal -u "$AZURE_CLIENT_ID" -p "$AZURE_CLIENT_SECRET" --tenant "$AZURE_TENANT_ID" >/dev/null
        fi
        mkdir -p "$DEST"
        # NOTE: `az ml data download` was removed from the Azure CLI ml
        # extension, so resolve the datastore-backed asset with
        # `az ml data show` (GA) and pull the blobs with `az storage`
        # (CLI core, works with the service-principal login above).
        # The twitter-splits asset is a uri_folder like
        # azureml://datastores/<datastore>/paths/<prefix>/.
        ASSET_PATH="$(az ml data show --name "$DATA_NAME" --version "$DATA_VERSION" \
          --subscription "$AZURE_SUBSCRIPTION_ID" \
          --resource-group "$AZURE_RESOURCE_GROUP" \
          --workspace-name "$AZUREML_WORKSPACE" \
          --query path -o tsv)"
        # `az ml data show` returns the path in short form
        # (azureml://datastores/<datastore>/paths/<prefix>/) or fully
        # qualified (azureml://subscriptions/.../datastores/<datastore>/paths/<prefix>/);
        # handle both.
        DS_NAME="$(printf '%s' "$ASSET_PATH" | sed -e 's#.*/datastores/##' -e 's#/.*$##')"
        DS_PREFIX="$(printf '%s' "$ASSET_PATH" | sed -e 's#.*/paths/##' -e 's#^/##' -e 's#/$##')"
        ACCT_NAME="$(az ml datastore show --name "$DS_NAME" \
          --subscription "$AZURE_SUBSCRIPTION_ID" \
          --resource-group "$AZURE_RESOURCE_GROUP" \
          --workspace-name "$AZUREML_WORKSPACE" \
          --query account_name -o tsv)"
        CONTAINER_NAME="$(az ml datastore show --name "$DS_NAME" \
          --subscription "$AZURE_SUBSCRIPTION_ID" \
          --resource-group "$AZURE_RESOURCE_GROUP" \
          --workspace-name "$AZUREML_WORKSPACE" \
          --query container_name -o tsv)"
        if [ -n "$DS_PREFIX" ]; then LIST_PREFIX="$DS_PREFIX/"; else LIST_PREFIX=""; fi
        BLOB_NAMES="$(az storage blob list --account-name "$ACCT_NAME" \
          --container-name "$CONTAINER_NAME" --prefix "$LIST_PREFIX" \
          --auth-mode login --query "[].name" -o tsv)"
        for BLOB in $BLOB_NAMES; do
          case "$BLOB" in
            *.csv)
              az storage blob download --account-name "$ACCT_NAME" \
                --container-name "$CONTAINER_NAME" --auth-mode login \
                --name "$BLOB" --file "$DEST/$(basename "$BLOB")" --no-progress
              ;;
          esac
        done
        ls -lh "$DEST"
      fi
      ;;
  esac

  # Fail fast if MLflow cannot reach the configured tracking backend. Without
  # this, lightning's MLFlowLogger silently falls back to a local file store
  # (file:./mlruns) and the pod is stopped before anyone notices the run never
  # reached Azure. Only real runs (AUTO_STOP=1) are checked; `--help`/debug
  # shells may legitimately run without MLflow.
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
      if ! python - <<'PY'
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
python -m twitter.main "$@"
EOF

RUN chmod +x /usr/local/bin/run.sh

# No CMD: with no docker args, run.sh runs `python -m twitter.main --help`
# (AUTO_STOP=0) instead of the old CMD, which the entrypoint would have
# prepended a second `python -m twitter.main` to.
ENTRYPOINT ["/usr/local/bin/run.sh"]
