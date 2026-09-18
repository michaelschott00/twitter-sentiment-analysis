# GPU training image. torch comes from the base image; the `torch` extra
# leaves torch unpinned on purpose so pip keeps the base image's CUDA
# build instead of reinstalling it. All other deps come from
# pyproject.toml (single source of truth) — see the `torch` extra.
#
# Build context: repo root (so ./twitter and ./configs can be baked in).
FROM runpod/pytorch:1.1.0-rc.154-cu1290-torch290-ubuntu2404

WORKDIR /project

ENV DEBIAN_FRONTEND=noninteractive

COPY ./pyproject.toml ./README.md ./
COPY ./twitter ./twitter
# NOTE: --ignore-installed blinker is required: Ubuntu 24.04 ships
# python3-blinker 1.7.0 via apt (no RECORD file, uninstallable by pip),
# while Flask (via mlflow) needs blinker>=1.9. Ignoring the apt copy lets
# pip install the newer wheel alongside instead of trying to uninstall it.
RUN pip install --no-cache-dir --ignore-installed blinker ".[torch]" && \
    pip install --no-cache-dir azure-cli && \
    az extension add --name ml --yes

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
  set -- python -m twitter.main --help
  AUTO_STOP=0
fi
case "${RUNPOD_AUTO_STOP:-}" in
  0|false|False|FALSE|no|No|NO)
    AUTO_STOP=0
    ;;
esac
# The RunPod platform resolves a relative start-command binary against the
# volume mount path (e.g. `python` arrives as `/workspace/python`) and uses
# that path as the working directory. Map the binary back onto PATH and run
# from the baked-in checkout so relative `configs/...` paths resolve.
if [ "$#" -gt 0 ]; then
  case "$1" in
    */*)
      if [ ! -x "$1" ]; then
        FIXED_CMD="$(command -v "$(basename "$1")" || true)"
        if [ -n "$FIXED_CMD" ]; then
          if [ "$#" -gt 1 ]; then set -- "$FIXED_CMD" "${@:2}"; else set -- "$FIXED_CMD"; fi
        fi
      fi
      ;;
  esac
fi
if [ -d /project/configs ]; then
  cd /project
fi
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
fi
"$@"
EXIT_CODE=$?
if [ "$AUTO_STOP" -eq 1 ]; then
  if [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null 2>&1; then
    runpodctl stop pod "$RUNPOD_POD_ID"
  else
    echo "run.sh: skipping pod stop (RUNPOD_POD_ID unset or runpodctl missing)" >&2
  fi
fi
exit "$EXIT_CODE"
EOF

RUN chmod +x /usr/local/bin/run.sh

ENTRYPOINT ["/usr/local/bin/run.sh"]
CMD ["python", "-m", "twitter.main", "--help"]
