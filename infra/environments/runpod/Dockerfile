# GPU training image. torch comes from the base image; LightGBM-only
# deps (lightgbm, imbalanced-learn) are not needed here.
#
# Build context: repo root (so ./twitter and ./configs can be baked in).
FROM runpod/pytorch:1.1.0-rc.154-cu1290-torch290-ubuntu2404

WORKDIR /project

ENV DEBIAN_FRONTEND=noninteractive

RUN pip install --no-cache-dir \
    "lightning" \
    "transformers" \
    "torchmetrics" \
    "pandas" \
    "scikit-learn" \
    "numpy" \
    "jsonargparse[signatures]" \
    "omegaconf" \
    "sacremoses" \
    "emoji" \
    "nltk" \
    "click" \
    "pyyaml" \
    "tqdm" \
    "mlflow<=2.16.2" \
    "azure-ai-ml" \
    "azureml-mlflow"

COPY ./twitter ./twitter
COPY ./configs ./configs

# RunPod base images expect /start.sh (Jupyter/SSH) to be running.
# Inline run.sh via heredoc so no extra file is needed. It starts
# RunPod services in the background, then runs whatever command is
# passed in (defaults to `python -m twitter.main --help`), so the image
# can be used for arbitrary LightningCLI runs, e.g.:
#   python -m twitter.main --config configs/tasks/classification.yaml \
#       --config configs/encoders/sentence_bert_base.yaml
# After a real (non-default) run finishes, the pod stops itself via
# `runpodctl stop pod $RUNPOD_POD_ID` since RunPod does not tear the
# pod down automatically when the script exits. Set RUNPOD_AUTO_STOP=0
# to opt out.
RUN <<'EOF' cat > /usr/local/bin/run.sh
#!/bin/bash
# Start RunPod base image services (Jupyter/SSH) in background
if [ -f /start.sh ]; then
  /start.sh &
  # Wait for services to start
  sleep 2
fi
# Default to showing LightningCLI help so the container does not exit
# immediately when no command is given. Override with any Lightning run,
# e.g.: python -m twitter.main --config configs/tasks/classification.yaml \
#   --config configs/encoders/sentence_bert_base.yaml
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
