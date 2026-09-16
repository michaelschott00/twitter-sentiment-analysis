"""LLM baseline for Twitter sentiment analysis via the OpenAI Batch API.

Uses few-shot prompting to obtain both classification (sentiment) and
regression (valence) in a single prompt. Requests are sent asynchronously
through the `Batch API <https://developers.openai.com/api/docs/guides/batch>`_
(50% lower cost, separate higher rate limits, 24h turnaround) instead of
one synchronous ``chat.completions.create`` call per tweet.

Data source (single source of truth: the ``twitter-splits`` Azure ML data
asset, ``infra/data/twitter-splits.yaml``):

1. Explicit ``--dev-path`` / ``--train-path`` CSVs (local files or an AML
   ``${{inputs.splits}}`` mount, e.g. ``--splits-dir ${{inputs.splits}}``).
2. ``--splits-dir`` folder containing ``tweets_dev.csv`` / ``tweets_train.csv``
   (same ``uri_folder`` mount pattern as ``infra/jobs/*.yaml``).
3. Direct download from the data asset via ``MLClient`` + ``azure-storage-blob``
   (same approach as ``infra/scripts/read_remote_data.py`` — bypasses
   ``azureml-fsspec`` browser login): pass ``--resource-group`` / ``--workspace``
   (plus ``--subscription-id`` or ``$AZURE_SUBSCRIPTION_ID``).
4. Legacy local fallback ``data/splits/tweets_*.csv`` (deprecated).

Workflow (live mode):

1. Build one ``/v1/chat/completions`` request per tweet and write them as
   JSONL (each line: ``custom_id``, ``method``, ``url``, ``body``).
2. Upload the JSONL with ``purpose="batch"`` via the Files API.
3. Create a batch (``endpoint="/v1/chat/completions"``,
   ``completion_window="24h"``).
4. Poll ``batches.retrieve`` until the batch reaches a terminal status
   (or exit early with ``--no-wait`` and resume later with ``--batch-id``).
5. Download the output JSONL via ``files.content(output_file_id)``, map each
   line back to its tweet with ``custom_id``, parse, evaluate and log.

Supports token estimation with tiktoken and dry-run mode (prints the batch
request lines instead of calling the API) when credentials are missing.

MLflow tracking:
  - Dry-run / --estimate-tokens paths stay fully local (no MLflow calls).
  - Live mode logs to MLflow: tracking URI is read from the
    ``MLFLOW_TRACKING_URI`` env var (e.g. an ``azureml://...`` URI when
    running locally but tracking to an Azure ML workspace), experiment
    ``twitter-sentiment``, run name ``llm-baseline``.
  - Authentication to Azure ML is via service-principal env vars
    (``AZURE_TENANT_ID``, ``AZURE_CLIENT_ID``, ``AZURE_CLIENT_SECRET``),
    picked up automatically by ``DefaultAzureCredential`` through the
    ``azureml-mlflow`` plugin — no explicit login code needed.
  - No model is logged or registered (no training); params, metrics and
    artifacts (predictions, LLM responses, report, prompts, batch
    input/manifest/output) are tracked.
"""

import json
import os
import re
import tempfile
import time

import click
import numpy as np
import pandas as pd
import tiktoken

try:
    import openai
except ImportError:
    openai = None  # type: ignore[assignment]

from sklearn.metrics import f1_score, mean_squared_error

try:
    from sklearn.metrics import root_mean_squared_error

    _HAS_RMSE = True
except ImportError:
    _HAS_RMSE = False

from twitter.labels import LABEL_CODING

DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_ENCODING = "o200k_base"

MLFLOW_EXPERIMENT_NAME = "twitter-sentiment"
MLFLOW_RUN_NAME = "llm-baseline"

BATCH_ENDPOINT = "/v1/chat/completions"
BATCH_METHOD = "POST"
DEFAULT_COMPLETION_WINDOW = "24h"
TERMINAL_BATCH_STATUSES = {"completed", "failed", "expired", "cancelled"}

DEFAULT_DATA_NAME = "twitter-splits"
DEFAULT_DATA_VERSION = "1"
LEGACY_DEV_PATH = "data/splits/tweets_dev.csv"
LEGACY_TRAIN_PATH = "data/splits/tweets_train.csv"

SYSTEM_PROMPT = (
    "You are an expert sentiment analysis model for Twitter data, "
    "specialized in AI and Machine Learning related tweets.\n\n"
    "Task: Given a tweet, predict:\n"
    "1. sentiment: one of 'negative', 'neutral', 'positive'\n"
    "2. valence: a float in [-1.0, 1.0] where -1 is most negative, "
    "0 is neutral, 1 is most positive.\n\n"
    "The valence score should align with sentiment but can be nuanced:\n"
    "- negative: valence in [-1.0, -0.1)\n"
    "- neutral: valence around 0, roughly [-0.3, 0.3]\n"
    "- positive: valence in (0.1, 1.0]\n\n"
    "Return a JSON object with exactly two keys:\n"
    '- "sentiment": string, one of "negative", "neutral", "positive"\n'
    '- "valence": number between -1 and 1\n\n'
    'Example: {"sentiment": "positive", "valence": 0.8}\n'
    "Output ONLY the JSON object, no extra text."
)


# ---------------------------------------------------------------------------
# Data loading: twitter-splits asset is the single source of truth.
# ---------------------------------------------------------------------------


def download_split_from_asset(
    filename: str,
    resource_group: str,
    workspace: str,
    subscription_id: str | None,
    data_name: str = DEFAULT_DATA_NAME,
    data_version: str = DEFAULT_DATA_VERSION,
    dest_path: str | None = None,
) -> str:
    """Download one CSV from the ``twitter-splits`` data asset.

    Same approach as ``infra/scripts/read_remote_data.py``: resolve the
    datastore via ``MLClient``, then read the blob directly with
    ``azure-storage-blob`` (bypasses ``azureml-fsspec`` browser login).

    Returns the local path the CSV was written to (``dest_path`` or a path
    inside the system temp dir).
    """
    try:
        from azure.ai.ml import MLClient
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import BlobClient
    except ImportError as e:
        raise click.ClickException(
            "Azure packages are required to load from the data asset "
            "(pip install -e '.[llm,dev]')."
        ) from e

    if not subscription_id:
        raise click.ClickException(
            "Missing --subscription-id / $AZURE_SUBSCRIPTION_ID."
        )
    credential = DefaultAzureCredential()
    ml_client = MLClient(credential, subscription_id, resource_group, workspace)

    asset = ml_client.data.get(name=data_name, version=data_version)
    match = re.search(r"/datastores/([^/]+)/paths/(.*)", asset.path)
    if not match:
        raise click.ClickException(
            f"Could not parse datastore from data asset path: {asset.path!r}"
        )
    datastore_name, prefix = match.groups()

    ds = ml_client.datastores.get(datastore_name)
    container = ds.container_name
    endpoint = getattr(ds, "endpoint", None) or "core.windows.net"

    blob_path = f"{prefix.strip('/')}/{filename}" if prefix.strip("/") else filename
    blob = BlobClient(
        f"https://{ds.account_name}.blob.{endpoint}",
        container,
        blob_path,
        credential=credential,
    )
    content = blob.download_blob().readall()

    if dest_path is None:
        dest_path = os.path.join(tempfile.gettempdir(), filename)
    with open(dest_path, "wb") as f:
        f.write(content)
    return dest_path


def resolve_split_paths(
    dev_path: str | None,
    train_path: str | None,
    splits_dir: str | None,
    resource_group: str | None,
    workspace: str | None,
    subscription_id: str | None,
    data_name: str,
    data_version: str,
) -> tuple[str, str, dict]:
    """Resolve dev/train CSV paths, preferring the data asset over local files.

    Precedence per split: explicit ``--dev-path``/``--train-path`` >
    ``--splits-dir`` (``uri_folder`` mount, same as ``infra/jobs/*.yaml``) >
    direct asset download (``--resource-group``/``--workspace``) > legacy
    ``data/splits/`` fallback.

    Returns ``(dev_path, train_path, provenance)`` where ``provenance`` records
    how each split was resolved (logged to MLflow).
    """
    provenance: dict = {"data_name": data_name, "data_version": data_version}
    use_asset = bool(resource_group and workspace)

    def _resolve_one(explicit: str | None, filename: str, label: str) -> str:
        if explicit:
            provenance[label] = f"explicit:{explicit}"
            return explicit
        if splits_dir:
            candidate = os.path.join(splits_dir, filename)
            provenance[label] = f"splits_dir:{candidate}"
            return candidate
        if use_asset:
            dest = os.path.join(tempfile.gettempdir(), filename)
            download_split_from_asset(
                filename,
                resource_group,
                workspace,
                subscription_id,
                data_name,
                data_version,
                dest_path=dest,
            )
            provenance[label] = f"asset:{data_name}:{data_version}/{filename}"
            click.echo(f"Downloaded {label} from {data_name}:{data_version} -> {dest}")
            return dest
        legacy = LEGACY_DEV_PATH if label == "dev" else LEGACY_TRAIN_PATH
        provenance[label] = f"legacy:{legacy}"
        click.echo(
            f"Warning: using legacy local {label} path {legacy}; prefer "
            "--splits-dir or --resource-group/--workspace (data asset).",
            err=True,
        )
        return legacy

    resolved_dev = _resolve_one(dev_path, "tweets_dev.csv", "dev")
    resolved_train = _resolve_one(train_path, "tweets_train.csv", "train")
    return resolved_dev, resolved_train, provenance


def read_split_csv(path: str, label: str) -> "pd.DataFrame":
    """Read a resolved split CSV, raising a Click error when missing."""
    if not os.path.exists(path):
        raise click.ClickException(f"{label} file not found: {path}")
    return pd.read_csv(path)


def _setup_mlflow_tracking() -> None:
    """Point MLflow at the tracking URI and experiment.

    Reads the tracking URI from the ``MLFLOW_TRACKING_URI`` env var
    (e.g. ``azureml://...`` for the Azure ML workspace). When the URI
    targets Azure ML, authentication relies on the service-principal env
    vars (``AZURE_TENANT_ID``, ``AZURE_CLIENT_ID``,
    ``AZURE_CLIENT_SECRET``) via ``DefaultAzureCredential`` in the
    ``azureml-mlflow`` plugin — no explicit login code needed.

    Falls back to MLflow's default local tracking (``./mlruns``) when the
    env var is unset.
    """
    try:
        import mlflow
    except ImportError as e:
        raise click.ClickException(
            "MLflow is required for live runs (pip install -e '.[llm]'). "
            "Use --dry-run for local runs without MLflow."
        ) from e

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        click.echo(f"MLflow tracking URI: {tracking_uri}")
    else:
        click.echo(
            "MLFLOW_TRACKING_URI not set — using default local tracking (./mlruns).",
            err=True,
        )

    if (tracking_uri or "").startswith("azureml://"):
        missing = [
            var
            for var in (
                "AZURE_TENANT_ID",
                "AZURE_CLIENT_ID",
                "AZURE_CLIENT_SECRET",
            )
            if not os.getenv(var)
        ]
        if missing:
            click.echo(
                f"Warning: Azure ML tracking URI set but missing: {', '.join(missing)}. "
                "Set them for service-principal auth.",
                err=True,
            )
        else:
            click.echo(
                "Using service-principal auth from AZURE_* env vars "
                "(via DefaultAzureCredential)."
            )

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def get_encoding(encoding_name: str = DEFAULT_ENCODING):
    """Return tiktoken encoding for token counting.

    Tries ``get_encoding`` first, falls back to ``encoding_for_model``.
    """
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception:  # noqa: BLE001
        try:
            return tiktoken.encoding_for_model(DEFAULT_MODEL)
        except Exception:  # noqa: BLE001
            return tiktoken.get_encoding("o200k_base")


def count_tokens(text: str, encoding) -> int:
    """Count tokens for a single text using the given encoding."""
    return len(encoding.encode(text))


def estimate_token_counts(dev_path: str, encoding_name: str = DEFAULT_ENCODING) -> dict:
    """Estimate token counts for tweets in validation set.

    Prints total, average, max, min, median and returns stats dict.
    """
    if not os.path.exists(dev_path):
        raise FileNotFoundError(f"Dev file not found: {dev_path}")

    enc = get_encoding(encoding_name)
    df = pd.read_csv(dev_path)
    if "text" not in df.columns:
        raise ValueError("CSV missing 'text' column")

    counts = df["text"].astype(str).apply(lambda x: count_tokens(x, enc))

    total = int(counts.sum())
    avg = float(counts.mean())
    mx = int(counts.max())
    mn = int(counts.min())
    median = float(counts.median())
    p95 = float(counts.quantile(0.95))
    std = float(counts.std())

    click.echo(f"Encoding: {encoding_name}")
    click.echo(f"Dev tweets: {len(df)}")
    click.echo(f"Total tokens: {total}")
    click.echo(f"Avg tokens per tweet: {avg:.2f}")
    click.echo(f"Median: {median:.1f}  P95: {p95:.1f}  Std: {std:.2f}")
    click.echo(f"Min: {mn}  Max: {mx}")

    # also estimate full prompt tokens (system + few-shot + tweet)
    # rough: system prompt tokens
    sys_tokens = count_tokens(SYSTEM_PROMPT, enc)
    click.echo(f"System prompt tokens: {sys_tokens}")
    click.echo(f"Avg tweet + system (no few-shot): {avg + sys_tokens:.1f}")
    return {
        "encoding": encoding_name,
        "count": len(df),
        "total": total,
        "avg": avg,
        "median": median,
        "p95": p95,
        "std": std,
        "min": mn,
        "max": mx,
        "system_tokens": sys_tokens,
    }


def load_few_shot_examples(
    train_path: str, num_shots: int = 3, seed: int = 42
) -> list[dict]:
    """Load stratified few-shot examples from train set.

    Picks ``num_shots`` examples stratified across negative/neutral/positive.
    """
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if num_shots < 1 or num_shots > 12:
        raise click.BadParameter("num-shots must be between 1 and 12")

    df = pd.read_csv(train_path)
    if "text" not in df.columns or "sentiment" not in df.columns:
        raise ValueError("Train CSV must contain 'text' and 'sentiment'")

    sentiments = ["negative", "neutral", "positive"]
    per = num_shots // len(sentiments)
    rem = num_shots % len(sentiments)

    examples: list[dict] = []
    for i, sent in enumerate(sentiments):
        n = per + (1 if i < rem else 0)
        if n == 0:
            continue
        sub = df[df["sentiment"] == sent]
        if sub.empty:
            continue
        n = min(n, len(sub))
        sampled = sub.sample(n=n, random_state=seed + i)
        for _, row in sampled.iterrows():
            examples.append(
                {
                    "text": str(row["text"]),
                    "sentiment": str(row["sentiment"]),
                    "valence": float(row["score_compound"]),
                }
            )
    # deterministic order: negative, neutral, positive as sampled
    # shuffle optional but keep reproducible
    return examples


def build_messages(
    tweet_text: str, few_shot_examples: list[dict], system_prompt: str = SYSTEM_PROMPT
) -> list[dict]:
    """Build OpenAI chat messages for a single tweet."""
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    for ex in few_shot_examples:
        messages.append({"role": "user", "content": f"Tweet: {ex['text']}"})
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {"sentiment": ex["sentiment"], "valence": ex["valence"]}
                ),
            }
        )
    messages.append({"role": "user", "content": f"Tweet: {tweet_text}"})
    return messages


def get_openai_client(base_url: str | None = None):
    """Create a regular OpenAI client from environment variables.

    Env vars:
      - OPENAI_API_KEY (required)
      - OPENAI_BASE_URL / OPENAI_API_BASE (optional, overrides ``base_url``;
        defaults to the official OpenAI API endpoint)
    Returns None if credentials missing.
    """
    if openai is None:
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    base = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")

    if not api_key:
        return None
    try:
        kwargs: dict = {"api_key": api_key}
        if base:
            kwargs["base_url"] = base
        return openai.OpenAI(**kwargs)
    except Exception as e:  # noqa: BLE001
        click.echo(f"Failed to create OpenAI client: {e}", err=True)
        return None


def parse_model_output(content: str) -> tuple[str, float]:
    """Parse model JSON output to (sentiment, valence).

    Handles markdown fences and surrounding text.
    """
    text = content.strip()
    # strip markdown code fences
    if "```" in text:
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            text = m.group(1)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise ValueError(f"Could not parse JSON from: {text[:300]}")
        data = json.loads(m.group(0))

    sentiment = str(data.get("sentiment", data.get("label", ""))).lower().strip()
    if sentiment not in {"negative", "neutral", "positive"}:
        if "pos" in sentiment:
            sentiment = "positive"
        elif "neg" in sentiment:
            sentiment = "negative"
        elif "neu" in sentiment:
            sentiment = "neutral"
        else:
            raise ValueError(f"Invalid sentiment: {sentiment!r} in {content[:300]}")

    raw_val = data.get("valence", data.get("score", data.get("score_compound", 0)))
    valence = float(raw_val)
    valence = max(-1.0, min(1.0, valence))
    return sentiment, valence


def _compute_rmse(y_true, y_pred) -> float:
    if _HAS_RMSE:
        return float(root_mean_squared_error(y_true, y_pred))
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# ---------------------------------------------------------------------------
# Batch API helpers
# ---------------------------------------------------------------------------


def build_chat_body(model: str, messages: list[dict]) -> dict:
    """Build the ``body`` payload for one ``/v1/chat/completions`` request."""
    body: dict = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }
    return body


def build_batch_request_line(custom_id: str, model: str, messages: list[dict]) -> dict:
    """Build one JSONL line for the Batch input file."""
    return {
        "custom_id": custom_id,
        "method": BATCH_METHOD,
        "url": BATCH_ENDPOINT,
        "body": build_chat_body(model, messages),
    }


def custom_id_for_row(position: int) -> str:
    """Return a unique ``custom_id`` for the tweet at ``position``."""
    return f"tweet-{position}"


def write_batch_input_file(
    df: pd.DataFrame,
    few_shot_examples: list[dict],
    model: str,
    output_path: str,
    system_prompt: str = SYSTEM_PROMPT,
) -> dict:
    """Write one batch request per row of ``df`` as JSONL.

    Returns a manifest mapping ``custom_id`` -> row metadata (original index,
    ``id`` column when present, tweet text and true labels when present) so
    batch results can be joined back without re-reading the CSV.
    """
    manifest: dict = {}
    with open(output_path, "w", encoding="utf-8") as f:
        for position, (idx, row) in enumerate(df.iterrows()):
            tweet = str(row["text"])
            messages = build_messages(tweet, few_shot_examples, system_prompt)
            custom_id = custom_id_for_row(position)
            line = build_batch_request_line(custom_id, model, messages)
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            entry: dict = {
                "position": position,
                "index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
                "id": row.get("id", idx),
                "text": tweet,
            }
            # Convert numpy/pandas scalars to plain python types for JSON.
            if "sentiment" in row and pd.notna(row["sentiment"]):
                entry["true_sentiment"] = str(row["sentiment"])
            if "score_compound" in row and pd.notna(row["score_compound"]):
                entry["true_score"] = float(row["score_compound"])
            manifest[custom_id] = entry
            # JSON-serializability guard for the ``id`` field.
            if isinstance(entry["id"], (np.integer,)):
                entry["id"] = int(entry["id"])
            elif isinstance(entry["id"], (np.floating,)):
                entry["id"] = float(entry["id"])
    return manifest


def save_manifest(manifest: dict, path: str) -> None:
    """Persist the ``custom_id`` manifest as JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def load_manifest(path: str) -> dict:
    """Load a ``custom_id`` manifest written by :func:`save_manifest`."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Manifest must be a JSON object: {path}")
    return data


def _batch_field(batch, name: str, default=None):
    """Read ``name`` from a Batch object (SDK model) or plain dict."""
    if isinstance(batch, dict):
        return batch.get(name, default)
    return getattr(batch, name, default)


def submit_batch(
    client,
    input_path: str,
    completion_window: str = DEFAULT_COMPLETION_WINDOW,
    metadata: dict | None = None,
):
    """Upload ``input_path`` and create a Batch; return the Batch object."""
    with open(input_path, "rb") as fh:
        batch_input_file = client.files.create(file=fh, purpose="batch")
    input_file_id = _batch_field(batch_input_file, "id")
    click.echo(f"Uploaded batch input file: {input_file_id}")
    batch = client.batches.create(
        input_file_id=input_file_id,
        endpoint=BATCH_ENDPOINT,
        completion_window=completion_window,
        metadata=metadata or {"description": "twitter llm-baseline"},
    )
    return batch


def poll_batch(
    client,
    batch_id: str,
    poll_interval: float = 60.0,
    timeout: float = 86400.0,
):
    """Poll ``batches.retrieve`` until the batch reaches a terminal status."""
    started = time.monotonic()
    while True:
        batch = client.batches.retrieve(batch_id)
        status = _batch_field(batch, "status")
        counts = _batch_field(batch, "request_counts", {})
        if not isinstance(counts, dict):
            try:
                counts = counts.model_dump()  # pydantic model
            except Exception:  # noqa: BLE001
                counts = {}
        click.echo(f"Batch {batch_id} status={status} counts={counts}")
        if status in TERMINAL_BATCH_STATUSES:
            return batch
        if timeout > 0 and (time.monotonic() - started) > timeout:
            raise TimeoutError(
                f"Timed out waiting for batch {batch_id} after {timeout:.0f}s. "
                f"Resume later with --batch-id {batch_id}."
            )
        time.sleep(max(poll_interval, 1.0))


def _file_text(client, file_id: str) -> str:
    """Download a file via the Files API and return its text content."""
    resp = client.files.content(file_id)
    for attr in ("text",):
        val = getattr(resp, attr, None)
        if isinstance(val, str):
            return val
    if hasattr(resp, "read"):
        data = resp.read()
        return data.decode("utf-8") if isinstance(data, bytes) else str(data)
    content = getattr(resp, "content", None)
    if isinstance(content, bytes):
        return content.decode("utf-8")
    if isinstance(content, str):
        return content
    raise ValueError(f"Could not read content of file {file_id}: {type(resp)}")


def download_batch_output(client, output_file_id: str, dest_path: str) -> list[dict]:
    """Download the batch output JSONL and return parsed lines.

    Also writes the raw JSONL to ``dest_path`` for provenance / resume.
    """
    text = _file_text(client, output_file_id)
    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(text)
    lines: list[dict] = []
    for raw in text.splitlines():
        raw = raw.strip()
        if raw:
            lines.append(json.loads(raw))
    return lines


def extract_content_from_result(result: dict) -> str:
    """Extract the assistant message content from one batch output line.

    Raises ``ValueError`` for failed / errored requests (including expired).
    """
    custom_id = result.get("custom_id", "?")
    error = result.get("error")
    response = result.get("response") or {}
    if error is not None:
        raise ValueError(f"Request {custom_id} failed: {error}")
    status_code = response.get("status_code")
    if status_code != 200:
        raise ValueError(
            f"Request {custom_id} bad status {status_code}: {response!r}"[:500]
        )
    body = response.get("body") or {}
    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(
            f"Request {custom_id} has no chat content: {body!r}"[:500]
        ) from e


def collect_batch_predictions(
    manifest: dict, output_lines: list[dict]
) -> tuple[list, list, list, list, list, list, int]:
    """Join batch output lines back to tweets and parse predictions.

    Returns ``(y_true_clf, y_pred_clf, y_true_reg, y_pred_reg, raw_outputs,
    llm_responses, num_errors)``. Failed requests and unparsable outputs
    fall back to ``("neutral", 0.0)`` and count towards ``num_errors``.
    Missing ``custom_id`` entries (e.g. expired requests only present in the
    error file) are also recorded as errors when present in the manifest.
    """
    by_id = {line.get("custom_id"): line for line in output_lines}
    y_true_clf: list[int] = []
    y_pred_clf: list[int] = []
    y_true_reg: list[float] = []
    y_pred_reg: list[float] = []
    raw_outputs: list[dict] = []
    llm_responses: list[dict] = []
    num_errors = 0

    for custom_id, entry in manifest.items():
        line = by_id.get(custom_id)
        if line is None:
            content: str | None = None
            sentiment, valence = "neutral", 0.0
            num_errors += 1
            click.echo(
                f"{custom_id} missing from batch output — using fallback.", err=True
            )
        else:
            try:
                content = extract_content_from_result(line)
                sentiment, valence = parse_model_output(content)
            except Exception as e:  # noqa: BLE001
                click.echo(f"{custom_id} result error: {e}", err=True)
                sentiment, valence = "neutral", 0.0
                content = f"ERROR: {e}"
                num_errors += 1

        if "true_sentiment" in entry:
            true_sent = str(entry["true_sentiment"])
            if true_sent.isdigit() or (true_sent.lstrip("-").isdigit()):
                true_sent_id = int(true_sent)
            else:
                true_sent_id = LABEL_CODING.get(true_sent, 1)
            y_true_clf.append(true_sent_id)
            y_pred_clf.append(LABEL_CODING[sentiment])
        if "true_score" in entry:
            y_true_reg.append(float(entry["true_score"]))
            y_pred_reg.append(float(valence))

        raw_outputs.append(
            {
                "id": entry.get("id"),
                "text": entry.get("text", ""),
                "pred_sentiment": sentiment,
                "pred_valence": valence,
                "raw": content,
            }
        )
        llm_responses.append({"id": entry.get("id"), "llm_response": content})

    return (
        y_true_clf,
        y_pred_clf,
        y_true_reg,
        y_pred_reg,
        raw_outputs,
        llm_responses,
        num_errors,
    )


@click.command()
@click.option(
    "--dev-path", default=None, help="Path to dev CSV (overrides --splits-dir/asset)"
)
@click.option(
    "--train-path",
    default=None,
    help="Path to train CSV (overrides --splits-dir/asset)",
)
@click.option(
    "--splits-dir",
    default=None,
    type=click.Path(exists=False, file_okay=False),
    help="Mounted twitter-splits uri_folder (e.g. ${{inputs.splits}}); "
    "contains tweets_dev.csv / tweets_train.csv",
)
@click.option(
    "--resource-group", default=None, help="Azure resource group (asset load)"
)
@click.option("--workspace", default=None, help="Azure ML workspace (asset load)")
@click.option(
    "--subscription-id",
    default=lambda: os.environ.get("AZURE_SUBSCRIPTION_ID"),
    help="Azure subscription id (defaults to $AZURE_SUBSCRIPTION_ID)",
)
@click.option("--data-name", default=DEFAULT_DATA_NAME, show_default=True)
@click.option("--data-version", default=DEFAULT_DATA_VERSION, show_default=True)
@click.option("--model", default=DEFAULT_MODEL, help="Model name for OpenAI API")
@click.option(
    "--base-url",
    default=None,
    help="Optional custom OpenAI base URL (overrides OPENAI_BASE_URL)",
)
@click.option(
    "--encoding", default=DEFAULT_ENCODING, help="tiktoken encoding (e.g. o200k_base)"
)
@click.option(
    "--num-shots",
    default=3,
    type=int,
    help="Number of few-shot examples (3-6 recommended, stratified)",
)
@click.option(
    "--limit", default=None, type=int, help="Limit number of dev tweets (for testing)"
)
@click.option(
    "--estimate-tokens",
    is_flag=True,
    help="Only estimate tokens for dev tweets and exit (no API call)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Dry run: print batch request lines instead of calling API",
)
@click.option(
    "--completion-window",
    default=DEFAULT_COMPLETION_WINDOW,
    show_default=True,
    help="Batch completion window (Batch API currently only supports 24h)",
)
@click.option(
    "--poll-interval",
    default=60.0,
    type=float,
    show_default=True,
    help="Seconds between batch status polls",
)
@click.option(
    "--poll-timeout",
    default=86400.0,
    type=float,
    show_default=True,
    help="Max seconds to wait for batch completion (0 = submit only)",
)
@click.option(
    "--no-wait",
    is_flag=True,
    help="Submit the batch and exit without waiting for results",
)
@click.option(
    "--batch-id",
    default=None,
    help="Resume: fetch results for an existing batch instead of submitting",
)
@click.option(
    "--manifest-path",
    default=None,
    type=click.Path(exists=False),
    help="Path to manifest.json for --batch-id resume (defaults to <workdir>/manifest.json)",
)
@click.option(
    "--workdir",
    default=None,
    type=click.Path(file_okay=False),
    help="Directory to persist batch_input.jsonl / manifest.json / batch_output.jsonl for resume",
)
def main(
    dev_path,
    train_path,
    splits_dir,
    resource_group,
    workspace,
    subscription_id,
    data_name,
    data_version,
    model,
    base_url,
    encoding,
    num_shots,
    limit,
    estimate_tokens,
    dry_run,
    completion_window,
    poll_interval,
    poll_timeout,
    no_wait,
    batch_id,
    manifest_path,
    workdir,
):
    """LLM baseline: few-shot sentiment + valence via OpenAI Batch API on dev set."""
    dev_path, train_path, provenance = resolve_split_paths(
        dev_path,
        train_path,
        splits_dir,
        resource_group,
        workspace,
        subscription_id,
        data_name,
        data_version,
    )
    if splits_dir and not os.path.isdir(splits_dir):
        raise click.ClickException(f"--splits-dir not found: {splits_dir}")

    # --estimate-tokens path
    if estimate_tokens:
        estimate_token_counts(dev_path, encoding)
        if not dry_run:
            return

    # Resume mode needs the manifest that maps custom_id -> tweet.
    if batch_id is not None and manifest_path is None and workdir is not None:
        manifest_path = os.path.join(workdir, "manifest.json")

    # Load dev data (not needed when resuming purely from a manifest, but
    # still useful for logging — only require it for the submit path).
    df_dev: pd.DataFrame | None = None
    dev_counts = None
    enc = get_encoding(encoding)
    if batch_id is None:
        df_dev = read_split_csv(dev_path, "Dev")
        if "text" not in df_dev.columns:
            raise click.ClickException("Dev CSV missing 'text' column")

        # Token stats for logging (always show briefly)
        dev_counts = df_dev["text"].astype(str).apply(lambda x: count_tokens(x, enc))
        click.echo(
            f"Dev tokens ({encoding}): total={int(dev_counts.sum())} "
            f"avg={dev_counts.mean():.1f} max={int(dev_counts.max())} "
            f"min={int(dev_counts.min())}"
        )
        if limit is not None:
            df_dev = df_dev.iloc[:limit]

    # Few-shot examples
    try:
        few_shot = load_few_shot_examples(train_path, num_shots=num_shots)
    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from e

    click.echo(
        f"System prompt ({len(SYSTEM_PROMPT)} chars, ~{count_tokens(SYSTEM_PROMPT, enc)} tokens):"
    )
    click.echo(SYSTEM_PROMPT[:500] + ("..." if len(SYSTEM_PROMPT) > 500 else ""))
    click.echo(f"\nFew-shot examples ({len(few_shot)} stratified):")
    for i, ex in enumerate(few_shot, 1):
        preview = ex["text"][:100].replace("\n", " ")
        click.echo(f"  {i}. [{ex['sentiment']}, {ex['valence']:.3f}] {preview}...")

    # Dry-run handling (preview the batch input, no client needed)
    if dry_run and batch_id is None:
        click.echo("\n--- DRY RUN: batch request preview (no API call) ---")
        assert df_dev is not None
        n_show = limit if limit is not None else 3
        n_show = min(n_show, len(df_dev))
        for pos in range(n_show):
            tweet = str(df_dev.iloc[pos]["text"])
            messages = build_messages(tweet, few_shot, SYSTEM_PROMPT)
            line = build_batch_request_line(custom_id_for_row(pos), model, messages)
            prompt_tokens = count_tokens(" ".join(m["content"] for m in messages), enc)
            click.echo(
                f"\nLine {pos + 1}/{n_show} custom_id={line['custom_id']} "
                f"(prompt ~{prompt_tokens} tokens):"
            )
            click.echo(json.dumps(line, indent=2, ensure_ascii=False)[:3000])
            if len(json.dumps(line)) > 3000:
                click.echo("... (truncated)")
        click.echo(
            "\nBatch API: 50% lower cost vs sync API, separate higher rate "
            "limits, 24h turnaround. Submit with live mode; use --no-wait to "
            "submit only and --batch-id to fetch results later."
        )
        click.echo(f"\nDry run done. Would submit {len(df_dev)} tweets as one batch.")
        click.echo("Set OPENAI_API_KEY to submit live batches.")
        return

    client = None
    if batch_id is None:
        client = get_openai_client(base_url=base_url)
        if client is None:
            click.echo(
                "No OpenAI credentials found (OPENAI_API_KEY). "
                "Switching to dry-run mode (print batch lines instead of calling API).",
                err=True,
            )
            # Fall through to batch-line preview without submitting.
            assert df_dev is not None
            n_show = limit if limit is not None else 3
            n_show = min(n_show, len(df_dev))
            for pos in range(n_show):
                tweet = str(df_dev.iloc[pos]["text"])
                messages = build_messages(tweet, few_shot, SYSTEM_PROMPT)
                line = build_batch_request_line(custom_id_for_row(pos), model, messages)
                click.echo(json.dumps(line, ensure_ascii=False)[:1000])
            return
    else:
        # Resume path still needs a client to fetch results.
        client = get_openai_client(base_url=base_url)
        if client is None:
            raise click.ClickException(
                "OPENAI_API_KEY is required to fetch batch results."
            )
    assert client is not None

    # Live mode (MLflow tracking enabled; dry-run above stays fully local)
    try:
        import mlflow
    except ImportError as e:
        raise click.ClickException(
            "MLflow is required for live runs (pip install -e '.[llm]'). "
            "Use --dry-run for local runs without MLflow."
        ) from e
    _setup_mlflow_tracking()
    with mlflow.start_run(run_name=MLFLOW_RUN_NAME):
        mlflow.log_params(
            {
                "model": model,
                "num_shots": num_shots,
                "limit": limit if limit is not None else -1,
                "encoding": encoding,
                "dev_path": dev_path,
                "train_path": train_path,
                "splits_dir": splits_dir or "",
                "data_name": provenance.get("data_name", ""),
                "data_version": provenance.get("data_version", ""),
                "dev_source": provenance.get("dev", ""),
                "train_source": provenance.get("train", ""),
                "base_url": base_url
                or os.getenv("OPENAI_BASE_URL")
                or os.getenv("OPENAI_API_BASE")
                or "",
                "endpoint": BATCH_ENDPOINT,
                "completion_window": completion_window,
                "poll_interval": poll_interval,
                "poll_timeout": poll_timeout,
                "no_wait": bool(no_wait),
            }
        )
        mlflow.set_tags({"baseline": "llm", "model": model, "api": "batch"})
        mlflow.log_text(SYSTEM_PROMPT, "system_prompt.txt")
        mlflow.log_text(
            json.dumps(few_shot, indent=2, ensure_ascii=False),
            "few_shot_examples.json",
        )
        if dev_counts is not None:
            mlflow.log_params(
                {
                    "dev_rows_total": len(df_dev) if df_dev is not None else -1,
                    "dev_total_tokens": int(dev_counts.sum()),
                    "dev_avg_tokens": float(dev_counts.mean()),
                    "system_prompt_tokens": count_tokens(SYSTEM_PROMPT, enc),
                }
            )

        tmpdir_ctx: tempfile.TemporaryDirectory | None = None
        persist_dir = workdir
        if persist_dir is not None:
            os.makedirs(persist_dir, exist_ok=True)
            batch_input_path = os.path.join(persist_dir, "batch_input.jsonl")
            local_manifest_path = os.path.join(persist_dir, "manifest.json")
            batch_output_path = os.path.join(persist_dir, "batch_output.jsonl")
        else:
            tmpdir_ctx = tempfile.TemporaryDirectory()
            batch_input_path = os.path.join(tmpdir_ctx.name, "batch_input.jsonl")
            local_manifest_path = os.path.join(tmpdir_ctx.name, "manifest.json")
            batch_output_path = os.path.join(tmpdir_ctx.name, "batch_output.jsonl")

        try:
            manifest: dict
            if batch_id is None:
                assert df_dev is not None
                n_rows = len(df_dev)
                if n_rows == 0:
                    raise click.ClickException("No dev rows to submit.")
                if n_rows > 50000:
                    raise click.ClickException(
                        f"Batch API supports at most 50,000 requests per batch "
                        f"(got {n_rows}). Use --limit to split the workload."
                    )
                click.echo(
                    f"\nBuilding batch input for {n_rows} tweets "
                    f"(endpoint={BATCH_ENDPOINT}, model={model}) ..."
                )
                manifest = write_batch_input_file(
                    df_dev, few_shot, model, batch_input_path
                )
                save_manifest(manifest, local_manifest_path)
                # mlflow.log_artifact(batch_input_path)
                mlflow.log_artifact(local_manifest_path)
                mlflow.log_param("num_requests", n_rows)

                batch = submit_batch(client, batch_input_path, completion_window)
                batch_id = _batch_field(batch, "id")
                input_file_id = _batch_field(batch, "input_file_id")
                click.echo(f"Created batch {batch_id} (input_file={input_file_id}).")
                mlflow.log_params(
                    {"batch_id": batch_id or "", "input_file_id": input_file_id or ""}
                )
            else:
                # Resume: load the manifest from --manifest-path / workdir.
                resolved_manifest = manifest_path or local_manifest_path
                if not resolved_manifest or not os.path.exists(resolved_manifest):
                    raise click.ClickException(
                        f"Manifest not found: {resolved_manifest}. "
                        "Pass --manifest-path (or --workdir containing manifest.json) "
                        "from the submit step so custom_ids can be mapped back."
                    )
                manifest = load_manifest(resolved_manifest)
                click.echo(
                    f"Resuming batch {batch_id} with {len(manifest)} requests "
                    f"(manifest={resolved_manifest})."
                )
                mlflow.log_params({"batch_id": batch_id})
                try:
                    mlflow.log_artifact(resolved_manifest)
                except Exception:  # noqa: BLE001
                    pass

            submit_only = bool(no_wait) or (
                poll_timeout is not None and poll_timeout <= 0
            )
            if submit_only:
                click.echo(
                    f"\nSubmitted batch {batch_id} (--no-wait). Fetch results later with:\n"
                    f"  python -m twitter.baselines.llm_baseline "
                    f"--batch-id {batch_id} --manifest-path {local_manifest_path}"
                    + (f" --dev-path {dev_path}" if batch_id else "")
                )
                return

            click.echo(
                f"\nWaiting for batch {batch_id} (poll every {poll_interval}s) ..."
            )
            final_batch = poll_batch(
                client, batch_id, poll_interval=poll_interval, timeout=poll_timeout
            )
            status = _batch_field(final_batch, "status")
            output_file_id = _batch_field(final_batch, "output_file_id")
            error_file_id = _batch_field(final_batch, "error_file_id")
            click.echo(
                f"Batch {batch_id} finished with status={status} "
                f"output_file={output_file_id} error_file={error_file_id}."
            )
            mlflow.log_params(
                {
                    "batch_status": status or "",
                    "output_file_id": output_file_id or "",
                    "error_file_id": error_file_id or "",
                }
            )
            request_counts = _batch_field(final_batch, "request_counts", {}) or {}
            if not isinstance(request_counts, dict):
                try:
                    request_counts = request_counts.model_dump()
                except Exception:  # noqa: BLE001
                    request_counts = {}
            for k, v in request_counts.items():
                try:
                    mlflow.log_metric(f"batch_{k}", float(v))
                except Exception:  # noqa: BLE001
                    pass

            if status != "completed" and output_file_id is None:
                raise click.ClickException(
                    f"Batch {batch_id} ended with status={status} and no output file. "
                    f"Inspect error_file={error_file_id} in the OpenAI dashboard."
                )
            if output_file_id is None:
                raise click.ClickException(
                    f"Batch {batch_id} has no output file (status={status})."
                )

            output_lines = download_batch_output(
                client, output_file_id, batch_output_path
            )
            click.echo(f"Downloaded {len(output_lines)} batch result lines.")
            mlflow.log_artifact(batch_output_path)

            (
                y_true_clf,
                y_pred_clf,
                y_true_reg,
                y_pred_reg,
                raw_outputs,
                llm_responses,
                num_errors,
            ) = collect_batch_predictions(manifest, output_lines)

            # Log predictions and raw LLM responses to MLflow only (no local copy
            # beyond --workdir).
            if persist_dir is not None:
                predictions_path = os.path.join(persist_dir, "predictions.csv")
                llm_log_path = os.path.join(persist_dir, "llm_responses.csv")
                pd.DataFrame(raw_outputs).to_csv(predictions_path, index=False)
                pd.DataFrame(llm_responses, columns=["id", "llm_response"]).to_csv(
                    llm_log_path, index=False
                )
                mlflow.log_artifact(predictions_path)
                mlflow.log_artifact(llm_log_path)
            else:
                with tempfile.TemporaryDirectory() as tmpdir:
                    llm_log_path = os.path.join(tmpdir, "llm_responses.csv")
                    pd.DataFrame(llm_responses, columns=["id", "llm_response"]).to_csv(
                        llm_log_path, index=False
                    )
                    mlflow.log_artifact(llm_log_path)

                    predictions_path = os.path.join(tmpdir, "predictions.csv")
                    pd.DataFrame(raw_outputs).to_csv(predictions_path, index=False)
                    mlflow.log_artifact(predictions_path)
            click.echo("Logged predictions and LLM responses to MLflow.")

            mlflow.log_metric("num_rows", float(len(manifest)))
            mlflow.log_metric("num_output_lines", float(len(output_lines)))
            mlflow.log_metric("num_errors", float(num_errors))

            # Evaluate
            if y_true_clf:
                macro_f1 = f1_score(y_true_clf, y_pred_clf, average="macro")
                click.echo(f"Classification Macro F1 (dev): {macro_f1:.4f}")
                from sklearn.metrics import classification_report

                report = classification_report(
                    y_true_clf, y_pred_clf, digits=4, zero_division=0
                )
                click.echo(report)
                mlflow.log_metric("macro_f1", float(macro_f1))
                mlflow.log_text(report, "classification_report.txt")
            else:
                click.echo("No classification labels for eval.")

            if y_true_reg:
                rmse = _compute_rmse(np.array(y_true_reg), np.array(y_pred_reg))
                click.echo(f"Regression RMSE (dev): {rmse:.4f}")
                mlflow.log_metric("rmse", float(rmse))
            else:
                click.echo("No regression labels for eval.")

            click.echo("Done.")
            # NOTE: no model to log or register — this baseline performs no
            # training, only few-shot inference.
        finally:
            if tmpdir_ctx is not None:
                tmpdir_ctx.cleanup()


if __name__ == "__main__":
    main()
