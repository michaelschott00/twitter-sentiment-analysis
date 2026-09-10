"""LLM baseline for Twitter sentiment analysis via Microsoft Foundry (Azure OpenAI).

Uses few-shot prompting to obtain both classification (sentiment) and
regression (valence) in a single prompt. Supports token estimation with
tiktoken and dry-run mode when credentials are missing.
"""

import json
import os
import re

import click
import numpy as np
import pandas as pd
import tiktoken

try:
    import openai
except ImportError:
    openai = None  # type: ignore[assignment]

from sklearn.metrics import f1_score, mean_squared_error
from tqdm import tqdm

try:
    from sklearn.metrics import root_mean_squared_error

    _HAS_RMSE = True
except ImportError:
    _HAS_RMSE = False

from twitter.labels import LABEL_CODING

DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_ENCODING = "o200k_base"
DEFAULT_API_VERSION = "2024-12-01-preview"

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


def get_encoding(encoding_name: str = DEFAULT_ENCODING):
    """Return tiktoken encoding for token counting.

    Tries ``get_encoding`` first, falls back to ``encoding_for_model``.
    """
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception:  # noqa: BLE001
        try:
            return tiktoken.encoding_for_model("gpt-5")
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


def get_openai_client():
    """Create OpenAI / AzureOpenAI client from environment variables.

    Env vars:
      - AZURE_OPENAI_ENDPOINT / AZURE_AI_FOUNDRY_ENDPOINT / OPENAI_API_BASE
      - AZURE_OPENAI_API_KEY / AZURE_OPENAI_KEY / OPENAI_API_KEY
      - AZURE_OPENAI_API_VERSION (default 2024-12-01-preview)
    Returns None if credentials missing.
    """
    if openai is None:
        return None

    endpoint = (
        os.getenv("AZURE_OPENAI_ENDPOINT")
        or os.getenv("AZURE_AI_FOUNDRY_ENDPOINT")
        or os.getenv("OPENAI_API_BASE")
        or os.getenv("OPENAI_BASE_URL")
    )
    api_key = (
        os.getenv("AZURE_OPENAI_API_KEY")
        or os.getenv("AZURE_OPENAI_KEY")
        or os.getenv("AZURE_AI_FOUNDRY_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", DEFAULT_API_VERSION)

    if endpoint and api_key:
        # Detect Azure endpoint vs generic OpenAI-compatible
        low = endpoint.lower()
        if "openai.azure.com" in low or "azure" in low:
            try:
                return openai.AzureOpenAI(
                    azure_endpoint=endpoint, api_key=api_key, api_version=api_version
                )
            except Exception as e:  # noqa: BLE001
                click.echo(f"Failed to create AzureOpenAI client: {e}", err=True)
                return None
        else:
            try:
                return openai.OpenAI(api_key=api_key, base_url=endpoint)
            except Exception as e:  # noqa: BLE001
                click.echo(
                    f"Failed to create OpenAI client with base_url: {e}", err=True
                )
                return None
    elif api_key:
        try:
            return openai.OpenAI(api_key=api_key)
        except Exception as e:  # noqa: BLE001
            click.echo(f"Failed to create OpenAI client: {e}", err=True)
            return None
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


def call_llm(client, model: str, messages: list[dict], temperature: float = 0.0) -> str:
    """Call chat completions and return content string."""
    # Try with json response_format for models that support it
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    # gpt-5.x supports response_format json_object
    if "gpt-5" in model or "gpt-4" in model:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception:
        # retry without response_format if it failed
        if "response_format" in kwargs:
            kwargs.pop("response_format")
            resp = client.chat.completions.create(**kwargs)
        else:
            raise
    return resp.choices[0].message.content  # type: ignore[union-attr]


@click.command()
@click.option(
    "--dev-path", default="data/splits/tweets_dev.csv", help="Path to dev CSV"
)
@click.option(
    "--train-path", default="data/splits/tweets_train.csv", help="Path to train CSV"
)
@click.option("--model", default=DEFAULT_MODEL, help="Model name for Azure OpenAI")
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
    "--temperature",
    default=0.0,
    type=float,
    help="LLM temperature (0 for deterministic)",
)
@click.option(
    "--output",
    default=None,
    type=str,
    help="Path to save predictions CSV (optional)",
)
@click.option(
    "--estimate-tokens",
    is_flag=True,
    help="Only estimate tokens for dev tweets and exit (no API call)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Dry run: print prompts instead of calling API",
)
def main(
    dev_path,
    train_path,
    model,
    encoding,
    num_shots,
    limit,
    temperature,
    output,
    estimate_tokens,
    dry_run,
):
    """LLM baseline: few-shot sentiment + valence via Azure OpenAI on dev set."""
    # --estimate-tokens path
    if estimate_tokens:
        estimate_token_counts(dev_path, encoding)
        if not dry_run:
            return

    # Load dev data
    if not os.path.exists(dev_path):
        raise click.ClickException(f"Dev file not found: {dev_path}")
    df_dev = pd.read_csv(dev_path)
    if "text" not in df_dev.columns:
        raise click.ClickException("Dev CSV missing 'text' column")

    # Token stats for logging (always show briefly)
    enc = get_encoding(encoding)
    dev_counts = df_dev["text"].astype(str).apply(lambda x: count_tokens(x, enc))
    click.echo(
        f"Dev tokens ({encoding}): total={int(dev_counts.sum())} "
        f"avg={dev_counts.mean():.1f} max={int(dev_counts.max())} "
        f"min={int(dev_counts.min())}"
    )

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

    # Dry-run handling
    client = None
    if not dry_run:
        client = get_openai_client()
        if client is None:
            click.echo(
                "No Azure/OpenAI credentials found (AZURE_OPENAI_ENDPOINT/OPENAI_API_KEY). "
                "Switching to dry-run mode (print prompts instead of calling API).",
                err=True,
            )
            dry_run = True

    if dry_run:
        click.echo("\n--- DRY RUN: prompt examples (no API call) ---")
        n_show = limit if limit is not None else 3
        n_show = min(n_show, len(df_dev))
        for idx in range(n_show):
            tweet = str(df_dev.iloc[idx]["text"])
            messages = build_messages(tweet, few_shot, SYSTEM_PROMPT)
            # token estimate for full prompt
            prompt_text = " ".join(m["content"] for m in messages)
            prompt_tokens = count_tokens(prompt_text, enc)
            click.echo(
                f"\nExample {idx + 1}/{n_show} (prompt ~{prompt_tokens} tokens):"
            )
            click.echo(json.dumps(messages, indent=2, ensure_ascii=False)[:3000])
            if len(json.dumps(messages)) > 3000:
                click.echo("... (truncated)")
            # show expected parsing
            click.echo(
                'Expected output JSON: {"sentiment": "positive|neutral|negative", "valence": 0.0}'
            )
        click.echo(
            f"\nDry run done. Would evaluate {len(df_dev) if limit is None else min(limit, len(df_dev))} tweets live."
        )
        click.echo(
            "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY to run live calls."
        )
        return

    # Live mode
    assert client is not None
    click.echo(
        f"\nCalling model {model} for {len(df_dev) if limit is None else limit} tweets ..."
    )
    if limit is not None:
        df_dev = df_dev.iloc[:limit]

    y_true_clf: list[int] = []
    y_pred_clf: list[int] = []
    y_true_reg: list[float] = []
    y_pred_reg: list[float] = []
    raw_outputs: list[dict] = []

    for idx, row in tqdm(df_dev.iterrows(), total=len(df_dev), desc="LLM eval"):
        tweet = str(row["text"])
        messages = build_messages(tweet, few_shot, SYSTEM_PROMPT)
        try:
            content = call_llm(client, model, messages, temperature=temperature)
            sentiment, valence = parse_model_output(content)
        except Exception as e:  # noqa: BLE001
            click.echo(f"Row {idx} API/parse error: {e}", err=True)
            # fallback to neutral 0
            sentiment, valence = "neutral", 0.0
            content = f"ERROR: {e}"

        # true labels
        if "sentiment" in row:
            true_sent = str(row["sentiment"])
            # handle encoded ints
            if true_sent.isdigit() or (true_sent.lstrip("-").isdigit()):
                true_sent_id = int(true_sent)
            else:
                true_sent_id = LABEL_CODING.get(true_sent, 1)
            pred_id = LABEL_CODING[sentiment]
            y_true_clf.append(true_sent_id)
            y_pred_clf.append(pred_id)
        if "score_compound" in row:
            y_true_reg.append(float(row["score_compound"]))
            y_pred_reg.append(float(valence))

        raw_outputs.append(
            {
                "id": row.get("id", idx),
                "text": tweet,
                "pred_sentiment": sentiment,
                "pred_valence": valence,
                "raw": content,
            }
        )

    # Evaluate
    if y_true_clf:
        macro_f1 = f1_score(y_true_clf, y_pred_clf, average="macro")
        click.echo(f"Classification Macro F1 (dev): {macro_f1:.4f}")
        from sklearn.metrics import classification_report

        click.echo(
            classification_report(y_true_clf, y_pred_clf, digits=4, zero_division=0)
        )
    else:
        click.echo("No classification labels for eval.")

    if y_true_reg:
        rmse = _compute_rmse(np.array(y_true_reg), np.array(y_pred_reg))
        click.echo(f"Regression RMSE (dev): {rmse:.4f}")
    else:
        click.echo("No regression labels for eval.")

    if output:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        out_df = pd.DataFrame(raw_outputs)
        out_df.to_csv(output, index=False)
        click.echo(f"Saved predictions to {output}")

    click.echo("Done.")


if __name__ == "__main__":
    main()
