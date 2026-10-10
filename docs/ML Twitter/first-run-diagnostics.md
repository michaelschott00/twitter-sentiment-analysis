---
date: 2026-09-20
---
# First training run — diagnostics & improvement plan

> Context: first end-to-end run of the single-task classification models plus the
> two baselines, evaluated on the dev set (`data/splits/tweets_dev.csv`, 1600 rows;
> 68% neutral / 26% positive / 5% negative). This is an optimization run, not a
> final comparison — the goal was to establish a starting point.
>
> Status: **not started**. The findings below are the backlog.

## Results under review

| Model | Macro F1 | Notes |
|---|---|---|
| Sentence-BERT-base | 0.7730 | best |
| BERTweet-large | 0.6710 | |
| LightGBM (TF-IDF) | 0.6172 | baseline |
| DeBERTa-v3-large | 0.6052 | |
| ModernBERT-large | 0.5942 | |
| GPT-5.6 Luna | 0.4540 | few-shot baseline (RMSE 0.484) |

Reference points: an all-neutral predictor scores **0.270** macro F1 on this dev set,
so the LLM is only ~0.18 above the majority class.

## Finding 1 — LR schedule never leaves warmup (root cause)

The resolved config logged by each run has `trainer.max_epochs: null`
(`configs/defaults.yaml:37`). Lightning silently treats that as **1000 epochs**, so
`trainer.estimated_stepping_batches = 200 steps/epoch × 1000 = 200,000`. With
`warmup_pct: 0.06` (`twitter/modules.py:30-65`) that is a **12,000-step warmup**.
Every run early-stopped between steps 2,599 and 3,999, i.e. while the LR was still
on the linear ramp:

| Run | Last step | Warmup fraction | Peak encoder LR reached (base) |
|---|---|---|---|
| bertweet-large | 2599 | 0.22 | ~2.2e-6 (1e-5) |
| deberta-v3-large | 2799 | 0.23 | ~1.4e-6 (6e-6) |
| modernbert-large | 3999 | 0.33 | ~3.3e-6 (1e-5) |
| sentence-bert-base | 3399 | 0.28 | ~8.5e-6 (3e-5) |

Implication: the encoders are barely fine-tuned and the models reduce to **linear
probes on frozen pretrained features**. That is why a mean-pooled NLI sentence
encoder (SBERT) wins — its features are the best for a linear head — and why the
ranking tracks the configured base LR. Do not trust the current ordering; it can
invert once the encoders actually train.

## Finding 2 — all four encoders overfit

Validation loss bottoms out at epoch 5–9 and diverges afterwards while macro F1 only
keeps inching (the F1 checkpoint is selected, so the best value is still used):

| Run | Val-loss min | Final val loss | Best F1 |
|---|---|---|---|
| sentence-bert-base | 0.517 @ ep7 | 1.306 | 0.773 @ ep14 |
| bertweet-large | 0.718 @ ep7 | 2.065 | 0.671 @ ep10 |
| deberta-v3-large | 0.798 @ ep9 | 1.394 | 0.605 @ ep11 |
| modernbert-large | 0.833 @ ep6 | 3.214 | 0.594 @ ep17 |

6.4k training examples, no augmentation, default dropout, `weight_decay=0.01`.
**Caveat:** fixing only the LR schedule (higher effective LR) will likely make
overfitting worse — schedule and regularization must move together.

## Finding 3 — LLM negative over-prediction (not a parsing bug)

`num_errors=0`, `batch_failed=0`, so this is real model behavior. Per-class report:

```
class 0 (negative): P 0.174  R 0.655  F1 0.275
class 1 (neutral):  P 0.859  R 0.398  F1 0.544
class 2 (positive): P 0.421  R 0.764  F1 0.542
accuracy 0.509
```

Negative precision 0.174 means it flags ~4x more negatives than exist. Likely cause is
the system prompt (`twitter/baselines/llm_baseline.py:95-111`), which hard-codes
`negative: valence ∈ [-1.0, -0.1)` / `neutral ≈ [-0.3, 0.3]`. The dataset's valence is
VADER and disagrees with the class label (46% of neutral train tweets have
|valence| > 0.3, min −0.913, max +0.922), so the LLM applies VADER logic instead of the
project's labeling convention. The few-shot examples are sampled from those
inconsistent rows, reinforcing the confusion.

## Recommended plan (prioritized)

1. **Fix the schedule horizon.** Set `max_epochs` per task config (e.g. 8–12) or derive
   `total_steps` from an explicit value instead of `estimated_stepping_batches`. Log the
   LR (`self.log("lr-AdamW", ...)`) so it is visible in MLflow.
2. **Add regularization** for the large encoders: dropout, higher `weight_decay`, label
   smoothing, and/or freeze-then-unfreeze (support already exists: `freeze_cfg` /
   `unfreeze_cfg` in `twitter/modules.py:254-261` and `twitter/models.py:118-124`, but no
   config uses them).
3. **Turn on augmentation** — `eda: false` and `external_data_paths: null` in
   `configs/data.yaml`; `twitter/augmentation.py` and `data/augmentation/` already exist.
   Biggest lever against overfitting.
4. **Log per-class metrics.** Currently only macro precision/recall/F1
   (`twitter/models.py:182-193`). Add `average=None` or a `classification_report`
   artifact so minority-class behavior is visible.
5. **Rework the LLM prompt.** Decouple sentiment from valence ranges, and/or use few-shot
   examples that demonstrate the dataset's (noisy) convention rather than VADER. Also
   re-evaluate whether 3 shots (1/class) is enough.
6. **Re-run the encoders after 1–3** and expect the ranking to change before drawing any
   conclusion about which encoder is best.

## Getting back up to speed

Experiment: `twitter-sentiment` in the Azure ML MLflow workspace (tracking URI comes from
`$MLFLOW_TRACKING_URI`). Inspect via the `mlflow_inspect` MCP tool:

```bash
python -I /opt/tools/mlflow_inspect.py list-experiments
python -I /opt/tools/mlflow_inspect.py list-runs --experiment twitter-sentiment
python -I /opt/tools/mlflow_inspect.py list-metrics --run <run_id>
python -I /opt/tools/mlflow_inspect.py history --run <run_id> --metric MulticlassF1Score
```

Run IDs from this analysis:

| Run | Run ID |
|---|---|
| clf-bertweet-large | `8ec63b3b-21fe-415d-ab6d-94949cf87a65` |
| clf-sentence-bert-base | `dd66741c-7f4f-40ef-8e06-b60a37981ef6` |
| clf-deberta-v3-large | `501b16f4-5a1f-440d-9296-5b9ddecd04fa` |
| clf-modernbert-large | `735bea8c-5b31-4532-bf2a-e503e17d19da` |
| llm-baseline | `0ee36177-684c-4f5b-bf6b-abe24668822f` |
| lightgbm-baseline-cpu | `placid_hair_0t1ph4q5l8` |

Each transformer run logs a resolved `config.yaml` artifact — download it to confirm the
trainer settings that were actually in effect, e.g.:

```bash
mlflow artifacts download --run-id <run_id> --artifact-path config.yaml --dst-path ./tmp
```

## Open questions

- Confirm the schedule is the dominant cause by re-running one encoder with
  `--trainer.max_epochs=10` and watching the LR/F1 curves.
- Does fixing the schedule + regularization let the large encoders beat SBERT?
- Is the encoder LR asymmetry (SBERT 3e-5 vs large 1e-5/6e-6) justified, or should it be
  normalized as a controlled variable?
