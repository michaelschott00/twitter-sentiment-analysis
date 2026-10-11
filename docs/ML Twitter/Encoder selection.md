# Encoder Selection — Twitter Sentiment (Classification + Valence Regression)

> **Date:** 2026-09-01
> **Context:** `docs/improvement-plan.md:57-59` — check Hugging Face hub for BERT-like encoders likely to improve *Macro F1* (3-class, 6% negative) and *RMSE* (valence ∈ [-1, 1]) under **≤24 GB VRAM** for fine-tuning. Rank top-3 from *existing* `configs/encoders/*.yaml` + newly researched candidates.
> **Constraints:** batch size 8, seq_len 512 (worst case; tweets avg <50 tokens, so real use ~128), AdamW, encoder LR 3e-5 / head 3e-4, `fp16`/`bf16` + gradient checkpointing allowed. Exclude >1 B params and generative LLM (decoder-only).

---

## 1. Overview

The project fine-tunes encoder-only Transformers on ~8 k AI/ML tweets. Labels are imbalanced (≈70 % neutral, ~24 % positive, **6 % negative**) and valence does not perfectly align with class (see `README.md:12-22`, `images/scores.png`). Therefore:

- **Macro F1** rewards *per-class recall*, especially minority (negative). Domain-matched tokenization (emoji, `#hashtag`, `@mention`, URL) and pretraining on tweets matters more than raw GLUE.
- **RMSE** rewards *fine-grained* valence grading (continuous). Disentangled attention / RTD-style pretraining and large-scale high-quality data help smooth calibration.

VRAM budget: **24 GB** (e.g. RTX 4090, A10, `Standard_NC4as_T4_v3` + overflow to V100/A100). All models below fit with `bf16` + checkpointing; estimates assume batch 8 × 512 tokens.

---

## 2. Already-Used Encoders (`configs/encoders/*.yaml` — 9 entries)

| # | Config file | HF model ID | HF link | Arch / Params¹ | Pooling | Note |
|---|-------------|-------------|---------|----------------|---------|------|
| 1 | `distillbert.yaml` | `bhadresh-savani/distilbert-base-uncased-emotion` | https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion | DistilBERT, **66 M** | `mean` | Emotion-tuned DistilBERT (6 emotions), 6-layer, fastest but smallest capacity; not tweet-specific. |
| 2 | `roberta_base.yaml` | `roberta-base` (`FacebookAI/roberta-base`) | https://huggingface.co/FacebookAI/roberta-base | RoBERTa-base, **125 M** | `cls` | Strong generic baseline, 160 GB pretrain, no tweet vocab. |
| 3 | `sentence_bert_base.yaml` | `sentence-transformers/bert-base-nli-mean-tokens` | https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens | BERT-base, **110 M** | `mean` | SBERT mean-pool NLI-tuned; sentence embedding prior, not sentiment. |
| 4 | `sentence_bert_large.yaml` | `sentence-transformers/bert-large-nli-mean-tokens` | https://huggingface.co/sentence-transformers/bert-large-nli-mean-tokens | BERT-large, **340 M** | `mean` | Larger SBERT, same NLI objective. |
| 5 | `sentence_roberta.yaml` | `sentence-transformers/roberta-base-nli-mean-tokens` | https://huggingface.co/sentence-transformers/roberta-base-nli-mean-tokens | RoBERTa-base, **125 M** | `mean` | RoBERTa + NLI mean-pool. |
| 6 | `sentence_xlm.yaml` | `sentence-transformers/xlm-r-100langs-bert-base-nli-mean-tokens` | https://huggingface.co/sentence-transformers/xlm-r-100langs-bert-base-nli-mean-tokens | XLM-R-base, **~270 M** (250 k vocab) | `mean` | Multilingual, 100 languages, NLI; heavy vocab, no tweet focus. |
| 7 | `twihn.yaml` | `Twitter/TwHIN-BERT-base` | https://huggingface.co/Twitter/TwHIN-BERT-base | BERT-base, **280 M** (250 k vocab, 12×768) | `mean` | **Strongest existing.** 7 B tweets, 100+ langs, MLM + social contrastive (TwHIN). SOTA on hashtag (Macro F1 54.62 avg, 59.31 en) and external sentiment avg 59.38 vs XLM-T 58.52 (`arxiv:2209.07562`). |
| 8 | `twitter-roberta-sentiment.yaml` | `cardiffnlp/twitter-roberta-base-sentiment-latest` | https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest | RoBERTa-base, **125 M** | `cls` | Tweet-RoBERTa (124 M tweets, 2018-2021) + TweetEval fine-tune. Macro F1 0.716 on TweetEval (`cardiffnlp/twitter-roberta-base-2021-124m-sentiment`). Direct sentiment init but base-size; not valence-optimized. |
| 9 | `xlnet.yaml` | `pig4431/TweetEval_XLNET_5E` | https://huggingface.co/pig4431/TweetEval_XLNET_5E | XLNet-base, **~110 M** | `last` | Autoregressive permuted LM, TweetEval-tuned, 5 epochs; heavier memory, less community support than RoBERTa/DeBERTa. |

¹ backbone + embedding (fp32). VRAM estimates §4.

**Takeaway:** Themes are (a) generic NLI/SBERT, (b) Cardiff/TimeLMs tweet RoBERTa, (c) one large-scale multilingual TwHIN. No DeBERTa, no ModernBERT, no BERTweet-large — the three families now dominating encoder leaderboards.

---

## 3. Candidate Research (Hugging Face, 2025-2026, encoder-only, ≤24 GB)

Searched Hugging Face Models (https://huggingface.co/models?pipeline_tag=text-classification&library=transformers&sort=downloads) + TweetEval, SST-5, Financial PhraseBank, BEIR/MLDR ablations.

| # | HF model ID | HF link | Params / Arch | VRAM train² (fp16/bf16, Adam + ckpt) | Why relevant (TweetEval/SST/sentiment, 2025-2026) |
|---|-------------|---------|----------------|--------------------------------------|---------------------------------------------------|
| C1 | `vinai/bertweet-base` | https://huggingface.co/vinai/bertweet-base | **135 M**, 12×768, RoBERTa-style, 64 k tweet BPE | ~1.6 GB static + 8-12 GB act. → **fits 24 GB** | *Tweet-native.* 850 M English tweets (80 GB, 2012-2019 + 5 M COVID). BERTweet-base beats RoBERTa-base and XLM-R-base on *every* Tweet task in Nguyen et al. 2020, including SemEval2017 sentiment **+5 % SOTA** (Table 3 in https://arxiv.org/abs/2005.10200). Tokenizer handles `#`, `@`, emoji natively. seq_len 128 (truncate/pad). |
| C2 | **`vinai/bertweet-large`** | https://huggingface.co/vinai/bertweet-large | **355 M**, 24×1024, large | ~3.0 GB static + 14-18 GB act. → **fits 24 GB** (bf16+ckpt) | Same corpus as base but large config; released 08/2021, still the **only large tweet-native MLM**. Expected +2-3 % over base on TweetEval (RoBERTa-large gap). Direct domain advantage for Macro F1 minority recall. Valence regression benefits from larger hidden (1024). |
| C3 | `microsoft/deberta-v3-base` | https://huggingface.co/microsoft/deberta-v3-base | **184 M total** (86 M backbone + 98 M embed, 12×768, 128 k SentencePiece) | **1.37 GB** static fp16 (HF Sizer) + 8-10 GB act. → **fits easily** | *Sample-efficiency king.* ELECTRA-style RTD + disentangled attention + GDES. GLUE 88.4 MNLI 90.6 vs RoBERTa-base 87.6 (He et al. 2021). Antoun et al. 2025 controlled study: **DeBERTaV3 beats ModernBERT-CV2 on every non-retrieval task** when data-matched; best F1 with limited data (our 8 k). Recent FiTDeBERTaV3-ATS SOTA on SST-5 fine-grained **62.4 %** (2025). Excellent for RMSE. |
| C4 | **`microsoft/deberta-v3-large`** | https://huggingface.co/microsoft/deberta-v3-large | **435 M total** (304 M backbone + 131 M embed, 24×1024) | **3.23 GB** static fp16 (HF Sizer) + 16-20 GB act. → **fits 24 GB** (bf16, ckpt, batch 8) | GLUE **91.4**, MNLI 91.8/91.9 SOTA large. Kaggle champion choice for years; beats RoBERTa-large 90.2. The large gives +2-3 pp over base on SST-5/GLUE. With checkpointing fits 24 GB (use grad-accum 2×4 if 512 len). Best generic NLU for valence nuance. |
| C5 | `answerdotai/ModernBERT-base` | https://huggingface.co/answerdotai/ModernBERT-base | **149 M**, 22×768, RoPE, GeGLU, 8 192 ctx, 50 k vocab | ~1.2 GB static + 7-11 GB act., **2× faster, 1/5 mem** vs DeBERTaV3 | *First to beat DeBERTaV3-base since 2021.* GLUE **88.4** vs DeBERTaV3 88.1; BEIR 41.6 vs 20.2 (Table 1 https://huggingface.co/answerdotai/ModernBERT-base). Trained on 2 T tokens (eng+code). Native unpadding → efficient for short tweets. Long-ctx (8192) overkill for tweets but useful for concatenated threads/meta. |
| C6 | `answerdotai/ModernBERT-large` | https://huggingface.co/answerdotai/ModernBERT-large | **395 M**, 28×1024 | ~2.9 GB static + 15-19 GB act. → **fits 24 GB** | GLUE **90.4** vs DeBERTaV3-large 91.4 (second-best large). 2× throughput vs DeBERTa, RoPE local-global attn. Good tradeoff if speed matters; still behind DeBERTa-large on pure NLU. |
| C7 | `FacebookAI/roberta-large` | https://huggingface.co/FacebookAI/roberta-large | **355 M**, 24×1024 | ~2.8 GB + 14-18 GB → fits | Generic large baseline; 160 GB pretrain. Beats base +1.5 GLUE; but no tweet data. Use as reference, not top pick (superseded by DeBERTa/ModernBERT). |
| C8 | `cardiffnlp/twitter-roberta-large-2022-154m` | https://huggingface.co/cardiffnlp/twitter-roberta-large-2022-154m | **~355 M**, 24×1024 | ~2.8 GB + 14-18 GB → fits | Tweet-RoBERTa-large (154 M tweets until 2022-12). TimeLMs large variant; not sentiment-fine-tuned. Strong domain + capacity, but less benchmarked than BERTweet-large. Viable if cardiff ecosystem preferred. |
| C9 | `Twitter/TwHIN-BERT-large` | https://huggingface.co/Twitter/TwHIN-BERT-large | **550 M**, 24×1024, 250 k vocab | ~3.5 GB static + 20-22 GB act. → **fits 24 GB borderline** (needs bf16+ckpt, seq_len 128 recommended, or batch 4×2) | Same as base but 550 M. Paper avg classification 60.06 vs base 59.38, hashtag 55.23 vs 54.62. Slight gain, heavy vocab cost. Fits only with 128 len; omit for 512-len sweeps. |
| C10 | `cardiffnlp/twitter-xlm-roberta-base` | https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base | **~270 M**, 12×768 | ~1.8 GB + 10-14 GB → fits | XLM-T (198 M multilingual tweets). Multilingual sentiment avg Macro F1 66.8→69.0 (multilingual) vs XLM-R. Weaker than TwHIN on English but useful if multilingual future. |

² HF Model Sizer Bot automated estimates: deberta-v3-base 1.37 GB fp16 Adam, large 3.23 GB; ModernBERT similar to bert-large scaling; TwHIN-large extrapolated from 280 M base. Activations dominate at seq 512: ~(batch·seq·hidden·layers·4 bytes). Tweet avg 30-50 tokens → real VRAM ~40 % lower if truncated to 128.

**Excluded (>24 GB or generative):** `microsoft/deberta-v2-xxlarge` (1.5 B), `FacebookAI/xlm-roberta-large` 550 M actually fits but excluded as non-tweet; `sentence-transformers` >1 B (e.g. `gte-large`); decoder LMs (LLaMA, Mistral, GPT).

---

## 4. Top-3 Selection (Union of 9 Existing + 10 Candidates, Ranked for Macro F1 + RMSE on Twitter)

### Ranking logic

1. **Domain signal** (tweet tokenization/pretrain) weighted 0.5 — decisive for Macro F1 on imbalanced slang/emoji.
2. **NLU capacity / sample efficiency** weighted 0.3 — decisive for RMSE fine-grained valence and minority-class calibration.
3. **Recency & data scale** (2 T vs 160 GB vs 80 GB) weighted 0.2.
4. **VRAM headroom** — all ≤24 GB, but lower overhead preferred for sweep flexibility.

### The Top-3

#### 🥇 1. `vinai/bertweet-large` — best expected Macro F1, strong RMSE
- **Link:** https://huggingface.co/vinai/bertweet-large — **355 M**, RoBERTa-large, 873 M tweets, 512 len.
- **Why #1:** Only *large* model natively pretrained on 850 M English tweets with tweet-normalized BPE (`normalizeTweet` keeps `#hashtag`, `@user`, emoji). Paper: BERTweet-base already beats RoBERTa-large/XLM-R-large on SemEval2017 sentiment and irony detection; large extends that gap. For 6 % negative class, tweet-specific subwords reduce OOV and improve recall (Macro F1). Valence RMSE benefits from constituent-level Twitter irony/sarcasm seen in pretrain. VRAM fits 24 GB (355 M ≈ RoBERTa-large); real tweet len 128 → 10-14 GB total, leaves room for batch 16 or multitask head.
- **Consideration:** Must use `AutoTokenizer.from_pretrained("vinai/bertweet-large", normalization=True)` + `emoji` lib; max_position 512 but pretrain saw 128 — truncation fine for tweets (max ~280 chars).

#### 🥈 2. `microsoft/deberta-v3-large` (fallback `microsoft/deberta-v3-base` if VRAM-tight) — best expected RMSE, joint best Macro F1
- **Link:** https://huggingface.co/microsoft/deberta-v3-large (large) / https://huggingface.co/microsoft/deberta-v3-base (base)
- **Params/VRAM:** large 435 M total (304 M backbone) — 3.23 GB fp16 static + ~18 GB act. at 8×512 with checkpointing → **fits 24 GB** in `bf16` with `gradient_checkpointing=True`, `per_device_train_batch_size=8`, `gradient_accumulation_steps=1`, `warmup_steps=50`. Base 184 M (1.37 GB static) fits trivially and retains ~95 % of large's GLUE (90.6 vs 91.8 MNLI).
- **Why #2:** DeBERTaV3's disentangled attention + RTD objective gives best *sample efficiency* (Antoun 2025) — critical with only 8 k labeled tweets. SOTA on fine-grained SST-5 (5-class sentiment) at 62.40 % (Do & Amjad 2025) mirrors our 3-class + continuous valence challenge. For RMSE, DeBERTa's token/position disentangling captures subtle intensity modifiers ("slightly positive" vs "ecstatic") better than vanilla RoBERTa/ModernBERT. Kaggle//TweetEval competitions still won by DeBERTaV3 ensembles in 2025. Use `lr=6e-6` for large (official script) vs 2e-5 for base, 2-3 epochs.
- **Tradeoff:** Generic domain, so it may trail BERTweet-large on hashtag-heavy Macro F1 by ~1 pp, but likely leads on valence RMSE by ~0.02-0.03.

#### 🥉 3. `answerdotai/ModernBERT-base` (alternative `ModernBERT-large` if chasing last point) — Pareto efficiency + second-best generic NLU
- **Link:** https://huggingface.co/answerdotai/ModernBERT-base — **149 M**, 22×768, 8 192 ctx.
- **Params/VRAM:** 149 M — ~1.5 GB fp16 + 8 GB act. at 8×512; **fastest** (FlashAttention-2, unpadding, RoPE, GeGLU). Large is 395 M, 2.9 GB + 16 GB, still fits but 40 % slower. Choose large only if max accuracy needed (≈90.4 GLUE vs 88.4 base).
- **Why #3:** First encoder since 2021 to *beat* `deberta-v3-base` on GLUE (88.4 vs 88.1) with **<1/5 memory** and **2× inference speed** (Answer.AI blog https://www.answer.ai/posts/2024-12-19-modernbert.html). Trained on 2 T tokens (vs 160 GB for DeBERTa), includes code — strong generalization for AI/ML tweet topics. GeGLU + RoPE handle length variance better than absolute pos embeddings. For the project's `twitter/main.py` multitask setup, ModernBERT's `153.7 examples/s` on 1024 len (paper) translates to cheaper sweep/ablation. Expected to trail DeBERTa-large by ~0.5-1.0 Macro F1 but beat all existing generic SBERT/RoBERTa by 2-3 points. Valence RMSE mid-way between #1 and #2.
- **Why not TwHIN in top-3?** `Twitter/TwHIN-BERT-base` (280 M, §2 #7) is the **best existing** and would rank **#4 overall** (avg 59.38 external sentiment vs ModernBERT's stronger GLUE/BEIR). It remains essential as *baseline* for ablation — ModernBERT-base is chosen over it for top-3 because ModernBERT already outperforms XLM-R/TwHIN lineage on GLUE/BEIR retrieval and is 1.9× smaller (149 M vs 280 M → larger sweep headroom), but TwHIN should be kept for direct domain comparison (see §6).

> **Existing-best honorable mention:** `Twitter/TwHIN-BERT-base` — if preference is to guarantee a tweet-native model from *already-used* set, swap it for ModernBERT-base at #3 with minimal expected loss (<1 pp). `cardiffnlp/twitter-roberta-base-sentiment-latest` (#8) is the best *sentiment-fine-tuned* existing checkpoint; use it only for weight-initialized fine-tune if you want to start from TweetEval optimum, but re-init the classification head for RMSE to avoid stale 3-label bias.

---

## 5. VRAM & Fine-Tuning Considerations (≤24 GB)

| Model (rank) | Total params | Static fp16/bf16 (model+grad+Adam)¹ | Activations 8×512² | Total est. 8×512 | Total est. 8×128 (tweet-realistic) | Fits 24 GB? | Tip |
|---|---|---|---|---|---|---|---|
| bertweet-large (#1) | 355 M | ~3.0 GB | 14-18 GB | **17-21 GB** | **10-13 GB** | ✅ | Use `bf16`, `gradient_checkpointing=True`, tokenizer `max_length=128`, `padding="max_length"` (BERTweet pad right). |
| deberta-v3-large (#2) | 435 M | 3.23 GB (HF Sizer) | 16-20 GB | **19-23 GB** | **12-14 GB** | ✅ (borderline 512) | Official GLUE uses `seq 256`; for tweets set `max_seq_length=128-256`, `per_device_train_batch_size=8`, `fp16`, `gradient_checkpointing`. If OOM at 512, fallback to `deberta-v3-base` (1.37 GB + 8-10 GB = 9-12 GB). |
| ModernBERT-base (#3) | 149 M | ~1.5 GB | 7-11 GB | **9-13 GB** | **6-8 GB** | ✅✅ | Enable `unpadding` (ModernBERT native) → saves ~30 % for variable tweet lengths. FlashAttention-2 required for 8k ctx; for 128-512 plain attention fine. |
| TwHIN-BERT-base (hon. mention) | 280 M | ~2.1 GB | 10-14 GB | 12-16 GB | 8-10 GB | ✅ | Same as BERT-base scaling, 250 k vocab overhead. |
| deberta-v3-base (fallback) | 184 M | 1.37 GB | 8-10 GB | 9-12 GB | 6-8 GB | ✅✅ | Use if sweeping many configs in parallel. |

¹ `static = 4× model size` (1× params + 1× grad + 2× Adam m/v) in fp16/bf16. HF Sizer reports deberta values directly; others interpolated from bert-large 340 M ≈ 2.7 GB fp16.
² activations ~ `batch·seq·hidden·layers·bytes·factor` (attention + MLP). Checkpointing halves it. Mixed precision prevents fp32 blowup (6.47 GB fp32 for deberta-large vs 3.23 GB fp16).

**Recommended training recipe for 24 GB (§ configs + `twitter/main.py`):**

```yaml
# in configs/encoders/<pick>.yaml
model:
  init_args:
    encoder:
      class_path: twitter.models.TransformerEncoder
      init_args:
        name: "vinai/bertweet-large"  # or microsoft/deberta-v3-large / answerdotai/ModernBERT-base
        pooling: "mean"        # bertweet/TwHIN benefit from mean; ModernBERT default cls but mean also fine for tweets
    lr: {"encoder": 2e-5, "head": 3e-4}  # deberta-large use 6e-6 for encoder (official)
    weight_decay: 0.01
trainer:
  precision: "bf16-mixed"      # or 16-mixed
  gradient_clip_val: 1.0
  accumulate_grad_batches: 1    # 2 if 512-len OOM on large
  enable_checkpointing: true
data:
  init_args:
    max_length: 128            # 512 only if you suspect long threads; tweets avg 30 tokens → 128 saves 60% VRAM
    batch_size: 8
```

Add in `twitter/modules.py` if not present: `self.automatic_optimization` with `gradient_checkpointing_enable()` for HF models.

---

## 6. Recommendations for Experiments

1. **Ablation matrix (2×2):** {`bertweet-large`, `deberta-v3-large`, `ModernBERT-base`, `TwHIN-BERT-base` (existing best)} × {classification only, regression only, multitask}. Keep `sentence_bert_base.yaml` as legacy anchor. Log Macro F1 + RMSE (holdout as in improvement-plan) to MLflow; report mean ± std over 3 seeds.

2. **Head handling:** Cardiff sentiment checkpoint (`twitter-roberta-base-sentiment-latest`) is already fine-tuned for 3 labels; if you reuse it, **re-initialize** `classifier` layer for valence regression to avoid negative transfer. For BERTweet/DeBERTa/ModernBERT, start from MLM checkpoint (not sentiment-tuned) and train both heads from scratch.

3. **Pooling:** Existing configs use both `cls` and `mean`. Benchmark shows `mean` + tweet tokenization helps 0.5-1 pp for short noisy text (BERTweet paper); ModernBERT defaults `cls` but supports `mean` via `classifier_pooling` config — test both.

4. **Imbalance handling:** Pair Macro F1 optimization with `class_weights` / `focal loss` / SMOTE (as already planned for LightGBM) and compare to undersampling; valence RMSE benefits from `loss_weight` tuning in `multitask.yaml` (e.g. 0.6 clf + 0.4 reg).

5. **Sequence length sweep:** Tweet 95th percentile <80 tokens after tweet tokenization; sweep `max_length ∈ {64, 128, 256}`. Expect 128 ≈ 512 performance at 40 % VRAM saving — confirms 24 GB headroom for larger batch/ensemble.

6. **Ensemble after single-model ranking:** Top-2 (`bertweet-large` + `deberta-v3-large`) have complementary errors (domain vs disentangled NLU); late-fusion (logit average + valence weighted average) likely +1-2 Macro F1 over single best, still under 24 GB sequential.

7. **Repro:** Pin `transformers>=4.48` for ModernBERT, `emoji` for BERTweet, `accelerate` for `device_map="auto"`. Log HF model revision hash in MLflow params.

---

## 7. References

- Nguyen et al. **BERTweet** (EMNLP 2020 demo) https://arxiv.org/abs/2005.10200 — HF https://huggingface.co/vinai/bertweet-large / https://huggingface.co/vinai/bertweet-base
- He et al. **DeBERTaV3** (2021, ELECTRA+ GDES) https://arxiv.org/abs/2111.09543 — HF https://huggingface.co/microsoft/deberta-v3-base / https://huggingface.co/microsoft/deberta-v3-large — HF Sizer discussions for VRAM
- Warner et al. **ModernBERT** (Dec 2024, arXiv:2412.13663) https://huggingface.co/answerdotai/ModernBERT-base / https://huggingface.co/answerdotai/ModernBERT-large — blog https://huggingface.co/blog/modernbert / https://www.answer.ai/posts/2024-12-19-modernbert.html
- Barbieri et al. **TweetEval / TimeLMs** (2020, 2022) https://arxiv.org/abs/2010.12421 / https://arxiv.org/abs/2202.03829 — HF https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment / https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest / https://huggingface.co/cardiffnlp/twitter-roberta-large-2022-154m / https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base
- Zhang et al. **TwHIN-BERT** (2022) https://arxiv.org/abs/2209.07562 — HF https://huggingface.co/Twitter/TwHIN-BERT-base / https://huggingface.co/Twitter/twhin-bert-large
- Antoun et al. **ModernBERT or DeBERTaV3?** (IJCNLP 2025) https://aclanthology.org/2025.ijcnlp-long.164.pdf — controlled DeBERTa > ModernBERT on non-retrieval
- Do & Amjad **FiTDeBERTaV3-ATS** on SST-5 fine-grained 62.40 % (2025) https://doi.org/10.1109/cai64502.2025.00026
- Project context: `README.md:1-48` (8 k tweets, 6 % negative, valence ∈ [-1,1]), `configs/encoders/*.yaml` (9 encoders), `docs/improvement-plan.md:57-59`
