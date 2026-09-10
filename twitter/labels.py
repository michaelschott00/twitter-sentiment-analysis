"""Shared label coding for Twitter sentiment analysis.

Kept dependency-free (stdlib only) so CPU-only / lightweight entry points
(e.g. the LightGBM and LLM baselines) can import it without pulling in
torch / lightning / transformers via ``twitter.data``.
"""

LABEL_CODING: dict[str, int] = {"negative": 0, "neutral": 1, "positive": 2}
INVERSE_LABEL_CODING: dict[int, str] = {v: k for k, v in LABEL_CODING.items()}
