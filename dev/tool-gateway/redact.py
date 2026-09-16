"""Best-effort redact-and-continue output filter.

Deterministic pass over known secret values (exact + url-encoded + base64
variants), then heuristic pass for token-like patterns. Never raises.
"""

from __future__ import annotations

import base64
import re
import urllib.parse

HEURISTIC_PATTERNS = [
    (re.compile(r"ghp_[A-Za-z0-9]{20,}"), "***REDACTED:HEURISTIC***"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), "***REDACTED:HEURISTIC***"),
    (re.compile(r"gho_[A-Za-z0-9]{20,}"), "***REDACTED:HEURISTIC***"),
    (re.compile(r"sk-[A-Za-z0-9]{16,}"), "***REDACTED:HEURISTIC***"),
    (re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"), "***REDACTED:HEURISTIC***"),
    (
        re.compile(r"eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}"),
        "***REDACTED:HEURISTIC***",
    ),
    (re.compile(r"-----BEGIN .*PRIVATE KEY-----"), "***REDACTED:HEURISTIC***"),
]


def _variants(value: str) -> list[str]:
    out = [value]
    try:
        out.append(urllib.parse.quote(value, safe=""))
        out.append(urllib.parse.quote_plus(value))
        raw = value.encode()
        out.append(base64.b64encode(raw).decode())
        out.append(base64.urlsafe_b64encode(raw).decode())
        out.append(base64.b64encode(raw).decode().rstrip("="))
    except Exception:  # noqa: BLE001, S110 - redaction must never fail
        pass
    # Deduplicate, longest first so nested replacements don't partially match.
    return sorted({v for v in out if v and len(v) >= 4}, key=len, reverse=True)


def redact(text: str, secrets: dict[str, str]) -> tuple[str, int]:
    """Return (redacted_text, heuristic_hit_count)."""
    if not text:
        return text, 0
    redacted = text
    for name, value in secrets.items():
        if not value or len(value) < 4:
            continue
        for variant in _variants(value):
            if variant in redacted:
                redacted = redacted.replace(variant, f"***REDACTED:{name}***")
    hits = 0
    for pattern, replacement in HEURISTIC_PATTERNS:
        redacted, n = pattern.subn(replacement, redacted)
        hits += n
    return redacted, hits
