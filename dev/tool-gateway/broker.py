"""Credential broker: per-tool least-privilege env + redaction table.

Compose injects secrets via env_file into this container's environ.
The broker snapshots them at startup, then hands each tool invocation
only the creds listed for that tool in tools.yaml.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
TOOLS_YAML = Path(os.environ.get("TOOLS_YAML", HERE.parent / "tools.yaml"))


def _load_secret_names() -> list[str]:
    """Collect every distinct `creds:` env var declared in tools.yaml.

    This keeps the broker's redaction + scoping table in sync with the
    manifest automatically, so adding a credential to a tool needs no
    broker.py edit.
    """
    with open(TOOLS_YAML) as f:
        manifest = yaml.safe_load(f) or {}
    names: dict[str, None] = {}
    for tool in manifest.get("tools", []) or []:
        for name in tool.get("creds", []) or []:
            names[name] = None
    return list(names)


# Secrets: exact env var names treated as credentials (redacted + scoped).
# Non-secret config (subscription IDs, defaults) passes through separately.
SECRET_NAMES = _load_secret_names()

MINIMAL_PASSTHROUGH = ["PATH", "HOME", "USER", "LANG", "LC_ALL", "TZ"]


class Broker:
    def __init__(self) -> None:
        self._secrets: dict[str, str] = {}
        for name in SECRET_NAMES:
            val = os.environ.get(name, "")
            if val:
                self._secrets[name] = val

    @property
    def secret_names(self) -> list[str]:
        return [n for n in self._secrets if self._secrets[n]]

    def scrub_startup_env(self) -> None:
        """Remove raw secrets from this process environ (defense in depth)."""
        for name in self._secrets:
            os.unsetenv(name)
            os.environ.pop(name, None)

    def clean_env(self, creds: list[str]) -> dict[str, str]:
        """Build minimal environ for one invocation: PATH/HOME + listed creds."""
        env: dict[str, str] = {}
        for name in MINIMAL_PASSTHROUGH:
            if val := os.environ.get(name):
                env[name] = val
        # Non-secret config passthrough (safe to share across tools).
        for name in ("AZURE_SUBSCRIPTION_ID", "AZURE_CONFIG_DIR", "PORT"):
            if val := os.environ.get(name):
                env[name] = val
        for name in creds:
            if val := self._secrets.get(name):
                env[name] = val
        return env

    def secret_values(self) -> dict[str, str]:
        return dict(self._secrets)
