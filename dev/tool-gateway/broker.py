"""Credential broker: per-tool least-privilege env + redaction table.

Compose injects secrets via env_file into this container's environ.
The broker snapshots them at startup, then hands each tool invocation
only the creds listed for that tool in tools.yaml.
"""

from __future__ import annotations

import os

# Secrets: exact env var names treated as credentials (redacted + scoped).
# Non-secret config (subscription IDs, defaults) passes through separately.
SECRET_NAMES = [
    "GITHUB_TOKEN",
    "GITHUB_PERSONAL_ACCESS_TOKEN",
    "RUNPOD_API_KEY",
    "AZURE_CLIENT_ID",
    "AZURE_TENANT_ID",
    "AZURE_CLIENT_SECRET",
    "OPENAI_API_KEY",
]

MINIMAL_PASSTHROUGH = ["PATH", "HOME", "USER", "LANG", "LC_ALL", "TZ"]


class Broker:
    def __init__(self) -> None:
        self._secrets: dict[str, str] = {}
        for name in SECRET_NAMES:
            val = os.environ.get(name, "")
            if val:
                self._secrets[name] = val
        # Normalize aliases: PAT -> TOKEN so gh tools work either way.
        pat = self._secrets.get("GITHUB_PERSONAL_ACCESS_TOKEN")
        if pat and "GITHUB_TOKEN" not in self._secrets:
            self._secrets["GITHUB_TOKEN"] = pat

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
                # gh CLI reads GH_TOKEN; azure CLI reads AZURE_* natively.
                if name == "GITHUB_TOKEN":
                    env.setdefault("GH_TOKEN", val)
        return env

    def secret_values(self) -> dict[str, str]:
        return dict(self._secrets)
