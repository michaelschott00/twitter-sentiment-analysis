"""Tests for the tool-gateway (broker scoping, validation, redaction).

Run with: pytest dev/mcp_tests.py
"""

import base64
import sys
import urllib.parse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "tool-gateway"))

import broker as broker_mod
import redact as redact_mod
import server as server_mod


@pytest.fixture
def tmp_ws(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setattr(server_mod, "REPO_ROOT", root)
    return root


def _set_broker_secrets(monkeypatch, **vals):
    for name in broker_mod.SECRET_NAMES:
        monkeypatch.delenv(name, raising=False)
    for k, v in vals.items():
        monkeypatch.setenv(k, v)


# --- broker scoping ----------------------------------------------------------


def test_gh_env_excludes_azure_and_runpod(monkeypatch):
    _set_broker_secrets(
        monkeypatch,
        GITHUB_TOKEN="gh-secret",
        AZURE_CLIENT_SECRET="az-secret",
        RUNPOD_API_KEY="rp-secret",
    )
    b = broker_mod.Broker()
    env = b.clean_env(["GITHUB_TOKEN"])
    assert env.get("GITHUB_TOKEN") == "gh-secret"
    assert "AZURE_CLIENT_SECRET" not in env
    assert "RUNPOD_API_KEY" not in env


def test_runpod_env_excludes_github(monkeypatch):
    _set_broker_secrets(
        monkeypatch, GITHUB_TOKEN="gh-secret", RUNPOD_API_KEY="rp-secret"
    )
    b = broker_mod.Broker()
    env = b.clean_env(["RUNPOD_API_KEY"])
    assert env["RUNPOD_API_KEY"] == "rp-secret"
    assert "GITHUB_TOKEN" not in env


# --- redaction ---------------------------------------------------------------


def test_redact_exact_url_b64_variants():
    secret = "s3cr3t-value/with?chars"
    secrets = {"AZURE_CLIENT_SECRET": secret}
    text = " ".join(
        [
            secret,
            urllib.parse.quote(secret, safe=""),
            base64.b64encode(secret.encode()).decode(),
        ]
    )
    out, _ = redact_mod.redact(text, secrets)
    assert secret not in out
    assert "AZURE_CLIENT_SECRET" in out


def test_heuristic_continues():
    out, hits = redact_mod.redact("leaked ghp_abcdefghijklmnopqrst ok", {})
    assert hits == 1 and "ghp_" not in out and out.endswith("ok")


# --- validation --------------------------------------------------------------


def test_rg_override_rejected():
    out = server_mod.azml_job_list(resource_group="BAD_RG!!")
    assert "returncode: 1" in out and "invalid resource_group" in out


def test_job_yaml_outside_workspace_rejected(tmp_ws):
    out = server_mod.azml_job_submit(job_yaml="/etc/passwd")
    assert "returncode: 1" in out


def test_job_yaml_missing_rejected(tmp_ws):
    out = server_mod.azml_job_submit(job_yaml="nope.yaml")
    assert "not found" in out


def test_job_submit_builds_argv(tmp_ws, monkeypatch):
    spec = tmp_ws / "job.yaml"
    spec.write_text("name: x\n")
    _set_broker_secrets(
        monkeypatch,
        AZURE_CLIENT_ID="cid",
        AZURE_TENANT_ID="tid",
        AZURE_CLIENT_SECRET="csecret",
    )
    monkeypatch.setattr(server_mod, "broker", broker_mod.Broker())
    with patch.object(server_mod.subprocess, "run") as m:
        m.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        out = server_mod.azml_job_submit(job_yaml="job.yaml")
    assert "returncode: 0" in out
    argv = m.call_args[0][0]
    assert argv[:4] == ["az", "ml", "job", "create"]
    assert "-g" in argv and "rg-twitter-ml" in argv
    assert "-w" in argv and "mlw-twitter-sentiment" in argv


def test_az_login_skipped_when_already_authenticated(monkeypatch, tmp_ws):
    spec = tmp_ws / "job.yaml"
    spec.write_text("name: x\n")
    _set_broker_secrets(
        monkeypatch,
        AZURE_CLIENT_ID="cid",
        AZURE_TENANT_ID="tid",
        AZURE_CLIENT_SECRET="csecret",
    )
    monkeypatch.setattr(server_mod, "broker", broker_mod.Broker())
    with patch.object(server_mod.subprocess, "run") as m:
        m.side_effect = [
            MagicMock(returncode=0, stdout="", stderr=""),  # account show
            MagicMock(returncode=0, stdout="ok", stderr=""),  # main cmd
        ]
        out = server_mod.azml_job_submit(job_yaml="job.yaml")
    assert "returncode: 0" in out
    assert m.call_count == 2
    assert m.call_args_list[0][0][0][:3] == ["az", "account", "show"]


def test_az_login_runs_on_cold_cache(monkeypatch, tmp_ws):
    spec = tmp_ws / "job.yaml"
    spec.write_text("name: x\n")
    _set_broker_secrets(
        monkeypatch,
        AZURE_CLIENT_ID="cid",
        AZURE_TENANT_ID="tid",
        AZURE_CLIENT_SECRET="csecret",
    )
    monkeypatch.setattr(server_mod, "broker", broker_mod.Broker())
    with patch.object(server_mod.subprocess, "run") as m:
        m.side_effect = [
            MagicMock(returncode=1, stdout="", stderr="not logged in"),
            MagicMock(returncode=0, stdout="", stderr=""),  # login
            MagicMock(returncode=0, stdout="ok", stderr=""),  # main cmd
        ]
        out = server_mod.azml_job_submit(job_yaml="job.yaml")
    assert "returncode: 0" in out
    assert m.call_count == 3
    assert m.call_args_list[1][0][0][:2] == ["az", "login"]
    assert "csecret" not in out  # secret never leaks into tool output


def test_az_login_only_for_azure_tools(monkeypatch):
    _set_broker_secrets(monkeypatch, GITHUB_TOKEN="gh-secret")
    monkeypatch.setattr(server_mod, "broker", broker_mod.Broker())
    with patch.object(server_mod.subprocess, "run") as m:
        m.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        server_mod.gh(["issue", "list"])
    for c in m.call_args_list:
        assert c[0][0][:2] != ["az", "login"]
        assert c[0][0][:3] != ["az", "account", "show"]


def test_az_login_failure_redacts_secret(monkeypatch, tmp_ws):
    secret = "super-secret-value-123"
    spec = tmp_ws / "job.yaml"
    spec.write_text("name: x\n")
    _set_broker_secrets(
        monkeypatch,
        AZURE_CLIENT_ID="cid",
        AZURE_TENANT_ID="tid",
        AZURE_CLIENT_SECRET=secret,
    )
    monkeypatch.setattr(server_mod, "broker", broker_mod.Broker())
    with patch.object(server_mod.subprocess, "run") as m:
        m.side_effect = [
            MagicMock(returncode=1, stdout="", stderr="not logged in"),
            MagicMock(returncode=1, stdout="", stderr=f"bad {secret}"),
        ]
        out = server_mod.azml_job_submit(job_yaml="job.yaml")
    assert "returncode: 1" in out and "azure login failed" in out
    assert secret not in out


def test_gh_rejects_disallowed_subcommand():
    out = server_mod.gh(["auth", "status"])
    assert "returncode: 1" in out


def test_runpodctl_rejects_shell_meta():
    out = server_mod.runpodctl_pod_create(["--name", "x; rm -rf /"])
    assert "returncode: 1" in out


def test_runpodctl_forwards_valid_options():
    with patch.object(server_mod.subprocess, "run") as m:
        m.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        server_mod.runpodctl_pod_create(["--name", "mypod"])
    assert m.call_args[0][0][:3] == ["runpodctl", "pod", "create"]


def test_azcopy_rejects_non_blob_url():
    out = server_mod.azcopy(
        ["copy", "https://evil.example/x", "https://evil.example/y"]
    )
    assert "URL not allowed" in out


def test_canary_never_leaks(monkeypatch, tmp_ws):
    canary = "canary-9f8e7d6c5b4a"
    _set_broker_secrets(monkeypatch, GITHUB_TOKEN=canary)
    monkeypatch.setattr(server_mod, "broker", broker_mod.Broker())
    with patch.object(server_mod.subprocess, "run") as m:
        m.return_value = MagicMock(returncode=0, stdout=f"token={canary}", stderr="")
        out = server_mod.gh(["issue", "list"])
    assert canary not in out
    assert "GITHUB_TOKEN" in out
