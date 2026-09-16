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


# --- argv validation ---------------------------------------------------------


def test_prefix_mismatch_rejected():
    out = server_mod.call_tool("gh_pr", argv=["gh", "issue", "list"])
    assert "returncode: 1" in out and "must start with" in out


def test_rg_override_rejected():
    out = server_mod.call_tool(
        "azml_job_list",
        argv=["az", "ml", "job", "list", "-g", "BAD_RG!!"],
    )
    assert "returncode: 1" in out and "not allowed" in out


def test_flag_eq_form_checked():
    out = server_mod.call_tool(
        "azml_job_list",
        argv=["az", "ml", "job", "list", "-g=BAD!!"],
    )
    assert "returncode: 1" in out and "not allowed" in out


def test_job_yaml_outside_workspace_rejected(tmp_ws):
    out = server_mod.call_tool(
        "azml_job_submit",
        argv=["az", "ml", "job", "create", "--file", "/etc/passwd"],
    )
    assert "returncode: 1" in out and "outside workspace" in out


def test_unlisted_flag_is_passthrough(monkeypatch):
    with patch.object(server_mod.subprocess, "run") as m:
        m.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        out = server_mod.call_tool(
            "gh_pr", argv=["gh", "pr", "list", "--repo", "o/r", "--limit", "5"]
        )
    assert "returncode: 0" in out


def test_shell_meta_rejected_globally():
    out = server_mod.call_tool(
        "runpodctl_pod_create",
        argv=["runpodctl", "pod", "create", "--name", "x; rm -rf /"],
    )
    assert "returncode: 1" in out and "shell metacharacters" in out
    out = server_mod.call_tool("gh_pr", argv=["gh", "pr", "list; evil"])
    assert "returncode: 1" in out and "shell metacharacters" in out


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
        out = server_mod.call_tool(
            "azml_job_submit",
            argv=["az", "ml", "job", "create", "--file", "job.yaml"],
        )
    assert "returncode: 0" in out
    argv = m.call_args[0][0]
    assert argv[:4] == ["az", "ml", "job", "create"]
    assert "-g" in argv and "rg-twitter-ml" in argv
    assert "-w" in argv and "mlw-twitter-sentiment" in argv


def test_explicit_scope_not_overridden(tmp_ws, monkeypatch):
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
        server_mod.call_tool(
            "azml_job_submit",
            argv=[
                "az",
                "ml",
                "job",
                "create",
                "--file",
                "job.yaml",
                "-g",
                "rg-custom",
                "-w",
                "ws-custom",
            ],
        )
    argv = m.call_args[0][0]
    assert argv.count("-g") == 1 and "rg-custom" in argv
    assert "rg-twitter-ml" not in argv


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
        out = server_mod.call_tool(
            "azml_job_submit",
            argv=["az", "ml", "job", "create", "--file", "job.yaml"],
        )
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
        out = server_mod.call_tool(
            "azml_job_submit",
            argv=["az", "ml", "job", "create", "--file", "job.yaml"],
        )
    assert "returncode: 0" in out
    assert m.call_count == 3
    assert m.call_args_list[1][0][0][:2] == ["az", "login"]
    assert "csecret" not in out  # secret never leaks into tool output


def test_az_login_only_for_azure_tools(monkeypatch):
    _set_broker_secrets(monkeypatch, GITHUB_TOKEN="gh-secret")
    monkeypatch.setattr(server_mod, "broker", broker_mod.Broker())
    with patch.object(server_mod.subprocess, "run") as m:
        m.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        server_mod.call_tool("gh_pr", argv=["gh", "pr", "list"])
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
        out = server_mod.call_tool(
            "azml_job_submit",
            argv=["az", "ml", "job", "create", "--file", "job.yaml"],
        )
    assert "returncode: 1" in out and "azure login failed" in out
    assert secret not in out


def test_show_help_skips_az_login(monkeypatch):
    _set_broker_secrets(monkeypatch)
    monkeypatch.setattr(server_mod, "broker", broker_mod.Broker())
    with patch.object(server_mod.subprocess, "run") as m:
        m.return_value = MagicMock(returncode=0, stdout="usage: az", stderr="")
        out = server_mod.call_tool("azml_job_list", show_help=True)
    assert "returncode: 0" in out
    argv = m.call_args[0][0]
    assert argv == ["az", "ml", "job", "list", "--help"]
    assert m.call_count == 1  # no account show / login


def test_runpodctl_forwards_valid_options():
    with patch.object(server_mod.subprocess, "run") as m:
        m.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        server_mod.call_tool(
            "runpodctl_pod_create",
            argv=["runpodctl", "pod", "create", "--name", "mypod"],
        )
    assert m.call_args[0][0][:3] == ["runpodctl", "pod", "create"]


def test_azcopy_rejects_non_blob_url():
    out = server_mod.call_tool(
        "azcopy",
        argv=["azcopy", "copy", "https://evil.example/x", "https://evil.example/y"],
    )
    assert "not allowed" in out


def test_azcopy_allows_blob_url(monkeypatch):
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
        out = server_mod.call_tool(
            "azcopy",
            argv=[
                "azcopy",
                "copy",
                "https://acct.blob.core.windows.net/c/x",
                "--recursive",
            ],
        )
    assert "returncode: 0" in out


def test_canary_never_leaks(monkeypatch, tmp_ws):
    canary = "canary-9f8e7d6c5b4a"
    _set_broker_secrets(monkeypatch, GITHUB_TOKEN=canary)
    monkeypatch.setattr(server_mod, "broker", broker_mod.Broker())
    with patch.object(server_mod.subprocess, "run") as m:
        m.return_value = MagicMock(returncode=0, stdout=f"token={canary}", stderr="")
        out = server_mod.call_tool("gh_pr", argv=["gh", "pr", "list"])
    assert canary not in out
    assert "GITHUB_TOKEN" in out
