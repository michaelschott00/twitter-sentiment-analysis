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
        GH_TOKEN="gh-secret",
        AZURE_CLIENT_SECRET="az-secret",
        RUNPOD_API_KEY="rp-secret",
    )
    b = broker_mod.Broker()
    env = b.clean_env(["GH_TOKEN"])
    assert env.get("GH_TOKEN") == "gh-secret"
    assert "AZURE_CLIENT_SECRET" not in env
    assert "RUNPOD_API_KEY" not in env


def test_runpod_env_excludes_github(monkeypatch):
    _set_broker_secrets(monkeypatch, GH_TOKEN="gh-secret", RUNPOD_API_KEY="rp-secret")
    b = broker_mod.Broker()
    env = b.clean_env(["RUNPOD_API_KEY"])
    assert env["RUNPOD_API_KEY"] == "rp-secret"
    assert "GH_TOKEN" not in env


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
    _set_broker_secrets(monkeypatch, GH_TOKEN="gh-secret")
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


def test_template_image_skips_confinement():
    with patch.object(server_mod.subprocess, "run") as m:
        m.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        out = server_mod.call_tool(
            "runpodctl_template_create",
            argv=[
                "runpodctl",
                "template",
                "create",
                "--name",
                "t",
                "--image",
                "michaelschott00/twitter-sentiment:dev",
            ],
        )
    assert "returncode: 0" in out
    assert "michaelschott00/twitter-sentiment:dev" in m.call_args[0][0]


def test_template_image_eq_form_preserved():
    with patch.object(server_mod.subprocess, "run") as m:
        m.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        server_mod.call_tool(
            "runpodctl_template_create",
            argv=[
                "runpodctl",
                "template",
                "create",
                "--image=runpod/pytorch:2.1.0",
            ],
        )
    assert "--image=runpod/pytorch:2.1.0" in m.call_args[0][0]


def test_template_image_rejects_path():
    out = server_mod.call_tool(
        "runpodctl_template_create",
        argv=[
            "runpodctl",
            "template",
            "create",
            "--name",
            "t",
            "--image",
            "/etc/passwd",
        ],
    )
    assert "returncode: 1" in out and "docker image" in out


def test_unmarked_slash_value_still_confined(tmp_ws):
    with patch.object(server_mod.subprocess, "run") as m:
        m.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        server_mod.call_tool(
            "gh_pr", argv=["gh", "pr", "list", "--repo", "o/r", "--limit", "5"]
        )
    assert str(tmp_ws / "o" / "r") in m.call_args[0][0]


def test_url_param_skips_confinement():
    cfg = {"base_argv": ["foo"], "params": [{"names": ["--u"], "url": True}]}
    out = server_mod._validate_argv(cfg, {}, ["foo", "--u", "https://x.example/a/b"])
    assert out == ["foo", "--u", "https://x.example/a/b"]
    with pytest.raises(ValueError, match="not a URL"):
        server_mod._validate_argv(cfg, {}, ["foo", "--u", "not-a-url"])


def test_template_ports_skips_confinement():
    with patch.object(server_mod.subprocess, "run") as m:
        m.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        out = server_mod.call_tool(
            "runpodctl_template_create",
            argv=[
                "runpodctl",
                "template",
                "create",
                "--name",
                "t",
                "--image",
                "nginx",
                "--ports",
                "22/tcp,8888/http",
            ],
        )
    assert "returncode: 0" in out
    assert "22/tcp,8888/http" in m.call_args[0][0]


def test_template_ports_rejects_path():
    out = server_mod.call_tool(
        "runpodctl_template_create",
        argv=[
            "runpodctl",
            "template",
            "create",
            "--name",
            "t",
            "--image",
            "nginx",
            "--ports",
            "configs/x.yaml",
        ],
    )
    assert "returncode: 1" in out and "port list" in out


def test_azcopy_rejects_non_blob_url():
    out = server_mod.call_tool(
        "azcopy_copy",
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
        m.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        out = server_mod.call_tool(
            "azcopy_copy",
            argv=[
                "azcopy",
                "copy",
                "https://acct.blob.core.windows.net/c/x",
                "--recursive",
            ],
        )
    assert "returncode: 0" in out


def test_azcopy_list_rejects_non_blob_url():
    out = server_mod.call_tool(
        "azcopy_list",
        argv=["azcopy", "list", "https://evil.example/x"],
    )
    assert "not allowed" in out


def test_azcopy_list_allows_blob_url(monkeypatch):
    monkeypatch.setattr(server_mod, "broker", broker_mod.Broker())
    with patch.object(server_mod.subprocess, "run") as m:
        m.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        out = server_mod.call_tool(
            "azcopy_list",
            argv=["azcopy", "list", "https://acct.blob.core.windows.net/c/x"],
        )
    assert "returncode: 0" in out


def test_canary_never_leaks(monkeypatch, tmp_ws):
    canary = "canary-9f8e7d6c5b4a"
    _set_broker_secrets(monkeypatch, GH_TOKEN=canary)
    monkeypatch.setattr(server_mod, "broker", broker_mod.Broker())
    with patch.object(server_mod.subprocess, "run") as m:
        m.return_value = MagicMock(returncode=0, stdout=f"token={canary}", stderr="")
        out = server_mod.call_tool("gh_pr", argv=["gh", "pr", "list"])
    assert canary not in out
    assert "GH_TOKEN" in out
