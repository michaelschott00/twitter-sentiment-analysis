"""Tests for dev/mcp_server.py sidecar tools.

Run with: pytest dev/mcp_tests.py
"""

import os
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dev import mcp_server
from dev.mcp_server import ToolError


@pytest.fixture
def ws_root(tmp_path, monkeypatch):
    """Isolated workspace root so file-guard tests don't touch the repo."""
    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setattr(mcp_server, "_WORKSPACE_ROOT", str(root))
    return root


@pytest.fixture
def spec_yaml(ws_root):
    spec = ws_root / "job.yaml"
    spec.write_text("name: test-job\n")
    return spec.name  # relative path, as callers would pass it


@pytest.fixture
def scope_env(monkeypatch):
    monkeypatch.setenv("AZUREML_WORKSPACE_NAME", "test-ws")
    monkeypatch.setenv("AZUREML_RESOURCE_GROUP", "test-rg")


def _ok(stdout="ok"):
    return MagicMock(returncode=0, stdout=stdout, stderr="")


def _run_tool(fn, **kwargs):
    with patch.object(mcp_server.subprocess, "run") as mock_run:
        mock_run.return_value = _ok()
        out = fn(**kwargs)
        return out, mock_run.call_args[0][0]


# --- tool registration -----------------------------------------------------


def test_all_tools_registered():
    for tool in (
        "azml_data",
        "azml_environment",
        "azml_job",
        "azml_online_endpoint",
        "azml_online_deployment",
        "azcopy_upload",
        "git_push",
    ):
        assert callable(getattr(mcp_server, tool, None)), tool


# --- _ensure_azure_login ----------------------------------------------------


def test_login_skipped_when_env_missing(monkeypatch, capsys):
    for var in (
        "AZCOPY_SPA_APPLICATION_ID",
        "AZCOPY_TENANT_ID",
        "AZCOPY_SPA_CLIENT_SECRET",
    ):
        monkeypatch.delenv(var, raising=False)
    mcp_server._ensure_azure_login()
    assert "skipped" in capsys.readouterr().err


def test_login_skipped_when_already_authenticated(monkeypatch):
    monkeypatch.setenv("AZCOPY_SPA_APPLICATION_ID", "app")
    monkeypatch.setenv("AZCOPY_TENANT_ID", "tenant")
    monkeypatch.setenv("AZCOPY_SPA_CLIENT_SECRET", "secret")
    with patch.object(mcp_server.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        mcp_server._ensure_azure_login()
        assert mock_run.call_count == 1
        assert mock_run.call_args[0][0][:3] == ["az", "account", "show"]


def test_login_runs_service_principal_login(monkeypatch):
    monkeypatch.setenv("AZCOPY_SPA_APPLICATION_ID", "test-app")
    monkeypatch.setenv("AZCOPY_TENANT_ID", "test-tenant")
    monkeypatch.setenv("AZCOPY_SPA_CLIENT_SECRET", "super-secret")
    with patch.object(mcp_server.subprocess, "run") as mock_run:
        mock_run.side_effect = [
            MagicMock(returncode=1, stdout="", stderr="not logged in"),
            MagicMock(returncode=0, stdout="", stderr=""),
        ]
        mcp_server._ensure_azure_login()
        login_cmd = mock_run.call_args_list[1][0][0]
        assert login_cmd == [
            "az",
            "login",
            "--service-principal",
            "--username",
            "test-app",
            "--tenant",
            "test-tenant",
            "--password",
            "super-secret",
            "--output",
            "none",
        ]


def test_login_failure_redacts_secret(monkeypatch, capsys):
    monkeypatch.setenv("AZCOPY_SPA_APPLICATION_ID", "test-app")
    monkeypatch.setenv("AZCOPY_TENANT_ID", "test-tenant")
    monkeypatch.setenv("AZCOPY_SPA_CLIENT_SECRET", "super-secret")
    with patch.object(mcp_server.subprocess, "run") as mock_run:
        mock_run.side_effect = [
            MagicMock(returncode=1, stdout="", stderr="not logged in"),
            MagicMock(returncode=1, stdout="", stderr="bad creds"),
        ]
        mcp_server._ensure_azure_login()
        err = capsys.readouterr().err
        assert "super-secret" not in err
        assert "azure login failed" in err


# --- scope / workspace guards ----------------------------------------------


def test_missing_scope_raises(monkeypatch):
    monkeypatch.delenv("AZUREML_WORKSPACE_NAME", raising=False)
    monkeypatch.delenv("AZUREML_RESOURCE_GROUP", raising=False)
    with pytest.raises(ToolError):
        mcp_server.azml_data("list")
    with pytest.raises(ToolError):
        mcp_server.azml_data("list", workspace="ws")
    with pytest.raises(ToolError):
        mcp_server.azml_data("list", resource_group="rg")


def test_scope_env_fallback(scope_env):
    _, cmd = _run_tool(mcp_server.azml_job, operation="list")
    assert "test-ws" in cmd and "test-rg" in cmd


# --- file guards ------------------------------------------------------------


def test_file_outside_workspace_rejected(scope_env):
    with pytest.raises(ToolError):
        mcp_server.azml_data(
            "create",
            file="/etc/passwd",
            workspace="ws",
            resource_group="rg",
        )


def test_file_escape_rejected(ws_root, scope_env):
    with pytest.raises(ToolError):
        mcp_server.azml_data(
            "create",
            file="../outside.yaml",
            workspace="ws",
            resource_group="rg",
        )


def test_file_not_found_rejected(ws_root, scope_env):
    with pytest.raises(ToolError):
        mcp_server.azml_data(
            "create",
            file="nonexistent.yaml",
            workspace="ws",
            resource_group="rg",
        )


def test_non_yaml_rejected(ws_root, scope_env):
    (ws_root / "notes.txt").write_text("hi\n")
    with pytest.raises(ToolError):
        mcp_server.azml_data(
            "create", file="notes.txt", workspace="ws", resource_group="rg"
        )


# --- argument guards --------------------------------------------------------


def test_set_args_must_be_key_value(spec_yaml, scope_env):
    with pytest.raises(ToolError):
        mcp_server.azml_data(
            "create",
            file=spec_yaml,
            workspace="ws",
            resource_group="rg",
            set_args=["--output json"],
        )


def test_endpoint_update_needs_change(scope_env):
    with pytest.raises(ToolError):
        mcp_server.azml_online_endpoint(
            "update", name="tw-sentiment", workspace="ws", resource_group="rg"
        )


def test_show_needs_name(scope_env):
    with pytest.raises(ToolError):
        mcp_server.azml_data("show", workspace="ws", resource_group="rg")


def test_deployment_show_needs_endpoint(scope_env):
    with pytest.raises(ToolError):
        mcp_server.azml_online_deployment(
            "show", name="blue", workspace="ws", resource_group="rg"
        )


def test_deployment_create_needs_file_or_names(scope_env):
    with pytest.raises(ToolError):
        mcp_server.azml_online_deployment("create", workspace="ws", resource_group="rg")


# --- command construction ---------------------------------------------------


def _assert_minimal_output(cmd):
    assert cmd[-3:] == ["--output", "table", "--only-show-errors"]
    assert "get-credentials" not in cmd
    assert "regenerate-keys" not in cmd


def test_data_create(spec_yaml, scope_env):
    _, cmd = _run_tool(mcp_server.azml_data, operation="create", file=spec_yaml)
    assert cmd[:4] == ["az", "ml", "data", "create"]
    assert "--file" in cmd
    assert "--workspace-name" in cmd and "--resource-group" in cmd
    _assert_minimal_output(cmd)


def test_environment_list(scope_env):
    _, cmd = _run_tool(mcp_server.azml_environment, operation="list")
    assert cmd[:4] == ["az", "ml", "environment", "list"]
    _assert_minimal_output(cmd)


def test_job_create_with_stream(spec_yaml, scope_env):
    _, cmd = _run_tool(
        mcp_server.azml_job,
        operation="create",
        file=spec_yaml,
        stream_logs=True,
    )
    assert cmd[:4] == ["az", "ml", "job", "create"]
    assert "--stream" in cmd
    _assert_minimal_output(cmd)


def test_job_stream_uses_name(scope_env):
    _, cmd = _run_tool(mcp_server.azml_job, operation="stream", name="job1")
    assert cmd[:4] == ["az", "ml", "job", "stream"]
    assert "--name" in cmd and "job1" in cmd
    _assert_minimal_output(cmd)


def test_endpoint_update_traffic(scope_env):
    _, cmd = _run_tool(
        mcp_server.azml_online_endpoint,
        operation="update",
        name="tw-sentiment",
        traffic="blue=100",
    )
    assert "update" in cmd and "--traffic" in cmd and "blue=100" in cmd
    _assert_minimal_output(cmd)


def test_endpoint_delete_auto_confirms(scope_env):
    _, cmd = _run_tool(
        mcp_server.azml_online_endpoint, operation="delete", name="tw-sentiment"
    )
    assert "delete" in cmd and "--yes" in cmd
    _assert_minimal_output(cmd)


def test_deployment_get_logs(scope_env):
    _, cmd = _run_tool(
        mcp_server.azml_online_deployment,
        operation="get-logs",
        name="blue",
        endpoint="tw-sentiment",
        lines=50,
    )
    assert "get-logs" in cmd
    assert "--lines" in cmd and "50" in cmd
    assert "--endpoint-name" in cmd and "tw-sentiment" in cmd
    _assert_minimal_output(cmd)


def test_set_args_forwarded(spec_yaml, scope_env):
    _, cmd = _run_tool(
        mcp_server.azml_data,
        operation="create",
        file=spec_yaml,
        set_args=["tags.env=dev"],
    )
    assert "--set" in cmd and "tags.env=dev" in cmd


# --- output handling --------------------------------------------------------


def test_long_output_truncated(scope_env):
    with patch.object(mcp_server.subprocess, "run") as mock_run:
        mock_run.return_value = _ok(stdout="x" * 9000)
        out = mcp_server.azml_environment("list")
        assert "truncated" in out
        assert len(out) < 9000


def test_stream_timeout_returns_partial(scope_env):
    with patch.object(mcp_server.subprocess, "run") as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired(
            cmd="az ml job stream", timeout=180, output="partial-logs"
        )
        out = mcp_server.azml_job("stream", name="job1")
        assert "returncode: 124" in out
        assert "partial-logs" in out


# --- existing tools keep minimal output -------------------------------------


def test_azcopy_rejects_verbosity_override():
    with pytest.raises(ToolError):
        mcp_server.azcopy_upload("src", "dst", ["--output-level=json"])
