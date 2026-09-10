import os
import re
import subprocess
import sys
from collections.abc import Callable
from typing import Literal

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

mcp = MCPServer("Sidecar")


# --- output redaction ------------------------------------------------------
# Post-process every tool output / stderr message with redact_secrets() so
# Azure credentials never leak to MCP clients. To add a new secret shape,
# append a (pattern, replacement) entry to _REDACT_PATTERNS below.
# Replacement may be a string or a re.sub callable receiving the match.

_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
# New-style Azure SP secrets contain a "~" segment, e.g. "abc12345~abc...".
_SP_SECRET_TILDE_RE = re.compile(r"\b[A-Za-z0-9_.\-]{3,}~[A-Za-z0-9_.\-~]{16,}\b")
# Labeled secrets, e.g. client_secret=..., "clientSecret": "...", password: ...
_LABELED_SECRET_RE = re.compile(
    r"(?i)(client[_-]?secret|clientSecret|spa[_-]?client[_-]?secret|password)"
    r"(\s*[:=]\s*['\"]?)([^'\"\s;,}]{8,})(['\"]?)"
)


def _labeled_secret_repl(m: re.Match[str]) -> str:
    return f"{m.group(1)}{m.group(2)}[REDACTED-SP-SECRET]{m.group(4)}"


_REDACT_PATTERNS: list[tuple[re.Pattern[str], str | Callable[[re.Match[str]], str]]] = [
    (_UUID_RE, "[REDACTED-UUID]"),
    (_SP_SECRET_TILDE_RE, "[REDACTED-SP-SECRET]"),
    (_LABELED_SECRET_RE, _labeled_secret_repl),
]

# Exact env-var values to redact verbatim when present (defense in depth:
# the SP secret itself may not match a heuristic pattern).
_KNOWN_SECRET_ENV_VARS = (
    "AZCOPY_SPA_CLIENT_SECRET",
    "AZCOPY_SPA_APPLICATION_ID",
    "AZCOPY_TENANT_ID",
)


def redact_secrets(text: str) -> str:
    """Redact Azure credentials/IDs from tool output text.

    Covers UUID-style IDs (XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX) and
    service-principal secrets (tilde-style values, labeled
    client_secret/password assignments, and known secret env-var values
    verbatim). Add new shapes via _REDACT_PATTERNS.
    """
    redacted = text
    for pattern, replacement in _REDACT_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    for var in _KNOWN_SECRET_ENV_VARS:
        value = os.environ.get(var)
        if value and len(value) >= 4 and value in redacted:
            redacted = redacted.replace(value, f"[REDACTED-{var}]")
    return redacted


def _ensure_azure_login() -> None:
    """Authenticate the container with `az login` before serving.

    Required for `azcopy` to work. Uses the service-principal credentials
    already passed into the sidecar container via the environment:
    AZCOPY_SPA_APPLICATION_ID, AZCOPY_TENANT_ID, AZCOPY_SPA_CLIENT_SECRET.

    Skips login when already authenticated or when credentials are missing.
    The client secret is never written to stdout/stderr.
    """
    app_id = os.environ.get("AZCOPY_SPA_APPLICATION_ID")
    tenant_id = os.environ.get("AZCOPY_TENANT_ID")
    client_secret = os.environ.get("AZCOPY_SPA_CLIENT_SECRET")
    if not app_id or not tenant_id or not client_secret:
        print(
            "azure login skipped: AZCOPY_SPA_APPLICATION_ID, AZCOPY_TENANT_ID, "
            "or AZCOPY_SPA_CLIENT_SECRET is not set",
            file=sys.stderr,
        )
        return

    try:
        already = subprocess.run(
            ["az", "account", "show", "--output", "none"],
            capture_output=True,
            text=True,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError) as e:
        print(f"azure login check failed, attempting login: {e}", file=sys.stderr)
        already = None
    if already is not None and already.returncode == 0:
        return

    cmd = [
        "az",
        "login",
        "--service-principal",
        "--username",
        app_id,
        "--tenant",
        tenant_id,
        "--password",
        client_secret,
        "--output",
        "none",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except (FileNotFoundError, subprocess.SubprocessError, OSError) as e:
        print(f"azure login failed: {e}", file=sys.stderr)
        return
    if result.returncode != 0:
        # Do not echo the command (it contains the secret); show stderr only.
        print(
            redact_secrets(
                f"azure login failed (returncode {result.returncode}): "
                f"{result.stderr.strip()}"
            ),
            file=sys.stderr,
        )
    elif result.stderr:
        print(redact_secrets(result.stderr), file=sys.stderr)


def _format_result(cmd: list[str], result: subprocess.CompletedProcess[str]) -> str:
    out = f"$ {' '.join(cmd)}\n"
    out += f"returncode: {result.returncode}\n"
    if result.stdout:
        out += f"--- stdout ---\n{result.stdout}\n"
    if result.stderr:
        out += f"--- stderr ---\n{result.stderr}\n"
    if not result.stdout and not result.stderr:
        out += "(no output)\n"
    return redact_secrets(out)


@mcp.tool()
def git_push() -> str:
    """Regular `git push` will not work because ssh is disabled. Use this tool to push to origin-ssh dev."""
    cmd = ["git", "push", "--quiet", "origin-ssh", "dev"]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd="/workspace", check=False
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError) as e:
        return redact_secrets(
            f"$ {' '.join(cmd)}\nreturncode: 1\n--- stderr ---\n{e}\n"
        )
    return _format_result(cmd, result)


@mcp.tool()
def azcopy_upload(source: str, destination: str, flags: list[str] | None = None) -> str:
    """Running `azcopy` directly will not work because it's not installed. Use this tool to upload data using azcopy.

    Mirrors `azcopy copy [source] [destination] [flags]`.
    Output verbosity is forced to the minimum (--output-level quiet,
    --log-level NONE) and cannot be overridden.
    """
    for f in flags or []:
        name = f.split("=", 1)[0]
        if name in ("--output-level", "--log-level", "--output-type"):
            raise ToolError(
                f"Overriding {name} is not permitted: "
                "output verbosity is fixed to the lowest possible setting."
            )
    cmd = ["azcopy", "copy", source, destination]
    if flags:
        cmd.extend(flags)
    # Lowest possible verbosity: quiet output, no log file output.
    cmd += ["--output-level", "quiet", "--log-level", "NONE"]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd="/workspace", check=False
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError) as e:
        return redact_secrets(
            f"$ {' '.join(cmd)}\nreturncode: 1\n--- stderr ---\n{e}\n"
        )
    return _format_result(cmd, result)


# --- az ml helpers -------------------------------------------------------
# Covers the `az ml` subcommands required by docs/azure-infrastructure-plan.md:
#   environment create (§7.1), data create (§12, Phase 2), job create/show/list
#   (§12), online-endpoint create/update/delete (§9.1, §9.3), online-deployment
#   create (§9.1). Credential-bearing commands (`online-endpoint
#   get-credentials`, `regenerate-keys`) are intentionally NOT exposed, and
#   output is forced to `--output table --only-show-errors` and truncated so
#   no credentials leak through verbose JSON output.

_AZ_ML_TIMEOUT = 300
_AZ_ML_STREAM_TIMEOUT = 180
_AZ_ML_MAX_CHARS = 4000
_WORKSPACE_ROOT = "/workspace"


def _resolve_ml_scope(
    workspace: str | None, resource_group: str | None
) -> tuple[str, str]:
    ws = workspace or os.environ.get("AZUREML_WORKSPACE_NAME")
    rg = resource_group or os.environ.get("AZUREML_RESOURCE_GROUP")
    missing = []
    if not ws:
        missing.append("workspace (or AZUREML_WORKSPACE_NAME env var)")
    if not rg:
        missing.append("resource_group (or AZUREML_RESOURCE_GROUP env var)")
    if missing:
        raise ToolError(f"Missing required scope: {', '.join(missing)}.")
    return ws, rg


def _check_ml_file(file: str | None) -> str:
    if not file:
        raise ToolError("A --file YAML spec path is required for this operation.")
    p = file if os.path.isabs(file) else os.path.join(_WORKSPACE_ROOT, file)
    real = os.path.realpath(p)
    if os.path.commonpath([real, _WORKSPACE_ROOT]) != _WORKSPACE_ROOT:
        raise ToolError(f"File must be inside {_WORKSPACE_ROOT}: {file}")
    if not os.path.isfile(real):
        raise ToolError(f"File not found: {file}")
    if not real.endswith((".yaml", ".yml")):
        raise ToolError(f"File must be a .yaml/.yml spec: {file}")
    return real


def _check_set_args(set_args: list[str] | None) -> list[str]:
    cleaned = []
    for s in set_args or []:
        if not s or s.startswith("-") or "=" not in s:
            raise ToolError(f"Invalid --set override {s!r}: expected key=value format.")
        cleaned.append(s)
    return cleaned


def _truncate(text: str) -> str:
    if len(text) > _AZ_ML_MAX_CHARS:
        return (
            text[:_AZ_ML_MAX_CHARS]
            + f"\n... [truncated {len(text) - _AZ_ML_MAX_CHARS} chars]"
        )
    return text


def _run_az_ml(cmd: list[str], timeout: int = _AZ_ML_TIMEOUT) -> str:
    # Fixed minimal verbosity; callers cannot override output flags.
    cmd = cmd + ["--output", "table", "--only-show-errors"]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=_WORKSPACE_ROOT,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        out = f"$ {' '.join(cmd)}\nreturncode: 124\n"
        stdout = e.stdout if isinstance(e.stdout, str) else ""
        stderr = (e.stderr if isinstance(e.stderr, str) else "") or (
            f"Timed out after {timeout}s."
        )
        if stdout:
            out += f"--- stdout (partial) ---\n{_truncate(stdout)}\n"
        out += f"--- stderr ---\n{_truncate(stderr)}\n"
        return redact_secrets(out)
    except (FileNotFoundError, subprocess.SubprocessError, OSError) as e:
        return redact_secrets(
            f"$ {' '.join(cmd)}\nreturncode: 1\n--- stderr ---\n{e}\n"
        )
    out = f"$ {' '.join(cmd)}\nreturncode: {result.returncode}\n"
    if result.stdout:
        out += f"--- stdout ---\n{_truncate(result.stdout)}\n"
    if result.stderr:
        out += f"--- stderr ---\n{_truncate(result.stderr)}\n"
    if not result.stdout and not result.stderr:
        out += "(no output)\n"
    return redact_secrets(out)


def _ml_scope_args(workspace: str | None, resource_group: str | None) -> list[str]:
    ws, rg = _resolve_ml_scope(workspace, resource_group)
    return ["--workspace-name", ws, "--resource-group", rg]


@mcp.tool()
def azml_data(
    operation: Literal["create", "show", "list"],
    file: str | None = None,
    name: str | None = None,
    version: str | None = None,
    label: str | None = None,
    workspace: str | None = None,
    resource_group: str | None = None,
    set_args: list[str] | None = None,
    no_wait: bool = False,
) -> str:
    """Manage Azure ML data assets (`az ml data ...`, plan §5/§12, Phase 2).

    operation "create" registers a data asset from a YAML spec (e.g.
    twitter-splits); "show"/"list" inspect assets. `file` must be a
    .yaml/.yml path inside /workspace. Output is forced to table format and
    truncated; credential commands are not exposed.
    """
    cmd = ["az", "ml", "data", operation]
    if operation == "create":
        cmd += ["--file", _check_ml_file(file)]
        if no_wait:
            cmd.append("--no-wait")
    elif operation == "show":
        if not name:
            raise ToolError("--name is required for `az ml data show`.")
        cmd += ["--name", name]
        if version:
            cmd += ["--version", version]
        if label:
            cmd += ["--label", label]
    elif operation == "list":
        if name:
            cmd += ["--name", name]
    else:
        raise ToolError(f"Unsupported operation: {operation}")
    for s in _check_set_args(set_args):
        cmd += ["--set", s]
    cmd += _ml_scope_args(workspace, resource_group)
    return _run_az_ml(cmd)


@mcp.tool()
def azml_environment(
    operation: Literal["create", "show", "list"],
    file: str | None = None,
    name: str | None = None,
    version: str | None = None,
    label: str | None = None,
    workspace: str | None = None,
    resource_group: str | None = None,
    set_args: list[str] | None = None,
    no_wait: bool = False,
) -> str:
    """Manage Azure ML environments (`az ml environment ...`, plan §7.1).

    operation "create" registers e.g. twitter-ml-env from a YAML spec;
    "show"/"list" inspect environments. `file` must be a .yaml/.yml path
    inside /workspace. Output is forced to table format and truncated;
    credential commands are not exposed.
    """
    cmd = ["az", "ml", "environment", operation]
    if operation == "create":
        cmd += ["--file", _check_ml_file(file)]
        if no_wait:
            cmd.append("--no-wait")
    elif operation == "show":
        if not name:
            raise ToolError("--name is required for `az ml environment show`.")
        cmd += ["--name", name]
        if version:
            cmd += ["--version", version]
        if label:
            cmd += ["--label", label]
    elif operation == "list":
        if name:
            cmd += ["--name", name]
    else:
        raise ToolError(f"Unsupported operation: {operation}")
    for s in _check_set_args(set_args):
        cmd += ["--set", s]
    cmd += _ml_scope_args(workspace, resource_group)
    return _run_az_ml(cmd)


@mcp.tool()
def azml_job(
    operation: Literal["create", "show", "list", "stream"],
    file: str | None = None,
    name: str | None = None,
    workspace: str | None = None,
    resource_group: str | None = None,
    set_args: list[str] | None = None,
    stream_logs: bool = False,
) -> str:
    """Manage Azure ML jobs (`az ml job ...`, plan §7.2, §12).

    operation "create" submits a job from a YAML spec (requires `file`);
    "show"/"list" inspect jobs; "stream" tails logs of a job (requires
    `name`). Set `stream_logs` to stream logs during "create". `file` must
    be a .yaml/.yml path inside /workspace. Output is forced to table
    format and truncated; logs are capped.
    """
    cmd = ["az", "ml", "job", operation]
    timeout = _AZ_ML_TIMEOUT
    if operation == "create":
        cmd += ["--file", _check_ml_file(file)]
        if stream_logs:
            cmd.append("--stream")
    elif operation in ("show", "stream"):
        if not name:
            raise ToolError(f"--name is required for `az ml job {operation}`.")
        cmd += ["--name", name]
        if operation == "stream":
            timeout = _AZ_ML_STREAM_TIMEOUT
    elif operation == "list":
        pass
    else:
        raise ToolError(f"Unsupported operation: {operation}")
    for s in _check_set_args(set_args):
        cmd += ["--set", s]
    cmd += _ml_scope_args(workspace, resource_group)
    return _run_az_ml(cmd, timeout=timeout)


@mcp.tool()
def azml_online_endpoint(
    operation: Literal["create", "update", "show", "list", "delete"],
    file: str | None = None,
    name: str | None = None,
    traffic: str | None = None,
    workspace: str | None = None,
    resource_group: str | None = None,
    set_args: list[str] | None = None,
    no_wait: bool = False,
) -> str:
    """Manage Azure ML online endpoints (`az ml online-endpoint ...`, §9.1).

    Covers create (endpoint.yaml), update (e.g. traffic "blue=100"), show,
    list, delete (auto-confirmed with --yes). Key retrieval
    (`get-credentials`) is intentionally unavailable to avoid leaking keys.
    Output is forced to table format and truncated.
    """
    cmd = ["az", "ml", "online-endpoint", operation]
    if operation == "create":
        if file:
            cmd += ["--file", _check_ml_file(file)]
        if name:
            cmd += ["--name", name]
        if no_wait:
            cmd.append("--no-wait")
    elif operation == "update":
        if not name:
            raise ToolError("--name is required for online-endpoint update.")
        cmd += ["--name", name]
        if file:
            cmd += ["--file", _check_ml_file(file)]
        if traffic:
            cmd += ["--traffic", traffic]
        if no_wait:
            cmd.append("--no-wait")
        if not file and not traffic and not set_args:
            raise ToolError("Nothing to update: provide file, traffic, or set_args.")
    elif operation in ("show", "delete"):
        if not name:
            raise ToolError(f"--name is required for online-endpoint {operation}.")
        cmd += ["--name", name]
        if operation == "delete":
            cmd += ["--yes"]
            if no_wait:
                cmd.append("--no-wait")
    elif operation == "list":
        pass
    else:
        raise ToolError(f"Unsupported operation: {operation}")
    for s in _check_set_args(set_args):
        cmd += ["--set", s]
    cmd += _ml_scope_args(workspace, resource_group)
    return _run_az_ml(cmd)


@mcp.tool()
def azml_online_deployment(
    operation: Literal["create", "update", "show", "list", "get-logs", "delete"],
    file: str | None = None,
    name: str | None = None,
    endpoint: str | None = None,
    lines: int = 200,
    workspace: str | None = None,
    resource_group: str | None = None,
    set_args: list[str] | None = None,
    no_wait: bool = False,
) -> str:
    """Manage Azure ML online deployments (`az ml online-deployment ...`, §9.1).

    Covers create (deployment-blue.yaml), update, show, list, get-logs
    (container logs, capped via `lines`), delete (auto-confirmed with
    --yes). `endpoint` is the parent online-endpoint name required by
    show/get-logs/delete. Output is forced to table format and truncated.
    """
    cmd = ["az", "ml", "online-deployment", operation]
    if operation == "create":
        if file:
            cmd += ["--file", _check_ml_file(file)]
        if name:
            cmd += ["--name", name]
        if endpoint:
            cmd += ["--endpoint-name", endpoint]
        if no_wait:
            cmd.append("--no-wait")
        if not file and not (name and endpoint):
            raise ToolError("Provide a YAML `file`, or both `name` and `endpoint`.")
    elif operation == "update":
        if not name or not endpoint:
            raise ToolError("--name and --endpoint are required for deployment update.")
        cmd += ["--name", name, "--endpoint-name", endpoint]
        if file:
            cmd += ["--file", _check_ml_file(file)]
        if no_wait:
            cmd.append("--no-wait")
    elif operation in ("show", "get-logs", "delete"):
        if not name or not endpoint:
            raise ToolError(
                f"--name and --endpoint are required for deployment {operation}."
            )
        cmd += ["--name", name, "--endpoint-name", endpoint]
        if operation == "get-logs":
            cmd += ["--lines", str(lines)]
        if operation == "delete":
            cmd += ["--yes"]
            if no_wait:
                cmd.append("--no-wait")
    elif operation == "list":
        if endpoint:
            cmd += ["--endpoint-name", endpoint]
    else:
        raise ToolError(f"Unsupported operation: {operation}")
    for s in _check_set_args(set_args):
        cmd += ["--set", s]
    cmd += _ml_scope_args(workspace, resource_group)
    return _run_az_ml(cmd)


if __name__ == "__main__":
    _ensure_azure_login()
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)
