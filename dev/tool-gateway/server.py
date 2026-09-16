"""Single authenticated exec-gateway MCP server.

Runs in its own compose container with secrets from env_file.
Exposes narrow, allowlisted tools (see dev/tools.yaml). Each invocation:
validate args -> broker.clean_env(creds) -> subprocess (no shell) ->
redact stdout/stderr (best-effort continue) -> audit log.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

import yaml
from broker import Broker
from mcp.server.mcpserver import MCPServer
from redact import redact

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent  # /workspace when mounted
TOOLS_YAML = Path(os.environ.get("TOOLS_YAML", HERE.parent / "tools.yaml"))
AUDIT_LOG = Path(os.environ.get("AUDIT_LOG", "/tmp/tool-gateway-audit.jsonl"))
TIMEOUT_S = int(os.environ.get("TOOL_TIMEOUT_S", "120"))
MAX_OUTPUT = int(os.environ.get("TOOL_MAX_OUTPUT", "262144"))

RG_RE = re.compile(r"^rg-[a-z0-9-]+$")
WS_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{1,60}$")
FLAG_RE = re.compile(r"^--[a-zA-Z0-9][a-zA-Z0-9._-]*$")
SHELL_META = re.compile("[;|&$`!\n\r]")

broker = Broker()
server = MCPServer("tool-gateway")


def _load_manifest() -> dict:
    with open(TOOLS_YAML) as f:
        return yaml.safe_load(f) or {}


def _tool_cfg(name: str) -> dict:
    manifest = _load_manifest()
    for t in manifest.get("tools", []):
        if t.get("name") == name:
            return t
    raise ValueError(f"unknown tool: {name}")


def _defaults() -> dict:
    return _load_manifest().get("defaults", {}) or {}


def _audit(tool: str, argv: list[str], rc: int, out_len: int, hits: int) -> None:
    try:
        AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(AUDIT_LOG, "a") as f:
            f.write(
                json.dumps(
                    {
                        "ts": time.time(),
                        "tool": tool,
                        "argv0": argv[:2],
                        "argc": len(argv),
                        "rc": rc,
                        "bytes": out_len,
                        "heuristic_hits": hits,
                    }
                )
                + "\n"
            )
    except Exception:  # noqa: BLE001, S110 - audit must never break tools
        pass


def _run(tool_name: str, argv: list[str]) -> str:
    cfg = _tool_cfg(tool_name)
    env = broker.clean_env(cfg.get("creds", []))
    try:
        proc = subprocess.run(
            argv,
            env=env,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
            check=False,
        )
        combined = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
        if len(combined) > MAX_OUTPUT:
            combined = combined[:MAX_OUTPUT] + "\n...[truncated]..."
        secrets = broker.secret_values()
        redacted, hits = redact(combined, secrets)
        _audit(tool_name, argv, proc.returncode, len(combined), hits)
        prefix = f"returncode: {proc.returncode}\n"
        return prefix + redacted
    except subprocess.TimeoutExpired as e:
        partial = (e.output or "") if isinstance(e.output, str) else ""
        redacted, hits = redact(partial, broker.secret_values())
        _audit(tool_name, argv, 124, len(partial), hits)
        return f"returncode: 124\ntimeout after {TIMEOUT_S}s\n{redacted}"
    except FileNotFoundError:
        return f"returncode: 127\nerror: executable not found: {argv[0]}"
    except Exception as e:  # noqa: BLE001 - return errors as tool output
        return f"returncode: 1\nerror: {type(e).__name__}: {e}"


def _check_no_shell(items: list[str]) -> None:
    for item in items:
        if SHELL_META.search(item):
            raise ValueError(f"rejected shell metacharacters in: {item!r}")


def _resolve_scope(
    resource_group: str | None, workspace: str | None
) -> tuple[str, str]:
    d = _defaults()
    rg = resource_group or d.get("resource_group", "rg-twitter-ml")
    ws = workspace or d.get("workspace", "mlw-twitter-sentiment")
    if not RG_RE.match(rg):
        raise ValueError(f"invalid resource_group: {rg!r}")
    if not WS_RE.match(ws):
        raise ValueError(f"invalid workspace: {ws!r}")
    return rg, ws


def _resolve_job_yaml(job_yaml: str) -> str:
    # Allow absolute paths under workspace or relative paths; must be .yaml/.yml.
    p = Path(job_yaml)
    if not p.is_absolute():
        p = (REPO_ROOT / job_yaml).resolve()
    else:
        p = p.resolve()
    try:
        p.relative_to(REPO_ROOT.resolve())
    except ValueError:
        raise ValueError(f"job_yaml outside workspace: {job_yaml!r}") from None
    if p.suffix not in (".yaml", ".yml"):
        raise ValueError("job_yaml must be a .yaml/.yml file")
    if not p.is_file():
        raise ValueError(f"job_yaml not found: {job_yaml!r}")
    return str(p)


@server.tool()
def gh(args: list[str]) -> str:
    """Run gh with constrained subcommands. Example: {"args": ["issue","list","--repo","o/r"]}."""
    cfg = _tool_cfg("gh")
    if not args:
        return "returncode: 1\nerror: args required"
    if args[0] not in cfg.get("allow_subcommands", []):
        return f"returncode: 1\nerror: subcommand not allowed: {args[0]!r}"
    _check_no_shell(args)
    if len(args) > 32:
        return "returncode: 1\nerror: too many args"
    return _run("gh", [*cfg["base_argv"], *args])


@server.tool()
def azml_job_submit(
    job_yaml: str,
    resource_group: str | None = None,
    workspace: str | None = None,
) -> str:
    """Submit any AML yaml job file. Defaults to rg-twitter-ml/mlw-twitter-sentiment."""
    try:
        rg, ws = _resolve_scope(resource_group, workspace)
        resolved = _resolve_job_yaml(job_yaml)
    except ValueError as e:
        return f"returncode: 1\nerror: {e}"
    cfg = _tool_cfg("azml_job_submit")
    return _run(
        "azml_job_submit",
        [*cfg["base_argv"], "--file", resolved, "-g", rg, "-w", ws],
    )


@server.tool()
def azml_job_list(
    resource_group: str | None = None,
    workspace: str | None = None,
) -> str:
    """List AML jobs in the workspace (fixed defaults + validated override)."""
    try:
        rg, ws = _resolve_scope(resource_group, workspace)
    except ValueError as e:
        return f"returncode: 1\nerror: {e}"
    cfg = _tool_cfg("azml_job_list")
    return _run("azml_job_list", [*cfg["base_argv"], "-g", rg, "-w", ws])


@server.tool()
def azcopy(args: list[str]) -> str:
    """Run azcopy. URLs constrained to Azure Blob Storage."""
    if not args:
        return "returncode: 1\nerror: args required"
    _check_no_shell(args)
    for a in args:
        if "://" in a and not (
            a.startswith("https://") and ".blob.core.windows.net/" in a
        ):
            return f"returncode: 1\nerror: URL not allowed: {a!r}"
    cfg = _tool_cfg("azcopy")
    return _run("azcopy", [*cfg["base_argv"], *args])


@server.tool()
def runpodctl_pod_create(options: list[str] | None = None) -> str:
    """Create a RunPod pod. Example: {"options": ["--name","x","--gpuType","A100"]}."""
    opts = options or []
    for o in opts:
        if SHELL_META.search(o):
            return f"returncode: 1\nerror: rejected shell metacharacters in: {o!r}"
    # Flags must look like --flags; values are free-form minus shell meta.
    for i, o in enumerate(opts):
        if o.startswith("-") and not FLAG_RE.match(o) and o != "-":
            return f"returncode: 1\nerror: invalid flag: {o!r}"
    if len(opts) > 40:
        return "returncode: 1\nerror: too many options"
    _ = i  # silence lint about unused loop var intent
    cfg = _tool_cfg("runpodctl_pod_create")
    return _run("runpodctl_pod_create", [*cfg["base_argv"], *opts])


def main() -> None:
    broker.scrub_startup_env()
    server.run(transport="streamable-http", host="0.0.0.0", port=8004)


if __name__ == "__main__":
    main()
