"""Single authenticated exec-gateway MCP server.

Runs in its own compose container with secrets from env_file.
Tools are fully specified by dev/tools.yaml (no per-tool Python code).
tools.yaml is read ONCE at startup; edits require a server restart to
take effect. While running, the server sticks to the startup snapshot
and never re-reads the file (no live-editing / no file watcher).
Each invocation: validate argv -> broker.clean_env(creds) ->
subprocess (no shell) -> redact stdout/stderr -> audit log.

Every tool has the same CLI-like surface:
  call(argv: list[str], show_help: bool = False)
The agent passes the FULL argv including the executable, e.g.
  azml_job_submit(argv=["az", "ml", "job", "create", "--file", "job.yaml"])
The server checks argv starts with the tool's base_argv, applies global
checks (shell-meta rejection) to every element, applies per-argument
restrictions from tools.yaml, and appends configured defaults for missing
flags. Path confinement is opt-in per argument via `path: true`.

tools.yaml schema per tool:
  name: str (required)
  description: str
  creds: [env names]
  needs_az_login: bool
  base_argv: [exe, ...] (required; enforced as argv prefix)
  help_argv: [...] (default [--help])
  argv_allow_regex: str (optional catch-all every trailing element must match)
  params (each entry selects named flags OR positionals, not both):
    - names: [--flag, -f]  (named arguments; aliases)
      positional: 1 (or [1, 2]; 1-based index among non-flag tokens)
      allow_regex: str (optional, each occurrence's value must match)
      path: bool (optional; value is confined to the workspace)
      default: scalar (optional literal appended when flag absent; named only)
      default_from: str (optional key into top-level `defaults:`; named only)
      default_flag: str (spelling used when auto-appending; default: first name)

Flags are parsed as `--flag value` or `--flag=value`. Unlisted flags,
lone boolean flags, and positionals are allowed (globals only).
"""

from __future__ import annotations

import inspect
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, NamedTuple

import yaml
from broker import Broker
from mcp.server.mcpserver import MCPServer
from redact import redact

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent  # /workspace when mounted
TOOLS_YAML = Path(os.environ.get("TOOLS_YAML", HERE.parent / "tools.yaml"))
# Common ancestor for gateway logs; override AUDIT_LOG/AZ_LOGIN_LOG to split them.
LOG_DIR = Path(os.environ.get("LOG_DIR", "/tmp"))
AUDIT_LOG = Path(os.environ.get("AUDIT_LOG", LOG_DIR / "tool-gateway-audit.jsonl"))
AZ_LOGIN_LOG = Path(
    os.environ.get("AZ_LOGIN_LOG", LOG_DIR / "tool-gateway-azure-login.jsonl")
)
TIMEOUT_S = int(os.environ.get("TOOL_TIMEOUT_S", "120"))
MAX_OUTPUT = int(os.environ.get("TOOL_MAX_OUTPUT", "262144"))

SHELL_META = re.compile("[;|&$`!\n\r]")
AZ_LOGIN_TIMEOUT_S = int(os.environ.get("AZ_LOGIN_TIMEOUT_S", "30"))

broker = Broker()
server = MCPServer("tool-gateway")
_az_login_lock = threading.Lock()

# name -> {"cfg": tool cfg, "fn": registered function}
REGISTERED: dict[str, dict[str, Any]] = {}


def _load_manifest() -> dict:
    """Read tools.yaml from disk. Called ONCE at startup only."""
    with open(TOOLS_YAML) as f:
        return yaml.safe_load(f) or {}


# Startup snapshot: tools.yaml is never re-read after this point, so
# live edits to the file have no effect until the server restarts.
_MANIFEST: dict = _load_manifest()
_DEFAULTS: dict = _MANIFEST.get("defaults", {}) or {}
_TOOL_CFGS: dict[str, dict] = {
    t.get("name"): t for t in _MANIFEST.get("tools", []) if t.get("name")
}


def _tool_cfg(name: str) -> dict:
    try:
        return _TOOL_CFGS[name]
    except KeyError:
        raise ValueError(f"unknown tool: {name}") from None


def _defaults() -> dict:
    return _DEFAULTS


def _format_exception(e: BaseException) -> str:
    """Unwrap ExceptionGroup/TaskGroup to the real message(s)."""
    if isinstance(e, BaseExceptionGroup):
        parts = [_format_exception(sub) for sub in e.exceptions]
        return "; ".join(p for p in parts if p)
    return f"{type(e).__name__}: {e}"


def _error_result(e: BaseException) -> str:
    redacted, _ = redact(_format_exception(e), broker.secret_values())
    return f"returncode: 1\nerror: {redacted}"


def _creds_env(cfg: dict, env: dict[str, str]) -> dict[str, str]:
    """Subset of env holding only the tool's `creds:` secrets (for audit)."""
    creds = cfg.get("creds", []) or []
    return {name: env[name] for name in creds if name in env}


def _audit(
    tool: str,
    argv: list[str],
    rc: int,
    output: str,
    hits: int,
    secrets: dict[str, str] | None = None,
) -> None:
    """Append one JSONL record with the redacted output and creds secrets."""
    try:
        AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(AUDIT_LOG, "a") as f:
            f.write(
                json.dumps(
                    {
                        "ts": time.time(),
                        "tool": tool,
                        "argv": argv,
                        "argc": len(argv),
                        "rc": rc,
                        "bytes": len(output),
                        "output": output,
                        "secrets": secrets or {},
                        "heuristic_hits": hits,
                    }
                )
                + "\n"
            )
    except Exception:  # noqa: BLE001, S110 - audit must never break tools
        pass


def _az_log(event: str, **fields: Any) -> None:
    """Append one JSONL azure-login record next to the audit trail.

    String fields are passed through the same redaction as tool output so
    credential material never lands on disk. Never raises: login logging
    must not affect tool invocations.
    """
    try:
        secrets = broker.secret_values()
        safe = {
            k: (redact(v, secrets)[0] if isinstance(v, str) else v)
            for k, v in fields.items()
        }
        AZ_LOGIN_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(AZ_LOGIN_LOG, "a") as f:
            f.write(json.dumps({"ts": time.time(), "event": event, **safe}) + "\n")
    except Exception:  # noqa: BLE001, S110 - login logging must never break tools
        pass


def _ensure_az_login(env: dict[str, str]) -> str | None:
    """Ensure `az` is authenticated using only the scoped env. Returns error or None.
    The lock ensures concurrent login calls don't trigger multiple logins."""
    client_id = env.get("AZURE_CLIENT_ID")
    tenant_id = env.get("AZURE_TENANT_ID")
    client_secret = env.get("AZURE_CLIENT_SECRET")
    if not client_id or not tenant_id or not client_secret:
        _az_log("missing_credentials")
        return "returncode: 1\nerror: azure login unavailable: missing credentials"
    with _az_login_lock:
        try:
            already = subprocess.run(
                ["az", "account", "show", "--output", "none"],
                env=env,
                capture_output=True,
                text=True,
                timeout=AZ_LOGIN_TIMEOUT_S,
                check=False,
            )
        except (FileNotFoundError, subprocess.SubprocessError, OSError):
            already = None
        except Exception:  # noqa: BLE001 - fall through to login attempt
            already = None
        if already is not None and already.returncode == 0:
            _az_log("already_authenticated")
            return None
        try:
            result = subprocess.run(
                [
                    "az",
                    "login",
                    "--service-principal",
                    "--username",
                    client_id,
                    "--tenant",
                    tenant_id,
                    "--password",
                    client_secret,
                    "--output",
                    "none",
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=AZ_LOGIN_TIMEOUT_S,
                check=False,
            )
        except FileNotFoundError:
            _az_log("executable_not_found")
            return "returncode: 127\nerror: executable not found: az"
        except Exception as e:  # noqa: BLE001 - return errors as tool output
            _az_log("login_error", error=f"{type(e).__name__}: {e}")
            redacted, _ = redact(f"{type(e).__name__}: {e}", broker.secret_values())
            return f"returncode: 1\nerror: azure login failed: {redacted}"
        if result.returncode != 0:
            detail = (result.stderr or "").strip() or "login failed"
            _az_log("login_failed", rc=result.returncode, detail=detail)
            redacted, _ = redact(detail, broker.secret_values())
            return f"returncode: 1\nerror: azure login failed: {redacted}"
        _az_log("login_ok")
        return None


def _run_cfg(cfg: dict, argv: list[str], skip_az_login: bool = False) -> str:
    tool_name = cfg.get("name", "unknown")
    try:
        env = broker.clean_env(cfg.get("creds", []))
        if (
            cfg.get("needs_az_login")
            and not skip_az_login
            and (err := _ensure_az_login(env))
        ):
            _audit(tool_name, argv, 1, err, 0, _creds_env(cfg, env))
            return err
    except Exception as e:  # noqa: BLE001 - return errors as tool output
        return _error_result(e)
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
        _audit(tool_name, argv, proc.returncode, redacted, hits, _creds_env(cfg, env))
        prefix = f"returncode: {proc.returncode}\n"
        return prefix + redacted
    except subprocess.TimeoutExpired as e:
        partial = (e.output or "") if isinstance(e.output, str) else ""
        redacted, hits = redact(partial, broker.secret_values())
        _audit(tool_name, argv, 124, redacted, hits, _creds_env(cfg, env))
        return f"returncode: 124\ntimeout after {TIMEOUT_S}s\n{redacted}"
    except FileNotFoundError:
        return f"returncode: 127\nerror: executable not found: {argv[0]}"
    except Exception as e:  # noqa: BLE001 - return errors as tool output
        return _error_result(e)


def _check_token(value: str) -> str:
    """Global check for every argv element: reject shell metacharacters."""
    if SHELL_META.search(value):
        raise ValueError(f"rejected shell metacharacters in: {value!r}")
    return value


def _maybe_confine(value: str) -> str:
    """Confine path-like values to the workspace; leave plain tokens untouched.

    Only called for arguments marked `path: true` in tools.yaml. Absolute
    paths must resolve inside REPO_ROOT. Relative values that look like paths
    (contain a slash, parent refs, or obvious file endings) are resolved
    against REPO_ROOT, rejected on escape, and returned absolute.
    Plain tokens (flags, names, URLs, IDs) pass through unchanged.
    """
    if not isinstance(value, str) or not value:
        return value
    stripped = value.strip()
    if (
        # "://" never occurs in a workspace path; a positional marked
        # `path: true` may still hold a URL (e.g. azcopy source/dest), so
        # URLs always pass through.
        "://" in value
        # A JSON object is never a path (e.g. runpodctl --env '{"K":"V"}').
        or (stripped.startswith("{") and stripped.endswith("}"))
    ):
        return value
    root = REPO_ROOT.resolve()
    p = Path(value)
    if p.is_absolute():
        resolved = p.resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            raise ValueError(f"path outside workspace: {value!r}") from None
        return str(resolved)
    looks_like_path = (
        "/" in value
        or value in (".", "..")
        or value.startswith((".", "./", "../", "~/"))
        or ".." in Path(value).parts
        or re.search(r"\.(ya?ml|json|txt|csv|parquet|bin|pt|pth|onnx)$", value)
        is not None
    )
    if not looks_like_path:
        return value
    resolved = (root / value).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError(f"path outside workspace: {value!r}") from None
    return str(resolved)


def _spec_is_positional(spec: dict) -> bool:
    return spec.get("positional") is not None


def _spec_names(spec: dict) -> list[str]:
    if spec.get("names"):
        return [str(n) for n in spec["names"]]
    return [str(spec.get("name"))]


def _spec_positionals(spec: dict) -> list[int]:
    """Return validated 1-based positional indices (int or list in yaml)."""
    raw = spec.get("positional")
    if raw is None:
        return []
    if isinstance(raw, bool):
        raise ValueError(  # noqa: TRY004 - config error, not a type error
            f"positional must be an int or list of ints: {spec!r}"
        )
    if isinstance(raw, int):
        values: list[Any] = [raw]
    elif isinstance(raw, (list, tuple)):
        values = list(raw)
    else:
        raise ValueError(  # noqa: TRY004 - config error, not a type error
            f"positional must be an int or list of ints: {spec!r}"
        )
    indices: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"positional indices must be ints >= 1: {spec!r}")
        indices.append(value)
    return indices


class _Arg(NamedTuple):
    flag: str | None
    value: str | None
    joined: bool
    pos_index: int | None


def _parse_flags(trailing: list[str]) -> list[_Arg]:
    """Parse trailing argv into records.

    `--flag value`, `--flag=value`, lone `--flag` (value None), and
    positionals (flag None) carrying their 0-based index among positionals.
    """
    records: list[_Arg] = []
    pidx = 0
    i = 0
    while i < len(trailing):
        tok = trailing[i]
        if tok.startswith("-") and len(tok) > 1 and "=" in tok:
            flag, _, val = tok.partition("=")
            records.append(_Arg(flag, val, True, None))
            i += 1
        elif tok.startswith("-") and len(tok) > 1:
            if i + 1 < len(trailing) and not (
                trailing[i + 1].startswith("-") and len(trailing[i + 1]) > 1
            ):
                records.append(_Arg(tok, trailing[i + 1], False, None))
                i += 2
            else:
                records.append(_Arg(tok, None, False, None))
                i += 1
        else:
            records.append(_Arg(None, tok, False, pidx))
            pidx += 1
            i += 1
    return records


def _check_value(spec: dict | None, value: str) -> str:
    """Check an argument value: shell-meta always; confine if `path: true`."""
    if SHELL_META.search(value):
        raise ValueError(f"rejected shell metacharacters in: {value!r}")
    if spec is not None and spec.get("path"):
        return _maybe_confine(value)
    return value


def _validate_argv(cfg: dict, defaults: dict, argv: list[str]) -> list[str]:
    """Validate full argv: prefix, globals, flag rules, catch-all, defaults."""
    if not isinstance(argv, list) or not argv:
        raise ValueError("argv required: full command argv as a list of strings")
    base_argv = [str(x) for x in (cfg.get("base_argv") or [])]
    if [str(x) for x in argv[: len(base_argv)]] != base_argv:
        raise ValueError(
            f"argv must start with {' '.join(base_argv)}: got {' '.join(str(x) for x in argv[: len(base_argv)])!r}"
        )
    trailing = [str(x) for x in argv[len(base_argv) :]]
    catch_all = cfg.get("argv_allow_regex")
    catch_rx = re.compile(catch_all) if catch_all else None
    specs = [
        p
        for p in (cfg.get("params") or [])
        if p.get("name") or p.get("names") or _spec_is_positional(p)
    ]
    rules: dict[str, dict] = {}
    pos_rules: dict[int, dict] = {}
    for spec in specs:
        if _spec_is_positional(spec):
            for index in _spec_positionals(spec):
                pos_rules[index - 1] = spec
        else:
            for n in _spec_names(spec):
                rules[n] = spec

    records = _parse_flags(trailing)
    for rec in records:
        spec = (
            rules.get(rec.flag)
            if rec.flag is not None
            else pos_rules.get(rec.pos_index)
        )
        if spec is None:
            continue  # unlisted argument: passthrough
        if rec.flag is not None and rec.value is None:
            if spec.get("allow_regex"):
                raise ValueError(f"flag {rec.flag!r} needs a value")
            continue
        if (
            rec.value is not None
            and spec.get("allow_regex")
            and not re.search(spec["allow_regex"], rec.value)
        ):
            raise ValueError(f"argument {rec.value!r} not allowed")

    # Auto-append defaults for absent named flags (positionals cannot default).
    present = {rec.flag for rec in records if rec.flag is not None}
    extra: list[str] = []
    for spec in specs:
        if _spec_is_positional(spec):
            continue
        names = _spec_names(spec)
        if any(n in present for n in names):
            continue
        dflt = spec.get("default", None)
        if dflt is None and spec.get("default_from") is not None:
            dflt = defaults.get(spec["default_from"])
        if dflt is None:
            continue
        extra.extend([str(spec.get("default_flag") or names[0]), str(dflt)])

    full = [str(x) for x in argv] + extra
    checked = full[: len(base_argv)]
    toks = full[len(base_argv) :]
    for tok in toks:
        if catch_rx and not catch_rx.search(tok):
            raise ValueError(f"argument not allowed: {tok!r}")
    # Walk records (preserving `--flag=value` shape) so each value is checked
    # with its argument's context (path confinement is opt-in via `path: true`).
    for rec in _parse_flags(toks):
        spec = (
            rules.get(rec.flag)
            if rec.flag is not None
            else pos_rules.get(rec.pos_index)
        )
        if rec.flag is None:
            checked.append(_check_value(spec, rec.value))
        elif rec.value is None:
            checked.append(_check_token(rec.flag))
        elif rec.joined:
            checked.append(_check_token(rec.flag) + "=" + _check_value(spec, rec.value))
        else:
            checked.append(_check_token(rec.flag))
            checked.append(_check_value(spec, rec.value))
    return checked


def _appendix(cfg: dict) -> str:
    """Plain-text restrictions note appended to every MCP description."""
    lines = ["", "Restrictions:"]
    lines.append(
        f"Full argv must start with: {' '.join(str(x) for x in cfg.get('base_argv', []))}"
    )
    for spec in cfg.get("params") or []:
        if _spec_is_positional(spec):
            label = "positional " + ", ".join(str(i) for i in _spec_positionals(spec))
            names: list[str] = []
        else:
            names = _spec_names(spec)
            label = "/".join(names)
        bits = [label]
        if spec.get("allow_regex"):
            bits.append(f"value must match {spec['allow_regex']}")
        if spec.get("path"):
            bits.append("value is a path confined to the workspace")
        if names and spec.get("default_from") is not None:
            bits.append(
                f"default from defaults.{spec['default_from']} appended as {spec.get('default_flag') or names[0]} when absent"
            )
        elif names and "default" in spec:
            bits.append(f"default {spec['default']!r} appended when absent")
        lines.append("- " + ", ".join(bits))
    if cfg.get("argv_allow_regex"):
        lines.append(f"- every argument must match {cfg['argv_allow_regex']}")
    lines.append(
        "Unlisted flags and positionals are allowed (no restrictions except globals)."
    )
    lines.append(
        "Globals: shell metacharacters rejected; paths confined only for arguments marked path: true."
    )
    lines.append("Pass show_help=true for the command's own help output.")
    return "\n".join(lines)


def _make_fn(cfg: dict, defaults: dict):
    name = cfg["name"]
    base_desc = cfg.get("description", "")
    full_desc = (base_desc + "\n" + _appendix(cfg)).strip()

    def fn(argv: list[str] | None = None, show_help: bool = False) -> str:
        try:
            if show_help:
                help_argv = cfg.get("help_argv", ["--help"])
                return _run_cfg(
                    cfg, [*cfg.get("base_argv", []), *help_argv], skip_az_login=True
                )
            checked = _validate_argv(cfg, defaults, list(argv or []))
            return _run_cfg(cfg, checked)
        except Exception as e:  # noqa: BLE001 - never raise; MCP wraps raises in ExceptionGroup
            return _error_result(e)

    fn.__name__ = name
    fn.__doc__ = full_desc
    fn.__annotations__ = {"argv": list[str] | None, "show_help": bool, "return": str}
    fn.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        [
            inspect.Parameter(
                "argv",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=None,
                annotation=list[str] | None,
            ),
            inspect.Parameter(
                "show_help",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=False,
                annotation=bool,
            ),
        ]
    )
    return fn


def _register_all(manifest: dict | None = None) -> None:
    # Default is the startup snapshot; never re-read tools.yaml here so
    # live edits cannot add/replace tools without a restart.
    manifest = manifest if manifest is not None else _MANIFEST
    defaults = manifest.get("defaults", {}) or {}
    seen: set[str] = set()
    for cfg in manifest.get("tools", []):
        # Fail-closed config validation at startup.
        if not cfg.get("name") or not cfg.get("base_argv"):
            raise ValueError(f"tool entry needs name + base_argv: {cfg!r}")
        if cfg["name"] in seen:
            raise ValueError(f"duplicate tool name: {cfg['name']!r}")
        seen.add(cfg["name"])
        if cfg.get("argv_allow_regex"):
            re.compile(cfg["argv_allow_regex"])  # raise early on bad regex
        for spec in cfg.get("params") or []:
            if _spec_is_positional(spec):
                if spec.get("name") or spec.get("names"):
                    raise ValueError(
                        f"param cannot have both names and positional: {spec!r}"
                    )
                positionals = _spec_positionals(spec)  # validates ints >= 1
                names: list[str] = []
            else:
                positionals = []
                names = _spec_names(spec)
                if any(not n.startswith("-") for n in names):
                    raise ValueError(
                        f"param names must be flags starting with '-': {spec!r}"
                    )
            if bool(names) == bool(positionals):
                raise ValueError(
                    f"param needs exactly one of names or positional: {spec!r}"
                )
            if spec.get("allow_regex"):
                re.compile(spec["allow_regex"])  # raise early on bad regex
            if "path" in spec and not isinstance(spec["path"], bool):
                raise ValueError(f"param 'path' must be true/false: {spec!r}")
            if positionals and any(
                key in spec for key in ("default", "default_from", "default_flag")
            ):
                raise ValueError(f"positional param cannot have defaults: {spec!r}")
        fn = _make_fn(cfg, defaults)
        server.add_tool(
            fn,
            name=cfg["name"],
            description=(cfg.get("description", "") + "\n" + _appendix(cfg)).strip(),
        )
        REGISTERED[cfg["name"]] = {"cfg": cfg, "fn": fn}


def call_tool(name: str, **kwargs: Any) -> str:
    """Dispatch to a registered tool (used by tests and local debugging)."""
    entry = REGISTERED.get(name)
    if entry is None:
        raise ValueError(f"unknown tool: {name}")
    return entry["fn"](**kwargs)


_register_all()


def main() -> None:
    broker.scrub_startup_env()
    server.run(transport="streamable-http", host="0.0.0.0", port=8004)


if __name__ == "__main__":
    main()
