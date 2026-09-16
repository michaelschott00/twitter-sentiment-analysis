"""Smoke-test the single tool-gateway MCP server.

Every gateway tool has the same surface: call(argv, show_help).
argv is the FULL command argv including the executable.

Run with:
    python dev/mcp_client.py --list
    python dev/mcp_client.py --call gh_pr --argv gh pr --argv status
    python dev/mcp_client.py --call gh_pr --args-json '{"argv": ["gh", "pr", "status"]}'
    python dev/mcp_client.py --call azml_job_list --args-json '{"argv": ["az", "ml", "job", "list"]}'
    python dev/mcp_client.py --call azml_job_list --show-help
    Pass --json for machine-readable JSON output (default is human-readable).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

import click
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://tool-gateway:8004/mcp")


def _format_result(result) -> str:
    """Extract plain text from a CallToolResult (list of content blocks)."""
    parts: list[str] = []
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        parts.append(text if text is not None else str(block))
    if parts:
        return "\n".join(parts)
    return str(result)


def _format_error(e: BaseException) -> str:
    """Unwrap ExceptionGroup/TaskGroup to the real underlying message."""
    if isinstance(e, BaseExceptionGroup):
        parts = [_format_error(sub) for sub in e.exceptions]
        return "; ".join(p for p in parts if p)
    return f"{type(e).__name__}: {e}"


async def _list(url: str, timeout: float) -> dict:
    try:
        async with streamable_http_client(url) as (read, write):  # noqa: SIM117
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                return {
                    "ok": True,
                    "tools": [
                        {"name": t.name, "description": t.description or ""}
                        for t in result.tools
                    ],
                }
    except Exception as e:  # noqa: BLE001 - smoke test reports failures
        return {"ok": False, "error": _format_error(e)}


async def _call(url: str, tool: str, args: dict, timeout: float) -> dict:
    try:
        async with streamable_http_client(url) as (read, write):  # noqa: SIM117
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    tool, args, read_timeout_seconds=timeout
                )
                return {
                    "ok": not getattr(result, "isError", False),
                    "result": _format_result(result),
                }
    except Exception as e:  # noqa: BLE001 - smoke test reports failures
        return {"ok": False, "error": _format_error(e)}


@click.command()
@click.option("--url", default=GATEWAY_URL, show_default=True)
@click.option("--timeout", type=float, default=30, show_default=True)
@click.option("--list", "list_tools", is_flag=True, help="List gateway tools.")
@click.option("--call", "call_tool", default=None, help="Tool name to call.")
@click.option(
    "--argv",
    multiple=True,
    default=(),
    help="Full argv element (repeatable). Combined with --show-help into args.",
)
@click.option(
    "--show-help", is_flag=True, help="Call with show_help=true (no argv needed)."
)
@click.option(
    "--args-json",
    default=None,
    help='Full tool args as JSON, e.g. \'{"argv": ["gh", "pr", "status"]}\'.',
)
@click.option("--json", "as_json", is_flag=True)
def main(
    url: str,
    timeout: float,
    list_tools: bool,
    call_tool: str | None,
    argv: tuple[str, ...],
    show_help: bool,
    args_json: str | None,
    as_json: bool,
) -> None:
    if call_tool:
        if args_json:
            args = json.loads(args_json)
        else:
            args: dict = {}
            if argv:
                args["argv"] = list(argv)
            if show_help:
                args["show_help"] = True
        res = asyncio.run(_call(url, call_tool, args, timeout))
    else:
        res = asyncio.run(_list(url, timeout))
    if as_json:
        click.echo(json.dumps(res, indent=2))
    elif not res.get("ok"):
        click.echo(f"Error: {res.get('error')}", err=True)
    elif "tools" in res:
        for t in res["tools"]:
            click.echo(f"{t['name']}\n  {t['description']}")
    else:
        click.echo(res.get("result", ""))
    if not res.get("ok"):
        sys.exit(1)


if __name__ == "__main__":
    main()
