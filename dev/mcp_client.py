"""Smoke-test the single tool-gateway MCP server.

Run with:
    python dev/mcp_client.py --list
    python dev/mcp_client.py --call gh --args-json '{"args": ["--version"]}'
    python dev/mcp_client.py --call azml_job_list --args-json '{}'
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

import click
from mcp import Client

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://tool-gateway:8004/mcp")


async def _list(url: str, timeout: float) -> dict:
    try:
        async with Client(url, read_timeout_seconds=timeout) as client:
            result = await client.list_tools()
            return {
                "ok": True,
                "tools": [
                    {"name": t.name, "description": t.description or ""}
                    for t in result.tools
                ],
            }
    except Exception as e:  # noqa: BLE001 - smoke test reports failures
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


async def _call(url: str, tool: str, args: dict, timeout: float) -> dict:
    try:
        async with Client(url, read_timeout_seconds=timeout) as client:
            result = await client.call_tool(tool, args)
            return {"ok": True, "result": str(result)}
    except Exception as e:  # noqa: BLE001 - smoke test reports failures
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


@click.command()
@click.option("--url", default=GATEWAY_URL, show_default=True)
@click.option("--timeout", type=float, default=30, show_default=True)
@click.option("--list", "list_tools", is_flag=True, help="List gateway tools.")
@click.option("--call", "call_tool", default=None, help="Tool name to call.")
@click.option("--args-json", default="{}", show_default=True)
@click.option("--json", "as_json", is_flag=True)
def main(
    url: str,
    timeout: float,
    list_tools: bool,
    call_tool: str | None,
    args_json: str,
    as_json: bool,
) -> None:
    if call_tool:
        args = json.loads(args_json)
        res = asyncio.run(_call(url, call_tool, args, timeout))
    else:
        res = asyncio.run(_list(url, timeout))
    if True:  # always JSON for now
        click.echo(json.dumps(res, indent=2))
    if not res.get("ok"):
        sys.exit(1)


if __name__ == "__main__":
    main()
