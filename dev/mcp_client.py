"""Smoke-test MCP servers from compose.yaml / opencode.json.

Lists the tools each server offers to verify the setup is correct.

Run with:
    python dev/mcp_client.py
    python dev/mcp_client.py --server runpod --verbose
    python dev/mcp_client.py --json
"""

import asyncio
import json
import sys

import click
from mcp import Client

# Keep in sync with compose.yaml service names/ports and opencode.json mcp urls.
SERVERS: dict[str, str] = {
    "runpod": "http://runpod-mcp:8001/mcp",
    "azure-mcp": "http://azure-mcp:8002/mcp",
    "github": "http://github-mcp:8082/mcp",
}


def _format_error(e: BaseException) -> str:
    """Unwrap ExceptionGroups (anyio TaskGroups) to show the root cause."""
    sub_exceptions = getattr(e, "exceptions", None)
    if isinstance(sub_exceptions, (tuple, list)):
        parts = [_format_error(sub) for sub in sub_exceptions]
        return f"{type(e).__name__}: {e} <- [{'; '.join(parts)}]"
    return f"{type(e).__name__}: {e}"


async def list_server_tools(name: str, url: str, timeout: float) -> dict:
    """Connect to one server and list its tools. Never raises."""
    try:
        async with Client(url, read_timeout_seconds=timeout) as client:
            result = await client.list_tools()
            tools = [
                {"name": t.name, "description": t.description or ""}
                for t in result.tools
            ]
            return {"name": name, "url": url, "ok": True, "tools": tools}
    except Exception as e:  # noqa: BLE001 - smoke test: any failure is a FAIL result
        return {"name": name, "url": url, "ok": False, "error": _format_error(e)}


async def check_all(names: tuple[str, ...], timeout: float) -> list[dict]:
    selected = [(n, SERVERS[n]) for n in names]
    results = await asyncio.gather(
        *(list_server_tools(n, u, timeout) for n, u in selected)
    )
    return list(results)


@click.command()
@click.option(
    "--server",
    "-s",
    "servers",
    multiple=True,
    type=click.Choice(list(SERVERS)),
    help="Only check these servers (default: all). Repeatable.",
)
@click.option(
    "--timeout",
    type=float,
    default=30,
    show_default=True,
    help="Per-request read timeout in seconds.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show tool descriptions as well as names.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit machine-readable JSON instead of human-readable text.",
)
def main(
    servers: tuple[str, ...], timeout: float, verbose: bool, as_json: bool
) -> None:
    """List tools offered by each MCP server to verify setup."""
    names = tuple(servers) or tuple(SERVERS)
    results = asyncio.run(check_all(names, timeout))

    if as_json:
        click.echo(json.dumps(results, indent=2))
    else:
        for r in results:
            if r["ok"]:
                click.echo(f"[{r['name']}] OK  {r['url']}  ({len(r['tools'])} tools)")
                for t in r["tools"]:
                    if verbose and t["description"]:
                        first_line = t["description"].splitlines()[0][:120]
                        click.echo(f"  - {t['name']}: {first_line}")
                    else:
                        click.echo(f"  - {t['name']}")
            else:
                click.echo(f"[{r['name']}] FAIL  {r['url']}\n  {r['error']}", err=True)

    if not all(r["ok"] for r in results):
        n_fail = sum(1 for r in results if not r["ok"])
        click.echo(f"\n{n_fail}/{len(results)} server(s) failed.", err=True)
        sys.exit(1)
    if not as_json:
        click.echo(f"\nAll {len(results)} server(s) OK.")


if __name__ == "__main__":
    main()
