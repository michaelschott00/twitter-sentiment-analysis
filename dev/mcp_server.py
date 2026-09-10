import subprocess

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

mcp = MCPServer("Sidecar")


def _format_result(cmd: list[str], result: subprocess.CompletedProcess[str]) -> str:
    out = f"$ {' '.join(cmd)}\n"
    out += f"returncode: {result.returncode}\n"
    if result.stdout:
        out += f"--- stdout ---\n{result.stdout}\n"
    if result.stderr:
        out += f"--- stderr ---\n{result.stderr}\n"
    if not result.stdout and not result.stderr:
        out += "(no output)\n"
    return out


@mcp.tool()
def git_push() -> str:
    """Regular `git push` will not work because ssh is disabled. Use this tool to push to origin-ssh dev."""
    cmd = ["git", "push", "--quiet", "origin-ssh", "dev"]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd="/workspace", check=False
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError) as e:
        return f"$ {' '.join(cmd)}\nreturncode: 1\n--- stderr ---\n{e}\n"
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
        return f"$ {' '.join(cmd)}\nreturncode: 1\n--- stderr ---\n{e}\n"
    return _format_result(cmd, result)


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)
