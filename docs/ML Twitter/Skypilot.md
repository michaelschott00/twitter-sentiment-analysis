# SkyPilot

## Why introduce SkyPilot?

RunPod has availability problems. SkyPilot lets a single workload be described
once and routed to whichever provider/region currently has capacity (RunPod
today; more clouds can be enabled later), instead of being pinned to one
provider.

## Architecture: SkyPilot runs in the tool gateway

This repo keeps cloud credentials out of the agent (`dev`) container: they live
only in the `tool-gateway` container, which exposes authenticated CLIs as MCP
tools (`dev/tools.yaml`). SkyPilot follows the same rule.

- SkyPilot is installed in the **tool-gateway** image (`dev/tool-gateway/Dockerfile`),
  in its own venv at `/opt/sky` (mirroring the `/opt/azcli` isolation in
  `dev/dev.Dockerfile`) so it cannot perturb the pinned azure-cli/mlflow stack.
  Only the `sky` entrypoint is exposed on `PATH`.
- The agent reaches it through the **`sky` tool** in `dev/tools.yaml`
  (`creds: [RUNPOD_API_KEY]`). The agent container never sees the key.
- SkyPilot's RunPod support (and the `runpod` SDK) read the key from
  `~/.runpod/config.toml`, **not** from `RUNPOD_API_KEY`. `compose.yaml`
  bind-mounts the host file `../.runpod/config.toml` read-only onto that exact
  path. Create it before `docker compose up` (e.g. `runpod config` on the host,
  or a hand-written `[default]` / `api_key = "..."` file); it must be readable
  by uid 1000. Docker would otherwise materialize the missing path as a
  directory.
- `rsync` and `openssh-client` are installed in the gateway because SkyPilot's
  command runner shells out to `rsync`/`ssh` for workdir sync and remote exec.
- `~/.sky` (SkyPilot's cluster/managed-job state) is a named volume
  (`sky-state`) so tracked RunPod pods are not orphaned when the gateway
  restarts. The credential file is bind-mounted, not stored in the image.

```
agent (dev)          tool-gateway                      RunPod
  |  MCP: sky(...)  |  host config.toml -> ~/.runpod/config.toml (ro)
  |---------------> |  sky CLI (/opt/sky) ------------>  provision pods
  |                 |  state: ~/.sky (volume)
  |<-- redacted ----|
```

## Usage

The `sky` tool is a full-CLI passthrough: pass the **entire argv including the
executable**, exactly as you would type it.

```python
sky(argv=["sky", "check"])
sky(argv=["sky", "status"])
sky(argv=["sky", "launch", "-c", "train", "task.yaml", "--gpus", "H100:1"])
sky(argv=["sky", "jobs", "queue"])
sky(argv=["sky", "down", "train"])
```

Verify RunPod is enabled (credentials are injected by the gateway):

```
sky check runpod
```

## Operational notes

- Calls are not bounded by a gateway timeout (the gateway is non-interactive,
  though). For long-lived work prefer non-streaming modes: `sky launch -d`
  (detach, leaves a cluster) or `sky jobs launch` (managed lifecycle), then
  poll with `sky status` / `sky jobs queue`. Pass `-y/--yes` to commands that
  would otherwise prompt for confirmation (e.g. `sky down`).
- SkyPilot auto-starts a local API server on first use; state persists in the
  `sky-state` volume.
- Enabling additional providers later means adding their credentials to the
  gateway `env_file` and installing `skypilot[<cloud>]` in
  `dev/tool-gateway/Dockerfile`.
