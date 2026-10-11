# Tasks

- [x] Add azure secrets to tools.yaml and to job yaml
- [ ] Add instructions to use `--secrets`
- [ ] Replace directory sync with git clone (same `workdir` variable but different setting)
- [ ] Build a CI/CD pipeline with notifications

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

## Training on RunPod

Real training is described once in `infra/skypilot/train-clf.yaml`. SkyPilot
does all the orchestration:

- **`workdir` is a git repo** — SkyPilot clones the sources onto the pod, so
  source edits (`twitter/**`, `configs/**`) never need an image rebuild. Only a
  dependency change (`pyproject.toml`) triggers the image build (see
  `.github/workflows/build-push-runpod.yml`).
- **`setup`** downloads the `twitter-splits` data asset via
  `/usr/local/bin/fetch_splits.py` (SP-auth via `DefaultAzureCredential`).
- **`run`** invokes the same LightningCLI command as the original RunPod
  template, logging to the Azure ML workspace via `MLFLOW_TRACKING_URI`.
- **`resources.autostop`** (idle → `down`) tears the pod down automatically, so
  nothing bills after the job finishes. RunPod cannot *stop* pods, only delete
  them, so `autostop` with `down: true` is the cleanup mechanism.

Launch (the values are pulled from the gateway's env via `--secret`/`--env`, so
they never touch the agent container):

```
sky check runpod

sky launch -y -c sky-train \
  --secret AZURE_CLIENT_ID --secret AZURE_CLIENT_SECRET \
  --secret AZURE_TENANT_ID --secret MLFLOW_TRACKING_URI \
  --env AZURE_SUBSCRIPTION_ID \
  infra/skypilot/train-clf.yaml

sky logs sky-train 1 --tail 100     # after the job completes
sky queue sky-train                 # job status
```

The image (`docker:michaelschott00/twitter-sentiment:skypilot`) ships the
pinned dependencies in `/opt/venv` *with no `ENTRYPOINT`*: SkyPilot boots the
pod by passing its own start command, and the old `run.sh` entrypoint (and its
sanity checks / pod self-stop) was incompatible with that, so it was removed.

## Operational notes

- Calls are not bounded by a gateway timeout (the gateway is non-interactive,
  though). For long-lived work prefer non-streaming modes: `sky launch -d`
  (detach, leaves a cluster) or `sky jobs launch` (managed lifecycle), then
  poll with `sky status` / `sky jobs queue`. Pass `-y/--yes` to commands that
  would otherwise prompt for confirmation (e.g. `sky down`).
- SkyPilot auto-starts a local API server on first use; state persists in the
  `sky-state` volume.
- The `sky` tool's `creds` (dev/tools.yaml) includes the RunPod key plus the
  Azure/MLflow values a training pod needs; they are forwarded with
  `--secret NAME` / `--env NAME` (no value → inherited from the gateway env)
  and never reach the agent. Add a new credential to that `creds:` list and to
  the launch command.
- Enabling additional providers later means adding their credentials to the
  gateway `env_file` and installing `skypilot[<cloud>]` in
  `dev/tool-gateway/Dockerfile`.
