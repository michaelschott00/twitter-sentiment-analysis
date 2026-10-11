"""Download the ``twitter-splits`` data-asset CSVs onto a RunPod pod.

Replaces the ``az ml data show`` / ``az storage blob`` calls that used to live
in ``run.sh`` so the image no longer needs the whole azure-cli (and its antlr
pin conflict with omegaconf). Mirrors ``infra/scripts/read_remote_data.py``:
resolve the datastore via ``MLClient``, then read the blobs directly with
``azure-storage-blob`` (bypasses azureml-fsspec's browser login).

Auth is ``DefaultAzureCredential``, which picks up the injected
service-principal env vars (``AZURE_CLIENT_ID`` / ``AZURE_CLIENT_SECRET`` /
``AZURE_TENANT_ID``) automatically.

Run: python fetch_splits.py --resource-group <rg> --workspace <ws> \
        --subscription-id <sub> --dest data/splits
"""

import os
import re
from pathlib import Path

import click
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.storage.blob import ContainerClient


@click.command()
@click.option(
    "--subscription-id",
    default=lambda: os.environ.get("AZURE_SUBSCRIPTION_ID"),
)
@click.option("--resource-group", required=True)
@click.option("--workspace", required=True)
@click.option("--data-name", default="twitter-splits")
@click.option("--data-version", default="1")
@click.option("--dest", default="data/splits", type=click.Path(file_okay=False))
def main(subscription_id, resource_group, workspace, data_name, data_version, dest):
    if not subscription_id:
        raise click.UsageError("Missing --subscription-id / $AZURE_SUBSCRIPTION_ID.")

    credential = DefaultAzureCredential()
    ml_client = MLClient(credential, subscription_id, resource_group, workspace)

    asset = ml_client.data.get(name=data_name, version=data_version)
    match = re.search(r"/datastores/([^/]+)/paths/(.*)", asset.path or "")
    if not match:
        raise click.ClickException(
            f"Could not parse datastore from data asset path: {asset.path!r}"
        )
    datastore_name, prefix = match.groups()

    ds = ml_client.datastores.get(datastore_name)
    endpoint = getattr(ds, "endpoint", None) or "core.windows.net"
    container = ContainerClient(
        account_url=f"https://{ds.account_name}.blob.{endpoint}",
        container_name=ds.container_name,
        credential=credential,
    )

    list_prefix = f"{prefix.strip('/')}/" if prefix.strip("/") else ""
    dest_dir = Path(dest)
    dest_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    for blob in container.list_blobs(name_starts_with=list_prefix):
        if not blob.name.endswith(".csv"):
            continue
        target = dest_dir / Path(blob.name).name
        with target.open("wb") as fh:
            fh.write(container.download_blob(blob.name).readall())
        click.echo(f"fetch_splits: {blob.name} -> {target}")
        downloaded += 1

    if downloaded == 0:
        raise click.ClickException(
            f"No .csv blobs found under "
            f"{ds.container_name}/{list_prefix} in datastore '{datastore_name}'."
        )
    click.echo(f"fetch_splits: downloaded {downloaded} CSV(s) into {dest_dir}")


if __name__ == "__main__":
    main()
