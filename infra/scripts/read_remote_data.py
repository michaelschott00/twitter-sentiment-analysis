"""Read tweets_dev.csv from the twitter-splits data asset (infra/data).

Auth via DefaultAzureCredential (reads $AZURE_* env vars). Resource group
and workspace are CLI args.

Bypasses azureml-fsspec (forces browser login locally, see
Azure/azure-sdk-for-python#37089): resolves the datastore via MLClient,
then reads the blob directly with azure-storage-blob.

Run: python infra/scripts/read_remote_data.py --resource-group <rg> --workspace <ws>
Requires: azure-ai-ml azure-identity azure-storage-blob pandas click
"""

import io
import os
import re

import click
import pandas as pd
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobClient


@click.command()
@click.option("--resource-group", required=True)
@click.option("--workspace", required=True)
@click.option(
    "--subscription-id", default=lambda: os.environ.get("AZURE_SUBSCRIPTION_ID")
)
@click.option("--data-name", default="twitter-splits")
@click.option("--data-version", default="1")
@click.option("--filename", default="tweets_dev.csv")
def main(resource_group, workspace, subscription_id, data_name, data_version, filename):
    if not subscription_id:
        raise click.UsageError("Missing --subscription-id / $AZURE_SUBSCRIPTION_ID.")
    credential = DefaultAzureCredential()
    ml_client = MLClient(credential, subscription_id, resource_group, workspace)

    asset = ml_client.data.get(name=data_name, version=data_version)
    datastore_name, prefix = re.search(
        r"/datastores/([^/]+)/paths/(.*)", asset.path
    ).groups()  # type: ignore[union-attr]

    ds = ml_client.datastores.get(datastore_name)
    container = ds.container_name
    endpoint = getattr(ds, "endpoint", None) or "core.windows.net"

    blob_path = f"{prefix.strip('/')}/{filename}" if prefix.strip("/") else filename
    blob = BlobClient(
        f"https://{ds.account_name}.blob.{endpoint}",
        container,
        blob_path,
        credential=credential,
    )
    df = pd.read_csv(io.BytesIO(blob.download_blob().readall()))

    click.echo(f"shape: {df.shape}")
    click.echo(df.head().to_string())


if __name__ == "__main__":
    main()
