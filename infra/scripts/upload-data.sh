#!/usr/bin/env bash
# Upload local data/ to the Terraform-managed blob containers and register
# AML data assets (plan Phase 2 — operational step, run from the dev container).
#
# Auth: azcopy uses the service-principal env vars from compose.yaml
# (AZCOPY_SPA_APPLICATION_ID / AZCOPY_SPA_CLIENT_SECRET / AZCOPY_TENANT_ID);
# `az ml` uses `az login` (see AGENTS.md).
set -euo pipefail

ACCOUNT="$(terraform -chdir=infra/terraform output -raw storage_account_name)"
WORKSPACE="$(terraform -chdir=infra/terraform output -raw workspace_name)"
RG="$(terraform -chdir=infra/terraform output -raw resource_group_name)"

azcopy copy "data/raw/*" "https://${ACCOUNT}.blob.core.windows.net/raw/" --recursive
azcopy copy "data/splits/*" "https://${ACCOUNT}.blob.core.windows.net/splits/" --recursive

az ml data create --file infra/data/twitter-raw.yaml \
  --workspace-name "$WORKSPACE" --resource-group "$RG"
az ml data create --file infra/data/twitter-splits.yaml \
  --workspace-name "$WORKSPACE" --resource-group "$RG"
