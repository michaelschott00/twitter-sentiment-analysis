#!/usr/bin/env bash
# One-step bring-up / tear-down of the tw-sentiment demo endpoint (see plan section 9.1).
# The endpoint/deployment is intentionally NOT Terraform-managed.
set -euo pipefail

WORKSPACE="$(terraform -chdir=infra/terraform output -raw workspace_name)"
RG="$(terraform -chdir=infra/terraform output -raw resource_group_name)"

case "${1:-create}" in
  create)
    az ml online-endpoint create --file infra/endpoints/endpoint.yaml \
      --workspace-name "$WORKSPACE" --resource-group "$RG"
    az ml online-deployment create --file infra/endpoints/deployment-blue.yaml \
      --workspace-name "$WORKSPACE" --resource-group "$RG"
    az ml online-endpoint update --name tw-sentiment --traffic "blue=100" \
      --workspace-name "$WORKSPACE" --resource-group "$RG"
    ;;
  delete)
    # Tear down right after demoing — an idle DS2_v2 endpoint costs ~$0.09/h.
    az ml online-endpoint delete --name tw-sentiment --yes \
      --workspace-name "$WORKSPACE" --resource-group "$RG"
    ;;
  *)
    echo "usage: $0 [create|delete]" >&2
    exit 1
    ;;
esac
