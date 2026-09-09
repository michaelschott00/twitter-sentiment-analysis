# External service principals (all Terraform-managed, no az CLI):
#
# - runpod-mlflow-sp: used by RunPod GPU pods to read datasets and log/register
#   to the Azure ML workspace + managed MLflow (see plan sections 7.4 and 10).
# - agent-upload-sp: used by the coding agent to upload phase-2 data (raw +
#   splits) to Azure Blob Storage via azcopy / Azure SDKs.

resource "azuread_application" "runpod" {
  display_name = "runpod-mlflow-sp"
}

resource "azuread_service_principal" "runpod" {
  client_id = azuread_application.runpod.client_id
}

resource "azuread_service_principal_password" "runpod" {
  service_principal_id = azuread_service_principal.runpod.id
}

resource "azuread_application" "agent" {
  display_name = "agent-upload-sp"
}

resource "azuread_service_principal" "agent" {
  client_id = azuread_application.agent.client_id
}

resource "azuread_service_principal_password" "agent" {
  service_principal_id = azuread_service_principal.agent.id
}

# RunPod SP -> AzureML Data Scientist on workspace (log experiments, register models).
resource "azurerm_role_assignment" "runpod_ml_ds" {
  scope                = var.workspace_id
  role_definition_name = "AzureML Data Scientist"
  principal_id         = azuread_service_principal.runpod.object_id
}

# RunPod SP -> Storage Blob Data Contributor (read splits, write model artifacts).
resource "azurerm_role_assignment" "runpod_storage" {
  scope                = var.storage_account_id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azuread_service_principal.runpod.object_id
}

# Agent SP -> Storage Blob Data Contributor (phase-2 data upload via azcopy/SDK).
resource "azurerm_role_assignment" "agent_storage" {
  scope                = var.storage_account_id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azuread_service_principal.agent.object_id
}

# Persist SP secrets in Key Vault (consumed as AZURE_CLIENT_SECRET env var).
resource "azurerm_key_vault_secret" "runpod" {
  name         = "runpod-mlflow-sp-secret"
  value        = azuread_service_principal_password.runpod.value
  key_vault_id = var.key_vault_id
}

resource "azurerm_key_vault_secret" "agent" {
  name         = "agent-upload-sp-secret"
  value        = azuread_service_principal_password.agent.value
  key_vault_id = var.key_vault_id
}
