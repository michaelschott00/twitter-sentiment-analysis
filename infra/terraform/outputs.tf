output "resource_group_name" {
  description = "Name of the resource group containing all ML resources."
  value       = module.rg.name
}

output "storage_account_name" {
  description = "Name of the storage account holding datasets and models."
  value       = module.storage.account_name
}

output "workspace_name" {
  description = "Name of the Azure ML workspace."
  value       = module.ml_workspace.name
}

output "workspace_id" {
  description = "Resource ID of the Azure ML workspace."
  value       = module.ml_workspace.id
}

output "mlflow_tracking_uri" {
  description = "MLflow tracking URI for the workspace-managed MLflow (see plan section 6.2)."
  value       = module.ml_workspace.mlflow_tracking_uri
}

output "acr_login_server" {
  description = "Login server of the container registry."
  value       = module.acr.login_server
}

output "runpod_client_id" {
  description = "Application (client) ID of runpod-mlflow-sp (AZURE_CLIENT_ID on RunPod)."
  value       = module.service_principals.runpod_client_id
}

output "agent_client_id" {
  description = "Application (client) ID of agent-upload-sp (AZURE_CLIENT_ID for phase-2 upload)."
  value       = module.service_principals.agent_client_id
}

output "runpod_secret_value" {
  description = "Client secret of runpod-mlflow-sp (also in Key Vault). Sensitive."
  value       = module.service_principals.runpod_secret_value
  sensitive   = true
}

output "agent_secret_value" {
  description = "Client secret of agent-upload-sp (also in Key Vault). Sensitive."
  value       = module.service_principals.agent_secret_value
  sensitive   = true
}
