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
