output "id" {
  value = azurerm_machine_learning_workspace.this.id
}

output "name" {
  value = azurerm_machine_learning_workspace.this.name
}

output "principal_id" {
  description = "Principal ID of the workspace system-assigned managed identity (used for role assignments)."
  value       = azurerm_machine_learning_workspace.this.identity[0].principal_id
}

output "mlflow_tracking_uri" {
  description = "Workspace-managed MLflow tracking URI built from region + workspace id (see plan section 6.2)."
  value = format(
    "azureml://%s.api.azureml.ms/mlflow/v1.0/%s",
    var.region,
    azurerm_machine_learning_workspace.this.id,
  )
}
