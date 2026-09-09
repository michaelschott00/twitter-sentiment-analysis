output "runpod_client_id" {
  description = "Application (client) ID of runpod-mlflow-sp (use as AZURE_CLIENT_ID on RunPod)."
  value       = azuread_service_principal.runpod.client_id
}

output "agent_client_id" {
  description = "Application (client) ID of agent-upload-sp (use as AZURE_CLIENT_ID for phase-2 upload)."
  value       = azuread_service_principal.agent.client_id
}

output "runpod_secret_value" {
  description = "Client secret of runpod-mlflow-sp (also persisted in Key Vault). Exposed for initial retrieval only."
  value       = azuread_service_principal_password.runpod.value
  sensitive   = true
}

output "agent_secret_value" {
  description = "Client secret of agent-upload-sp (also persisted in Key Vault). Exposed for initial retrieval only."
  value       = azuread_service_principal_password.agent.value
  sensitive   = true
}
