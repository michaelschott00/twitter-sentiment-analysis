variable "workspace_id" {
  description = "Resource ID of the Azure ML workspace (scope for the RunPod SP role)."
  type        = string
}

variable "storage_account_id" {
  description = "Resource ID of the storage account holding datasets (scope for Blob Data Contributor roles)."
  type        = string
}

variable "key_vault_id" {
  description = "Resource ID of the Key Vault used to persist SP secrets."
  type        = string
}
