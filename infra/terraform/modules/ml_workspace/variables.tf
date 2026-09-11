variable "name" {
  type = string
}

variable "resource_group_name" {
  type = string
}

variable "location" {
  type = string
}

variable "storage_account_id" {
  type = string
}

variable "storage_account_primary_access_key" {
  description = "Primary access key for the ML storage account (used for datastore key-based auth)."
  type        = string
  sensitive   = true
}

variable "key_vault_id" {
  type = string
}

variable "application_insights_id" {
  type = string
}

variable "container_registry_id" {
  type = string
}

variable "raw_container_id" {
  type = string
}

variable "splits_container_id" {
  type = string
}

variable "external_container_id" {
  type = string
}

variable "models_container_id" {
  type = string
}

variable "region" {
  description = "Azure region slug used to build the MLflow tracking URI (e.g. westeurope)."
  type        = string
}

variable "tags" {
  type = map(string)
}
