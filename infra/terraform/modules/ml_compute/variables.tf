variable "location" {
  type = string
}

variable "workspace_id" {
  type = string
}

variable "workspace_principal_id" {
  description = "Principal ID of the workspace system-assigned managed identity."
  type        = string
}

variable "container_registry_id" {
  type = string
}

variable "storage_account_id" {
  description = "Resource ID of the ML storage account (scope for compute cluster blob role assignments)."
  type        = string
}

variable "vm_size" {
  description = "VM size for the dedicated CPU training cluster."
  type        = string
  default     = "Standard_DS3_v2"
}

variable "vm_priority" {
  description = "Must stay Dedicated: no low-priority quota available."
  type        = string
  default     = "Dedicated"
}

variable "min_nodes" {
  type    = number
  default = 0
}

variable "max_nodes" {
  type    = number
  default = 2
}

variable "tags" {
  type = map(string)
}
