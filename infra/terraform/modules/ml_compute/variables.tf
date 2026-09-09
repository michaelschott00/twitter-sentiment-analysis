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

variable "storage_account_id" {
  type = string
}

variable "container_registry_id" {
  type = string
}

variable "gpu_vm_size" {
  type    = string
  default = "Standard_NC6"
}

variable "cpu_vm_size" {
  type    = string
  default = "Standard_D4s_v6"
}

variable "vm_priority" {
  type    = string
  default = "LowPriority"
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
