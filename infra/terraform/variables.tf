variable "env" {
  description = "Environment name (dev/prod). Used in tags."
  type        = string
  default     = "dev"
}

variable "location" {
  description = "Azure region for all resources."
  type        = string
  default     = "westeurope"
}

variable "prefix" {
  description = "Lowercase alphanumeric prefix used to build globally unique names."
  type        = string
  default     = "twitterml"
}

variable "compute_vm_size" {
  description = "VM size for the dedicated CPU training cluster (no low-priority quota, no GPU quota). Changing this destroys and recreates the cluster."
  type        = string
  default     = "Standard_DS3_v2"
}

variable "compute_min_nodes" {
  description = "Minimum node count for training clusters (0 = scale to zero, pay nothing when idle)."
  type        = number
  default     = 0
}

variable "compute_max_nodes" {
  description = "Maximum node count for training clusters."
  type        = number
  default     = 2
}

variable "vm_priority" {
  description = "Must stay Dedicated: no low-priority quota available. Changing this destroys and recreates the cluster."
  type        = string
  default     = "Dedicated"
}

variable "budget_amount" {
  description = "Monthly budget amount in USD for the resource-group consumption budget."
  type        = number
  default     = 50
}

variable "owner" {
  description = "Owner tag value for all resources."
  type        = string
  default     = "agent"
}
