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
  description = "VM size for the GPU spot training cluster. Changing this destroys and recreates the cluster."
  type        = string
  default     = "Standard_NC4as_T4_v3"
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
  description = "Dedicated vs LowPriority (spot). Spot is 60-70% cheaper; changing this destroys and recreates the cluster."
  type        = string
  default     = "LowPriority"
}

variable "cpu_vm_size" {
  description = "VM size for the CPU training cluster (lightgbm baseline, data validation)."
  type        = string
  default     = "Standard_D4s_v3"
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
