data "azurerm_client_config" "current" {}

resource "azurerm_machine_learning_compute_cluster" "gpu" {
  name                          = "cluster-gpu-spot"
  location                      = var.location
  vm_priority                   = var.vm_priority
  vm_size                       = var.gpu_vm_size
  machine_learning_workspace_id = var.workspace_id

  scale_settings {
    min_node_count                       = var.min_nodes
    max_node_count                       = var.max_nodes
    scale_down_nodes_after_idle_duration = "PT5M"
  }

  tags = merge(var.tags, { purpose = "training-gpu" })
}

resource "azurerm_machine_learning_compute_cluster" "cpu" {
  name                          = "cluster-cpu"
  location                      = var.location
  vm_priority                   = var.vm_priority
  vm_size                       = var.cpu_vm_size
  machine_learning_workspace_id = var.workspace_id

  scale_settings {
    min_node_count                       = var.min_nodes
    max_node_count                       = var.max_nodes
    scale_down_nodes_after_idle_duration = "PT5M"
  }

  tags = merge(var.tags, { purpose = "training-cpu" })
}

# Workspace system-assigned managed identity -> Storage Blob Data Contributor
# (needed to read/write datasets + model artifacts).
resource "azurerm_role_assignment" "ws_msi_storage" {
  scope                = var.storage_account_id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = var.workspace_principal_id
}

# Workspace system-assigned managed identity -> AcrPull (pull training/inference images).
resource "azurerm_role_assignment" "ws_msi_acr" {
  scope                = var.container_registry_id
  role_definition_name = "AcrPull"
  principal_id         = var.workspace_principal_id
}

# Signed-in user -> AzureML Data Scientist on workspace (required for az ml jobs).
resource "azurerm_role_assignment" "user_ml_ds" {
  scope                = var.workspace_id
  role_definition_name = "AzureML Data Scientist"
  principal_id         = data.azurerm_client_config.current.object_id
}
