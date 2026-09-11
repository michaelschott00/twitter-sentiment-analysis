data "azurerm_client_config" "current" {}

# Single dedicated CPU cluster (no GPU quota, no low-priority quota on this
# subscription). GPU training runs on RunPod (see plan section 7.4).
resource "azurerm_machine_learning_compute_cluster" "cpu" {
  name                          = "cluster-cpu"
  location                      = var.location
  vm_priority                   = var.vm_priority
  vm_size                       = var.vm_size
  machine_learning_workspace_id = var.workspace_id

  identity {
    type = "SystemAssigned"
  }

  scale_settings {
    min_node_count                       = var.min_nodes
    max_node_count                       = var.max_nodes
    scale_down_nodes_after_idle_duration = "PT5M"
  }

  tags = merge(var.tags, { purpose = "training-cpu" })
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
