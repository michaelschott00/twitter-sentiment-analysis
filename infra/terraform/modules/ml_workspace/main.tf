resource "azurerm_machine_learning_workspace" "this" {
  name                        = var.name
  location                    = var.location
  resource_group_name         = var.resource_group_name
  application_insights_id     = var.application_insights_id
  key_vault_id                = var.key_vault_id
  storage_account_id          = var.storage_account_id
  container_registry_id       = var.container_registry_id
  storage_account_access_type = "Identity"

  identity {
    type = "SystemAssigned"
  }

  public_network_access_enabled = true

  tags = var.tags
}

# One datastore per container.
resource "azurerm_machine_learning_datastore_blobstorage" "raw" {
  name                 = "ds_raw"
  workspace_id         = azurerm_machine_learning_workspace.this.id
  storage_container_id = var.raw_container_id
}

resource "azurerm_machine_learning_datastore_blobstorage" "splits" {
  name                 = "ds_splits"
  workspace_id         = azurerm_machine_learning_workspace.this.id
  storage_container_id = var.splits_container_id
}

resource "azurerm_machine_learning_datastore_blobstorage" "external" {
  name                 = "ds_external"
  workspace_id         = azurerm_machine_learning_workspace.this.id
  storage_container_id = var.external_container_id
}

resource "azurerm_machine_learning_datastore_blobstorage" "models" {
  name                 = "ds_models"
  workspace_id         = azurerm_machine_learning_workspace.this.id
  storage_container_id = var.models_container_id
}
