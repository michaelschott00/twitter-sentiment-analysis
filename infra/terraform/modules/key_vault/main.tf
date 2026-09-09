data "azurerm_client_config" "current" {}

resource "azurerm_key_vault" "this" {
  name                = "kv-${var.prefix}-${var.suffix}"
  location            = var.location
  resource_group_name = var.resource_group_name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "standard"

  rbac_authorization_enabled = false

  soft_delete_retention_days = 7
  purge_protection_enabled   = false

  tags = var.tags
}
