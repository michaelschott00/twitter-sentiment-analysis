resource "azurerm_container_registry" "this" {
  name                = "cr${var.prefix}${var.suffix}"
  resource_group_name = var.resource_group_name
  location            = var.location
  sku                 = "Basic"
  admin_enabled       = true

  tags = var.tags
}
