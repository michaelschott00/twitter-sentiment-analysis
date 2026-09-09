resource "azurerm_log_analytics_workspace" "this" {
  name                = var.law_name
  location            = var.location
  resource_group_name = var.resource_group_name
  sku                 = "PerGB2018"
  retention_in_days   = 30
  daily_quota_gb      = 1
  tags                = var.tags
}

resource "azurerm_application_insights" "this" {
  name                = var.app_insights_name
  location            = var.location
  resource_group_name = var.resource_group_name
  application_type    = "other"
  workspace_id        = azurerm_log_analytics_workspace.this.id
  tags                = var.tags
}
