output "law_id" {
  value = azurerm_log_analytics_workspace.this.id
}

output "app_insights_id" {
  value = azurerm_application_insights.this.id
}
