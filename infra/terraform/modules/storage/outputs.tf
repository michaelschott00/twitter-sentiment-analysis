output "account_name" {
  value = azurerm_storage_account.ml.name
}

output "account_id" {
  value = azurerm_storage_account.ml.id
}

output "raw_container_id" {
  value = azurerm_storage_container.raw.id
}

output "splits_container_id" {
  value = azurerm_storage_container.splits.id
}

output "external_container_id" {
  value = azurerm_storage_container.external.id
}

output "models_container_id" {
  value = azurerm_storage_container.models.id
}
