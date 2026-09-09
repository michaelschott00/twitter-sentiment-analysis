resource "azurerm_storage_account" "ml" {
  name                     = "st${var.prefix}${var.suffix}"
  resource_group_name      = var.resource_group_name
  location                 = var.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  account_kind             = "StorageV2"
  min_tls_version          = "TLS1_2"

  allow_nested_items_to_be_public = false

  blob_properties {
    versioning_enabled = true

    delete_retention_policy {
      days = 7
    }

    container_delete_retention_policy {
      days = 7
    }
  }

  tags = var.tags
}

resource "azurerm_storage_container" "raw" {
  name               = "raw"
  storage_account_id = azurerm_storage_account.ml.id
}

resource "azurerm_storage_container" "splits" {
  name               = "splits"
  storage_account_id = azurerm_storage_account.ml.id
}

resource "azurerm_storage_container" "external" {
  name               = "external"
  storage_account_id = azurerm_storage_account.ml.id
}

resource "azurerm_storage_container" "models" {
  name               = "models"
  storage_account_id = azurerm_storage_account.ml.id
}

# Lifecycle: transition to Cool after 30d, delete temp blobs after 90d.
resource "azurerm_storage_management_policy" "ml" {
  storage_account_id = azurerm_storage_account.ml.id

  rule {
    name    = "cool-after-30d"
    enabled = true

    filters {
      blob_types = ["blockBlob"]
    }

    actions {
      base_blob {
        tier_to_cool_after_days_since_modification_greater_than = 30
      }
    }
  }

  rule {
    name    = "delete-temp-after-90d"
    enabled = true

    filters {
      prefix_match = ["models/temp/"]
      blob_types   = ["blockBlob"]
    }

    actions {
      base_blob {
        delete_after_days_since_modification_greater_than = 90
      }
    }
  }
}
