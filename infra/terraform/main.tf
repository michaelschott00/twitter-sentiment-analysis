resource "random_string" "suffix" {
  length  = 6
  upper   = false
  lower   = true
  numeric = true
  special = false
}

module "rg" {
  source   = "./modules/resource_group"
  name     = local.resource_group_name
  location = var.location
  tags     = local.tags
}

module "monitoring" {
  source              = "./modules/monitoring"
  resource_group_name = module.rg.name
  location            = module.rg.location
  law_name            = local.law_name
  app_insights_name   = local.app_insights_name
  tags                = local.tags
}

module "storage" {
  source              = "./modules/storage"
  resource_group_name = module.rg.name
  location            = module.rg.location
  prefix              = var.prefix
  suffix              = random_string.suffix.result
  tags                = local.tags
}

module "kv" {
  source              = "./modules/key_vault"
  resource_group_name = module.rg.name
  location            = module.rg.location
  prefix              = var.prefix
  suffix              = random_string.suffix.result
  tags                = local.tags
}

module "acr" {
  source              = "./modules/container_registry"
  resource_group_name = module.rg.name
  location            = module.rg.location
  prefix              = var.prefix
  suffix              = random_string.suffix.result
  tags                = local.tags
}

module "ml_workspace" {
  source = "./modules/ml_workspace"

  name                    = local.workspace_name
  resource_group_name     = module.rg.name
  location                = module.rg.location
  storage_account_id      = module.storage.account_id
  key_vault_id            = module.kv.id
  application_insights_id = module.monitoring.app_insights_id
  container_registry_id   = module.acr.id

  raw_container_id      = module.storage.raw_container_id
  splits_container_id   = module.storage.splits_container_id
  external_container_id = module.storage.external_container_id
  models_container_id   = module.storage.models_container_id

  region = var.location
  tags   = local.tags

  depends_on = [
    module.storage,
    module.kv,
    module.monitoring,
    module.acr,
  ]
}

module "ml_compute" {
  source = "./modules/ml_compute"

  location               = module.rg.location
  workspace_id           = module.ml_workspace.id
  workspace_principal_id = module.ml_workspace.principal_id
  container_registry_id  = module.acr.id
  storage_account_id     = module.storage.account_id

  vm_size     = var.compute_vm_size
  vm_priority = var.vm_priority
  min_nodes   = var.compute_min_nodes
  max_nodes   = var.compute_max_nodes
  tags        = local.tags
}

# External service principals: RunPod GPU training (MLflow/workspace access)
# + coding-agent data upload (Blob Storage). All Terraform-managed.
module "service_principals" {
  source = "./modules/service_principals"

  workspace_id       = module.ml_workspace.id
  storage_account_id = module.storage.account_id
  key_vault_id       = module.kv.id

  depends_on = [
    module.ml_workspace,
    module.storage,
    module.kv,
  ]
}

# Action group notifying subscription Owners (used by the budget alert below).
resource "azurerm_monitor_action_group" "budget" {
  name                = "ag-twitter-ml-budget"
  resource_group_name = module.rg.name
  short_name          = "twmlbudget"

  arm_role_receiver {
    name = "owners"
    # Built-in Owner role definition (role_id expects the bare role definition GUID).
    role_id                 = "8e3af657-a8ff-443c-a75c-2fe8c4bcb635"
    use_common_alert_schema = true
  }

  tags = local.tags
}

# Cost safety net: alert at 50% ($25) and 100% ($50 by default) of the monthly budget.
resource "azurerm_consumption_budget_resource_group" "twitter_ml" {
  name              = "budget-twitter-ml"
  resource_group_id = module.rg.id
  amount            = var.budget_amount
  time_grain        = "Monthly"

  time_period {
    start_date = formatdate("YYYY-MM-01'T'00:00:00Z", timestamp())
  }

  notification {
    enabled        = true
    threshold      = 50.0
    operator       = "GreaterThan"
    contact_groups = [azurerm_monitor_action_group.budget.id]
  }

  notification {
    enabled        = true
    threshold      = 100.0
    operator       = "GreaterThan"
    contact_groups = [azurerm_monitor_action_group.budget.id]
  }

  lifecycle {
    # Start of the current month moves forward every month; ignore it after creation.
    ignore_changes = [time_period]
  }
}

# Inference (online endpoint/deployment) is created via `az ml online-*`
# (see infra/endpoints/deploy.sh) — not Terraform-managed.
