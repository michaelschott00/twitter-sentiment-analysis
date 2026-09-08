# Configure the Azure provider
terraform {
  required_version = "~> 1.16.0"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 5.2.0"
    }
  }
}

provider "azurerm" {
  features {}
}

resource "azurerm_resource_group" "wd" {
  name     = "rg-twitter-ml"
  location = "germanywestcentral"
}
