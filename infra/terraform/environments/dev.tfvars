# Dev environment values for Terraform HCP (learning project: cheapest viable SKUs).
env               = "dev"
location          = "westeurope"
prefix            = "twitterml"
compute_vm_size   = "Standard_DS3_v2"
compute_min_nodes = 0
compute_max_nodes = 2
vm_priority       = "Dedicated"
budget_amount     = 50
owner             = "agent"
