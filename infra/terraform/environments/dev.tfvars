# Dev environment values for Terraform HCP (learning project: cheapest viable SKUs).
env               = "dev"
location          = "westeurope"
prefix            = "twitterml"
compute_vm_size   = "Standard_NC6"
compute_min_nodes = 0
compute_max_nodes = 2
vm_priority       = "LowPriority"
cpu_vm_size       = "Standard_D4s_v6"
budget_amount     = 50
owner             = "agent"
