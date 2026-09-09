# Dev environment values for Terraform HCP (learning project: cheapest viable SKUs).
env               = "dev"
location          = "westeurope"
prefix            = "twitterml"
compute_vm_size   = "Standard_NC4as_T4_v3"
compute_min_nodes = 0
compute_max_nodes = 2
vm_priority       = "LowPriority"
cpu_vm_size       = "Standard_D4s_v3"
budget_amount     = 50
owner             = "agent"
