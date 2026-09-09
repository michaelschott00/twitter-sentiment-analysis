# Terraform — Twitter Sentiment on Azure ML

All resources live in one resource group (`rg-twitter-ml`, region `westeurope`)
for cost visibility and easy teardown. See `docs/azure-infrastructure-plan.md`.

## How to plan/apply (via Terraform HCP)

Terraform state and plan/apply are handled by **Terraform HCP**, triggered on
`git push` to `dev`; no local bootstrap or remote backend config is needed.

```bash
git push origin-http dev
# -> HCP runs `terraform plan`; a human approves and applies in HCP.
```

Local read-only checks (no credentials needed for `fmt`, needs init for `validate`):

```bash
terraform -chdir=infra/terraform fmt -recursive
terraform -chdir=infra/terraform init -backend=false
terraform -chdir=infra/terraform validate
```

## Layout

```text
infra/terraform/
├── versions.tf / variables.tf / locals.tf / main.tf / outputs.tf
├── terraform.tfvars.example
├── environments/dev.tfvars        # committed (negated in .gitignore)
└── modules/
     ├── resource_group/  monitoring/  storage/  key_vault/
     ├── container_registry/  ml_workspace/  ml_compute/
     ├── service_principals/  # runpod-mlflow-sp + agent-upload-sp + roles + KV secrets
```

## After apply

```bash
# 1. Workspace name + MLflow URI
terraform -chdir=infra/terraform output workspace_name
terraform -chdir=infra/terraform output mlflow_tracking_uri

# 2. Register the training environment (operational step, not Terraform)
az ml environment create --file infra/environments/twitter-ml-env.yaml

# 3. Upload data + register data assets, then submit jobs (see infra/jobs/).
```

Inference (online endpoint/deployment) is **not** Terraform-managed; use
`infra/endpoints/deploy.sh` (wraps `az ml online-*`).
