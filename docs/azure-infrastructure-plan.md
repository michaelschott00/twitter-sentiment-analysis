# Azure ML Infrastructure Plan — Twitter Sentiment Project

> **Date:** 2026-09-06

---

## Table of Contents

1. [Overview & Goals](#1-overview--goals)
2. [Target Architecture](#3-target-architecture)
3. [Azure Resources — Inventory](#4-azure-resources--inventory)
4. [Terraform Layout & Modules](#5-terraform-layout--modules)
5. [Data Storage Strategy](#6-data-storage-strategy)
6. [MLflow Integration](#7-mlflow-integration)
7. [Training Pipeline (Azure ML Jobs)](#8-training-pipeline-azure-ml-jobs)
8. [Model Registry & Artifacts](#9-model-registry--artifacts)
9. [Inference Endpoint (Non-Production)](#10-inference-endpoint-non-production)
10. [IAM, Security & Networking](#11-iam-security--networking)
11. [Cost Control](#12-cost-control)
12. [CI/CD & Local Dev Workflow](#13-cicd--local-dev-workflow)
13. [Implementation Roadmap (Phases)](#14-implementation-roadmap-phases)
14. [Risks & Open Decisions](#15-risks--open-decisions)
15. [Terraform Snippets (Illustrative)](#16-terraform-snippets-illustrative)
16. [Appendices](#17-appendices)

---

## 1. Overview & Goals

The project is originally a ML competition codebase (Twitter sentiment classification + valence regression, ~8k tweets, heavy class imbalance 6% negative). This plan outlines how to port it to Azure cloud infrastructure. No production intent — goal is learning ML platform technologies.

**Hard requirements:**

| ID | Requirement                   | Success Criteria                                                                                                                                                                     |
| -- | ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| R1 | Store dataset in Azure        | Raw + splits in versioned Azure Storage, registered as Azure ML Data Assets, reproducible `TwitterDataModule(root_dir=...)` without local `data/`                                    |
| R2 | Train & store models on Azure | `python -m twitter.main` equivalent runs as AML Command Job on Azure ML compute, checkpoints valued via `Monitor: MulticlassF1Score / MSE` persisted to cloud                            |
| R3 | Use MLflow                    | Every training run logged via MLflow (params, metrics, artifacts, model signature). integrate `lightning.pytorch.loggers.MLFlowLogger`                                               |
| R4 | Inference endpoint (explore)  | Deploy one registered model to a Managed Online Endpoint (non-prod, 0/1 instance, key auth) with `score.py` that wraps `AutoTokenizer` + `TransformerClassifier`                     |

**Constraints:**

- Terraform runs via Terraform HCP (triggered on `git push` to `dev`); no local terraform install needed (see `AGENTS.md`).
- Keep local Docker/Compose DX (`compose.yaml`, `AGENTS.md`). Don’t break `pytest`/`ruff` workflow.
- Budget: learning project → cheapest viable SKUs, auto-shutdown, spot.
- VRAM limit from later plan step: `≤24GB VRAM` per model → compute choice must respect it.
- NAT gateways are expensive → avoid them.

---

## 2. Target Architecture

### 2.1 mermaid — logical view

```mermaid
flowchart TB
    subgraph Local["Local Dev (Docker / compose.yaml)"]
        DEV["dev container\npython:3.12\nopencode / hermes"]
        CODE["twitter/ + configs/"]
        GIT["git"]
    end

    subgraph AzureRG["Azure Resource Group: rg-twitter-ml"]
        ST["Storage Account\nsttwittermlXXXX\ncontainers: raw/splits/external/models"]
        KV["Key Vault\nkv-twitter-ml-XXXX"]
        ACR["Container Registry\ncrtwittermlXXXX"]
        AI["Application Insights\n+ Log Analytics"]
        WS["Azure ML Workspace\nmlw-twitter-sentiment"]

        WS --- ST
        WS --- KV
        WS --- ACR
        WS --- AI

        subgraph AML["Azure ML Workspace Internals"]
            DS["Datastore\n(azureml blob)"]
            DA["Data Assets\ntwitter-raw@1, twitter-splits@2"]
            ENV["Environment\ntwitter-ml:py3.12-cuda\n(dockerized)"]
            COMP_CLUSTER["Compute Cluster\ncpu-or-gpu cluster\n(Standard_NC4as_T4_v3 / spot)"]
            COMP_INST["Compute Instance\nci-dev (dev/test)"]
            JOB["Command Job\ntwitter-train-clf: xlm-roberta\nmlflow logging"]
            REG["Model Registry\ntwitter-bert-clf:3\n+ MLflow registry"]
            EP["Managed Online Endpoint\ntw-sentiment\n(deployment: blue, 0-1 inst)"]
        end

        MLFLOW["MLflow Tracking\n(Workspace-managed)\nMLFLOW_TRACKING_URI=https://<ws>.ml.azure.com"]
    end

    DEV -- "git push dev -> HCP plan/apply" --> AzureRG
    CODE -- "AML SDK job submit\n+ mlflow logging" --> WS
    CODE -- "docker build -> ACR" --> ACR
    ST -- "blob mount / download" --> COMP_CLUSTER
    COMP_CLUSTER -- "runs" --> JOB
    JOB -- "logs metrics/params/artifacts" --> MLFLOW
    JOB -- "registers" --> REG
    REG -- "deploys" --> EP
    GIT -- "CI (optional)" --> ACR & JOB
```

### 2.2 mermaid — data / training flow

```mermaid
sequenceDiagram
    participant Dev as Developer (local)
    participant SA as Azure Blob (Storage Account)
    participant WS as AML Workspace
    participant CC as Compute Cluster
    participant ML as MLflow (via WS)
    participant REG as Model Registry
    participant EP as Online Endpoint

    Dev->>SA: azcopy (raw + splits)
    Dev->>WS: terraform apply provisions environment + compute
    Dev->>WS: register data assets (operational step)
    Dev->>WS: AML SDK submit training-job
    WS->>CC: schedule job (queue, spot)
    CC->>SA: mount datastore (WASBS) -> /mnt/data/splits
    CC->>ML: MLflowLogger(tracking_uri=<ws>) logs params/metrics
    CC->>REG: mlflow.register_model (MLflow model registry)
    Dev->>REG: approve / tag model (MacroF1: 0.79)
    Dev->>EP: terraform apply creates endpoint + deployment
    EP->>EP: score.py (AutoTokenizer + TransformerClassifier) health probe
    Dev->>EP: curl -H "Authorization: Bearer <key>" -d '{"text":"..."}'
```

---

## 3. Azure Resources — Inventory

All resources in **one resource group** for cost visibility + easy teardown (learning project). Region: `West Europe`. Naming: lowercase alphanumeric.

| #  | Terraform Resource Type                                                        | Name pattern                             | Purpose                                                        | SKU / Notes                                                                                                                                                                       |
| -- | ------------------------------------------------------------------------------ | ------------------------------------- | -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1  | `azurerm_resource_group`                                                       | `rg-twitter-ml`                          | Container for all resources                                    | Tags: `project=twitter-sentiment`, `env=dev`, `owner=<you>`                                                                                                                       |
| 2  | `azurerm_storage_account`                                                      | `sttwitterml`                            | Blob for datasets, checkpoints, external augmentations         | `Standard_LRS`, `kind=StorageV2`, `allow_blob_public_access=false`, `min_tls_version=TLS1_2`, versioning ON, container soft-delete 7d                                             |
| 3  | `azurerm_storage_container` (×4)                                               | `raw`, `splits`, `external`, `models`    | Logical separation mirroring local `data/*`                    | `container_access_type=private`                                                                                                                                                   |
| 4  | `azurerm_storage_management_policy`                                            | —                                        | Lifecycle: transition to Cool after 30d, delete temp after 90d | Saves cost for old logs                                                                                                                                                           |
| 5  | `azurerm_log_analytics_workspace`                                              | `log-twitter-ml`                         | Backend for App Insights                                       | `PerGB2018`, retention 30d (lowest)                                                                                                                                               |
| 6  | `azurerm_application_insights`                                                 | `appi-twitter-ml`                        | Required by AML workspace                                      | `application_type=other`, `workspace_id` → LAW                                                                                                                                    |
| 7  | `azurerm_key_vault`                                                            | `kv-twitter-ml`                          | Secrets (MLflow keys, Storage keys)                            | `sku=standard`, `purge_protection_enabled=false` for dev (true in prod)                                                                                                           |
| 8  | `azurerm_container_registry`                                                   | `crtwitterml`                            | Custom training & inference Docker images                      | `Basic` SKU (≈ $0.17/day), `admin_enabled=false` (use managed identity)                                                                                                           |
| 9  | `azurerm_machine_learning_workspace`                                           | `mlw-twitter-sentiment`                  | Core AML                                                       | `kind=default`, `storage_account_id`, `key_vault_id`, `application_insights_id`, `container_registry_id`, `public_network_access_enabled=true` (simpler for learning; VNet later) |
| 10 | `azurerm_machine_learning_compute_cluster`                                     | `cluster-gpu-spot` + `cluster-cpu`       | Training                                                       | See §4.1 — spot, 0→1 min nodes, idle 300s                                                                                                                                         |
| 11 | `azurerm_machine_learning_compute_instance` (optional)                         | `ci-dev`                                 | Interactive debugging / notebook                               | `Standard_DS3_v2`, `idle_time_before_shutdown=30` min                                                                                                                             |
| 12 | `azurerm_role_assignment` (×N)                                                 | —                                        | Least-privilege                                                | Workspace MSI → `Storage Blob Data Contributor` on SA; user → `AzureML Data Scientist` on WS                                                                                      |
| 13 | *(Future)* `azurerm_machine_learning_online_endpoint`                          | `tw-sentiment`                           | Managed Online Endpoint                                        | `auth_mode=key`, `public_network_access_enabled=true` for demo                                                                                                                    |
| 14 | `azurerm_consumption_budget_resource_group`                                   | `budget-twitter-ml`                      | Cost safety net                                                | Alert thresholds at $25 / $50; emails subscription owner                                                                                                                            |
| 15 | *(Future)* `azurerm_machine_learning_online_deployment`                        | `blue`                                   | Deployment for registered model                                | `instance_type=Standard_DS2_v2`, `instance_count=1`, `scale` min 0 for cost                                                                                                       |

### 3.1 Compute sizing (maps to ≤24GB VRAM requirement)

| Cluster                             | VM Size                                                 | vCPU / RAM / GPU            | VRAM       | When to use                                                                    | Cost hint (W. Europe, spot ~60% off)                    |
| ----------------------------------- | ------------------------------------------------------- | --------------------------- | ---------- | ------------------------------------------------------------------------------ | ------------------------------------------------------- |
| `cluster-cpu`                       | `Standard_DS3_v2` or `Standard_D4s_v3`                  | 4 / 14-16GB / —             | —          | `lightgbm` TF-IDF baseline, data validation                                    | ~$0.19/h, spot ~ $0.04                                  |
| `cluster-gpu-spot` (primary)        | `Standard_NC4as_T4_v3`                                  | 4 / 28GB / 1× T4            | 16 GB      | `sentence-bert`, `distilbert`, `twihn` (~110-250M params) fine-tune batch 8-16 | ~$0.53/h, spot ~$0.17                                   |
| `cluster-gpu-spot-large` (optional) | `Standard_NC6s_v3` (V100) or `Standard_NC24ads_A100_v4` | 6-24 / 112GB / 1× V100/A100 | 16 / 80 GB | Larger `xlm-roberta-large` if beating 0.79 F1                                  | V100 ~$3.06/h, A100 ~$3.67/h — use only for final sweep |

*Rule:* Default to T4 cluster (`enable_node_public_ip=false`, `scale_settings { min_node_count=0, max=4, scale_down_idle_minutes=5 }`). The plan explicitly **allows switching SKU via `var.compute_vm_size`** without Terraform re-create (set the variable, then apply goes through HCP).

---

## 4. Terraform Layout & Modules

### 4.1 Directory tree

```text
infra/
├── terraform/
│   ├── README.md                         # how to plan/apply (via Terraform HCP)
│   ├── versions.tf                       # required_version 1.16.0, azurerm ~>3.100, random
│   ├── variables.tf                      # env, location, prefix, vm sizes
│   ├── locals.tf                         # naming, tags, suffix
│   ├── main.tf                           # root module wiring
│   ├── outputs.tf                        # workspace id, storage name, tracking uri
│   ├── terraform.tfvars.example
│   ├── environments/
│   │   ├── dev.tfvars                    # location=westeurope, env=dev, spot=true
│   │   └── prod.tfvars                   # (future) not needed now
│   └── modules/
│       ├── resource_group/
│       │   ├── main.tf
│       │   ├── variables.tf
│       │   └── outputs.tf
│       ├── storage/
│       │   ├── main.tf                   # storage account + containers + lifecycle
│       │   ├── variables.tf
│       │   └── outputs.tf
│       ├── key_vault/
│       ├── container_registry/
│       ├── ml_workspace/
│       │   ├── main.tf                   # azurerm_machine_learning_workspace + depends_on
│       │   ├── variables.tf
│       │   └── outputs.tf
│       ├── ml_compute/
│       │   ├── main.tf                   # cluster + compute instance + role assignments
│       │   ├── variables.tf
│       │   └── outputs.tf
│       └── monitoring/
│           ├── main.tf                   # log analytics + app insights
│           └── outputs.tf
│   # Future (plan includes but NOT in initial apply):
│   # └── modules/ml_inference/  # online endpoint + deployment (count = var.enable_inference ? 1 : 0)
```

**Why modules?** Modules make `main.tf` readable and let us toggle inference via `enable_inference`.

Terraform state and plan/apply are handled by **Terraform HCP**, triggered on `git push` to `dev`; no local bootstrap or remote backend config is needed.

### 4.2 Variables (excerpt)

```hcl
variable "env"              { type = string, default = "dev" }
variable "location"         { type = string, default = "westeurope" }
variable "prefix"           { type = string, default = "twitterml" }
variable "enable_inference" { type = bool, default = false } # phase 4 only
variable "compute_vm_size"  { type = string, default = "Standard_NC4as_T4_v3" }
variable "compute_min_nodes"{ type = number, default = 0 }
variable "compute_max_nodes"{ type = number, default = 2 }
variable "vm_priority"      { type = string, default = "LowPriority" } # Dedicated vs LowPriority (spot)
```

### 4.3 Root wiring (`main.tf` sketch)

```hcl
module "rg"         { source = "./modules/resource_group" }
module "monitoring" { source = "./modules/monitoring" }
module "storage"    { source = "./modules/storage" }
module "kv"         { source = "./modules/key_vault" }
module "acr"        { source = "./modules/container_registry" }
module "ml_workspace" {
  source   = "./modules/ml_workspace"
  depends_on = [module.storage, module.kv, module.monitoring, module.acr]
}
module "ml_compute" {
  source   = "./modules/ml_compute"
  workspace_id = module.ml_workspace.id
}
# module "ml_inference" { count = var.enable_inference ? 1 : 0 ... }
```

---

## 5. Data Storage Strategy

### 5.1 What to store where

| Local Path                                                  | Azure Destination                                                     | Azure ML Abstraction                                                                         | Versioning                              |
| ----------------------------------------------------------- | --------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | --------------------------------------- |
| `data/raw/tweets_train.csv` (+ test_*)                      | `st…/raw/tweets_train.csv`                                            | `Data Asset` `twitter-raw` (`uri_folder`, `type=uri_folder`, version `1`, `2`…)              | Blob versioning ON + data asset version |
| `data/splits/tweets_{train,dev,test}*.csv` (+ `*_bert.csv`) | `st…/splits/`                                                         | `Data Asset` `twitter-splits` (`uri_folder`) — consumed by `TwitterDataModule(root_dir=...)` | New asset version on each split change  |
| `data/augmentation/{backtranslation,random_insertion}.csv`  | `st…/external/augmentation/`                                          | Part of `twitter-external` asset or separate `twitter-aug`                                   | Append-friendly                         |
| `data/external/` (empty today)                              | `st…/external/`                                                       | Reserved for future SMOTE/TF-IDF artefacts                                                   | —                                       |
| `lightning_logs/`                                           | **not** uploaded — checkpoints go to `st…/models/` or AML job outputs | Job outputs (`outputs/model.ckpt`) auto-uploaded to datastore                                | MLflow artifact store                   |

### 5.2 How `TwitterDataModule` stays compatible

`configs/data.yaml` keeps its local default so `python -m twitter.main` still runs on-device:

- **Local (unchanged default):** `configs/data.yaml` → `root_dir: "data/splits/"`
- **Cloud:** `root_dir` is overridden **at job submission time** (never by editing the committed config), via the job’s CLI argument — not by changing `configs/data.yaml`:

```yaml
# infra/jobs/job-clf-sbert.yaml
inputs:
  splits:
    type: uri_folder
    path: azureml:twitter-splits:1
```

```text
python -m twitter.main \
  --config ... \
  data.init_args.root_dir=${{inputs.splits}}
```

`${{inputs.splits}}` is an Azure ML expression that resolves to the mounted datastore path only inside the AML run; locally it would be an invalid path, so the committed default stays `data/splits/`.

### 5.3 Terraform for storage

```hcl
resource "azurerm_storage_account" "ml" {
  name                     = "st${var.prefix}${random_string.suffix.result}"
  resource_group_name      = module.rg.name
  location                 = module.rg.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  account_kind             = "StorageV2"
  min_tls_version          = "TLS1_2"
  blob_properties {
    versioning_enabled = true
    delete_retention_policy { days = 7 }
  }
  tags = local.tags
}

resource "azurerm_storage_container" "raw"      { name = "raw";      storage_account_name = azurerm_storage_account.ml.name }
resource "azurerm_storage_container" "splits"   { name = "splits";   storage_account_name = azurerm_storage_account.ml.name }
resource "azurerm_storage_container" "external" { name = "external"; storage_account_name = azurerm_storage_account.ml.name }
resource "azurerm_storage_container" "models"   { name = "models";   storage_account_name = azurerm_storage_account.ml.name }
```

### 5.4 Datastore linking

AML workspace auto-creates `workspaceblobstore` pointing at its linked storage. Additional datastores for the 4 containers are created via `azurerm_machine_learning_datastore_blob_storage` in Terraform:

```hcl
resource "azurerm_machine_learning_datastore_blob_storage" "raw" {
  name                 = "ds-raw"
  workspace_id         = module.ml_workspace.id
  storage_account_id   = module.storage.account_id
  container_name       = "raw"
  authentication_type  = "managed_identity" # workspace MSI
}

resource "azurerm_machine_learning_datastore_blob_storage" "splits" {
  name                 = "ds-splits"
  workspace_id         = module.ml_workspace.id
  storage_account_id   = module.storage.account_id
  container_name       = "splits"
  authentication_type  = "managed_identity"
}
```

Repeat for `external` and `models` containers.

---

## 6. MLflow Integration

### 6.1 Azure ML’s managed MLflow (zero self-host)

Overview:

- Azure ML managed MLflow (Workspace tracking URI)
- No infra, auto-backed by workspace SA+KV, UI in Azure ML studio, `mlflow` SDK works unchanged

Implementation:

1. **Tracking URI:** `MLFLOW_TRACKING_URI` = workspace URI from Terraform output (`azurerm_machine_learning_workspace.ml_workspace.mlflow_tracking_uri`). Use Managed Identity — no PAT.

2. **Lightning logger swap:**

   ```yaml
   # configs/defaults.yaml — before
   trainer:
     logger: null
   # after (dev + cloud)
   trainer:
     logger:
       class_path: lightning.pytorch.loggers.MLFlowLogger
       init_args:
         experiment_name: twitter-sentiment
         tracking_uri: ${MLFLOW_TRACKING_URI}
         tags: {model: sbert, task: clf}
         log_model: true   # stores MLflow pyfunc + checkpoints
   ```

   Add `MLFlowLogger` to all `configs/tasks/*.yaml` or default. Autolog can be added in `twitter/modules.py` (`mlflow.pytorch.autolog()` optional).

3. **Code change (minimal):** In `twitter/modules.py`, allow `self.logger` to be `MLFlowLogger` (already duck-typed). Remove TensorBoard-only branches — keep `add_text/figure` but gate on logger type (MLflow logs via `logger.experiment.log_text` or `mlflow.log_figure`). Simpler: log confusion matrix as `mlflow.log_figure(fig, "confusion_matrix.png")`.

4. **Auth:** Locally `az login` then `mlflow` uses `DefaultAzureCredential`. In AML job, `MLFLOW_TRACKING_URI` + managed identity auto-auth (no secret).

5. **Artifacts:** Checkpoints (`ModelCheckpoint` callback) + `lightning_logs/` → MLflow artifact store (backed by `st…/azureml` container). Register model via `mlflow.register_model()`.

### 6.2 Terraform output needed

```hcl
output "mlflow_tracking_uri" {
  value = azurerm_machine_learning_workspace.this.mlflow_tracking_uri
}
output "workspace_name" { value = azurerm_machine_learning_workspace.this.name }
```

### 6.3 Local vs cloud parity

- **Local dev:** `MLFLOW_TRACKING_URI` can point to `http://localhost:5000` (run `mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns`) or directly to Azure (`az login` required). Document both in `.env.example`.
- **CI:** Not required initially; mention for completeness.

---

## 7. Training Pipeline (Azure ML Jobs)

No need for Kubeflow/Airflow — Azure ML **Command Jobs** suffice for learning.

### 7.1 Environment

```dockerfile
# infra/environments/Dockerfile (extends base)
FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04
COPY requirements-shared.txt .
COPY requirements-cloud.txt .
RUN pip install --no-cache-dir -r requirements-shared.txt \
    && pip install --no-cache-dir -r requirements-cloud.txt
# Hugging Face cache env
ENV HF_HOME=/tmp/hf_cache
```

Register the environment via Terraform using `azurerm_machine_learning_environment`:

```hcl
resource "azurerm_machine_learning_environment" "twitter_ml" {
  name                   = "twitter-ml-env"
  workspace_id           = module.ml_workspace.id
  environment_type       = "Curated"
  image                  = "mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04"
  inference_config {
    # managed online endpoint uses this env
  }
}
```

For custom environments with `requirements.txt`, use the `az ml environment create` command as an operational step (not infrastructure-as-code) — the YAML lives in `infra/environments/twitter-ml-env.yaml`.

### 7.2 Job YAML (example: classification)

```yaml
# infra/jobs/job-clf-sbert.yaml
$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json
command: >-
  python -m twitter.main
    --config configs/tasks/classification.yaml
    --config configs/encoders/sentence_bert_base.yaml
    data.init_args.root_dir=${{inputs.splits}}
    trainer.logger.class_path=lightning.pytorch.loggers.MLFlowLogger
    trainer.logger.init_args.experiment_name=twitter-clf
    trainer.logger.init_args.tracking_uri=${{env.MLFLOW_TRACKING_URI}}
    trainer.max_epochs=10
code: .  # uploaded from local
inputs:
  splits:
    type: uri_folder
    path: azureml:twitter-splits:1
environment: azureml:twitter-ml-env:1
compute: azureml:cluster-gpu-spot
experiment_name: twitter-sentiment
display_name: sbert-clf-run-${{BUILD_ID}}
outputs:
  model_dir:
    type: uri_folder
    path: azureml://datastores/workspaceblobstore/paths/models/sbert-clf/${{name}}/
services:
  # optional: enable TensorBoard via AML
```

Submit the job via the AML SDK (operational step, not Terraform):

```python
# Or via Azure ML Python SDK v2
from azure.ai.ml import MLClient, command

ml_client = MLClient.from_config()
job = ml_client.jobs.create_or_update(
    command(
        code=".",
        command="python -m twitter.main --config ...",
        environment="azureml:twitter-ml-env:1",
        compute="azureml:cluster-gpu-spot",
        experiment_name="twitter-sentiment",
    )
)
ml_client.jobs.stream(job.name)
```

### 7.3 Mapping existing configs

| Task                | Config pair                                       | Cluster                | Metrics                                           |
| ------------------- | ------------------------------------------------- | ---------------------- | ------------------------------------------------- |
| `clf`               | `classification.yaml` + `sentence_bert_base.yaml` | `cluster-gpu-spot`     | `MulticlassF1Score` (macro), `MulticlassAccuracy` |
| `reg`               | `regression.yaml` + `sentence_bert_base.yaml`     | `cluster-gpu-spot`     | `MeanSquaredError` (RMSE)                         |
| `multitask`         | `multitask.yaml` + `sentence_bert_base.yaml`      | `cluster-gpu-spot`     | both — `loss_weight` logged to MLflow             |
| `lightgbm` baseline | `twitter/baselines/lightgbm_baseline.py` + TF-IDF | `cluster-cpu`          | Macro F1 / RMSE                                   |
| `llm` baseline      | `llm_baseline.py` (calls Foundry)                 | `cluster-cpu` or local | No training — skip AML                            |

---

## 8. Model Registry & Artifacts

Use the **workspace-backed MLflow Model Registry** for model versioning, registering via `mlflow.register_model("runs:/<run_id>/model", "twitter-bert-clf")`; the AML endpoint can deploy that registered MLflow model directly.

Artifacts to log per run:

- `configs/*.yaml` (as `mlflow.log_artifact`)
- Checkpoint `*.ckpt` (via `ModelCheckpoint` + `MLFlowLogger(log_model=True)` → `MLmodel` + `python_env.yaml`)
- Tokenizer (`AutoTokenizer.save_pretrained` → artifact)
- Metrics CSV + confusion matrix PNG + `lightning_logs/` summary
- `requirements.txt` hash for reproducibility

Naming: `twitter-{encoder}-{task}:{version}` e.g., `twitter-sbert-clf:2`, tags `macro_f1=0.79`, `rmse=0.21`, `encoder=sentence-transformers/all-MiniLM-L6-v2`.

---

## 9. Inference Endpoint (Non-Production)

**Goal:** *Try it out* — not SLA, scale-to-zero allowed.

### 9.1 Resource plan

```hcl
# modules/ml_inference/main.tf  (only when var.enable_inference=true)
resource "azurerm_machine_learning_online_endpoint" "tw" {
  name                = "tw-sentiment"
  workspace_id        = var.workspace_id
  location            = var.location
  auth_mode           = "key"   # simple for demo; AAD later
  public_network_access_enabled = true
  tags = local.tags
}

resource "azurerm_machine_learning_online_deployment" "blue" {
  name                = "blue"
  endpoint_name       = azurerm_machine_learning_online_endpoint.tw.name
  workspace_id        = var.workspace_id
  location            = var.location
  model_id            = var.model_id  # e.g. twitter-sbert-clf:2
  environment_id      = var.environment_id
  instance_type       = "Standard_DS2_v2"   # CPU cheap; GPU only if latency matters
  instance_count      = 1
  scale_settings { scale_type = "default" } # or target_utilization

  code_configuration {
    code  = "../../src/scoring"      # contains score.py + requirements.txt
    scoring_script = "score.py"
  }
  # liveness/readiness probe, request_timeout_ms=5000
}
```

Both the endpoint and deployment are fully managed via Terraform — no manual CLI steps needed.

### 9.2 Scoring script (`score.py`)

```python
# infra/endpoints/scoring/score.py
import json, torch
from transformers import AutoTokenizer, AutoModel
from twitter.models import TransformerClassifier

def init():
    global model, tokenizer
    model = TransformerClassifier.load_from_checkpoint(...)  # or mlflow.pyfunc.load_model
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model.eval()

def run(raw_data):
    data = json.loads(raw_data)
    inputs = tokenizer(data["text"], return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(inputs["input_ids"], inputs["attention_mask"])
        pred = logits.argmax(-1).item()
    return {"prediction": int(pred), "label": ["negative","neutral","positive"][pred]}
```

Include `requirements.txt` pinning `mlflow`, `transformers`, `torch`, `lightning`.

### 9.3 Non-prod guardrails

- `instance_count=0` or `scale min 0` when idle — controlled via Terraform `var.enable_inference` toggle.
- Key auth only for demo; rotate via `az ml online-endpoint get-credentials` (operational step).
- Monitor cost: endpoint idle ≈ $0.09/h for DS2_v2; tear down via `terraform destroy -target=module.ml_inference` after demo.
- Document `curl` + `mlflow deployments predict` both.

---

## 10. IAM, Security & Networking

| Area          | Dev Plan                                                                                                                             |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **Auth**      | `az login` (user), managed identity for compute/endpoint                                                                             |
| **RBAC**      | User = `Contributor` on RG + `AzureML Data Scientist` on WS; Workspace MSI = `Storage Blob Data Contributor` on SA, `AcrPull` on ACR |
| **Key Vault** | Soft-delete 7d, purge protection OFF (easy cleanup)                                                                                  |
| **Network**   | `public_network_access_enabled=true` for learning                                                                                    |
| **Secrets**   | `MLFLOW_TRACKING_URI` output, SA keys not used (identity)                                                                            |
| **Data**      | `.gitignore` keeps `data/` local; blob private                                                                                       |

**All IAM, role assignments, and access policies are managed via Terraform** — no ad-hoc `az role assignment create` or `az keyvault set-policy` commands. Role assignments live in the same module as the resource they grant access to (e.g., `modules/ml_compute/main.tf` for workspace MSI roles, `modules/storage/main.tf` for SA roles). The signed-in user's Contributor role on the RG is assumed pre-existing (created via Azure portal during subscription setup) and not part of the Terraform state.

---

## 11. Cost Control

This is a learning project — every resource is chosen to be as cheap as possible while still useful. The main cost driver is compute time; everything else is negligible in comparison.

### Compute

- Compute clusters scale to **zero nodes** (`min_node_count=0`, `scale_down_nodes_after_idle_duration=PT5M`). You pay nothing when idle.
- Use **spot VMs** (`LowPriority`) for training — 60-70% cheaper than dedicated. `ModelCheckpoint` handles preemption gracefully.
- Keep `Standard_NC4as_T4_v3` (T4) as default; only switch to V100/A100 for a final sweep if needed.
- Compute instances auto-shutdown after 30 min of idle. Stop them explicitly overnight to avoid surprises.

### Storage & Services

- `Standard_LRS` — replication overhead is unnecessary for a learning project.
- Log Analytics daily cap at 1 GB and retention at 30 days.
- Container soft-delete 7 days on blob storage — enough to recover accidental deletes without paying for long retention.

### Inference endpoint

- Gate behind `var.enable_inference` so `terraform apply` never creates it by accident.
- Scale to zero or delete after demo hours. An idle DS2_v2 endpoint adds meaningful cost for no benefit.

### Safety net

- A `Budget Alert` at $25 / $50 via `azurerm_consumption_budget_resource_group` is created using Terraform

---

## 12. CI/CD & Local Dev Workflow

**Local (day-1):**

```bash
# 0. Auth
az login
az account set --subscription <id>

# 1. Deploy infra (via Terraform HCP)
#    Push to `dev` -> HCP triggers `terraform plan` -> approve -> apply
cd infra/terraform
terraform fmt -check && terraform validate

# 2. Upload data (operational step — data lives in Terraform-managed storage)
azcopy copy data/raw/* blob.core.windows.net/<st>/raw
# Register data assets in AML
az ml data create --file infra/data/twitter-splits.yaml

# 3. Train
export MLFLOW_TRACKING_URI=<from terraform output>
az ml job create --file infra/jobs/job-clf-sbert.yaml

# 4. Register model & deploy
mlflow register-model runs:/<run_id>/model twitter-sbert-clf
# Endpoint is created by Terraform when var.enable_inference=true
```

**CI (Terraform HCP):**

- Push to `dev` triggers Terraform HCP to run `terraform plan`; a human approves and applies.
- `terraform fmt`/`validate`/`tflint` runs in a pre-commit hook.
- AML operational commands (`az ml job`, `az ml data`) run from the Docker dev container (same image as `Dockerfile`).

No change to `compose.yaml` needed; add `.env` for `MLFLOW_TRACKING_URI`, `AZURE_SUBSCRIPTION_ID`.

---

## 13. Implementation Roadmap (Phases)

### Phase 1 — Terraform scaffolding (this plan → code)

- [ ] Create `infra/terraform/{versions,variables,main,outputs}.tf` + `modules/*` per §5
- [ ] Add `azurerm_consumption_budget_resource_group` ($25 / $50 alert thresholds)
- [ ] Add `terraform.tfvars.example`, `environments/dev.tfvars`
- [ ] Add pre-commit hooks:
  - `terraform fmt`, `tflint` before every commit
  - `pytest` before every push
- [ ] `git push origin-http dev` to trigger `terraform plan`

**Exit:** `infra/terraform/README.md` explains `plan` vs `apply` + naming.

### Phase 2 — Data on Azure

- [ ] `terraform apply -target=module.storage` (or full apply — decision)
- [ ] `azcopy` to upload `data/raw` + `data/splits` to Terraform-managed containers
- [ ] Register `twitter-raw:1`, `twitter-splits:1` as AML Data Assets (operational step)
- [ ] Validate `TwitterDataModule(root_dir=<azureml mounted>)` locally

### Phase 3 — MLflow + Training on AML

- [ ] Add `MLFlowLogger` to `configs/defaults.yaml` (or task configs) + conditional `MLFLOW_TRACKING_URI`
- [ ] Update `twitter/modules.py` to support MLflow figure logging fallback
- [ ] Create `infra/environments/twitter-ml-env.yaml` + `Dockerfile`
- [ ] `terraform apply` provisions compute cluster; register AML environment (operational step)
- [ ] Submit `job-clf-sbert.yaml` on `cluster-gpu-spot` (T4 spot), check MLflow UI (`azureml://...`)
- [ ] Repeat for `reg` + `multitask` + `lightgbm` (CPU cluster)

**Success:** MLflow run shows `MulticlassF1Score` and `MeanSquaredError`.

### Phase 4 — Model Registry & Inference (non-prod)

- [ ] `mlflow.register_model` to the MLflow model registry
- [ ] Build `infra/endpoints/scoring/{score.py,requirements.txt}`
- [ ] Toggle `var.enable_inference=true` → `terraform apply` adds `online_endpoint` + `deployment`
- [ ] Smoke test endpoint with `curl` + `mlflow deployments predict`

**Success:** `curl` returns `{"prediction":1,"label":"neutral"}` for sample tweet; cost < $1 for test.

### Phase 5 — Hardening & Cleanup

- [ ] Add `azurerm_consumption_budget`, `azurerm_monitor_diagnostic_setting`
- [ ] Document teardown: `terraform destroy -var-file=environments/dev.tfvars`
- [ ] Update `README.md` with Azure Quickstart + `docs/azure-infrastructure-plan.md` link

---

## 14. Risks & Open Decisions

| Risk / Decision                                                                                          | Impact | Mitigation / Recommendation                                                                          |
| -------------------------------------------------------------------------------------------------------- | ------ | ---------------------------------------------------------------------------------------------------- |
| **MLflow version skew** — AML supports `mlflow 2.9–2.13` (check). Local `mlflow` latest may mismatch     | Low    | Pin `mlflow==2.12.0` in `requirements.txt` after checking supported versions via `terraform output` or Azure portal  |
| **T4 spot preemption** during 10-epoch S-BERT fine-tune (≈ 15 min)                                       | Low    | `ModelCheckpoint` every epoch + spot `eviction_policy=Deallocate`; retry job automatically           |
| **Data size 8k tweets ~ <10MB** but `tiktoken` + `openai` LLM baseline uses Foundry (not AML)            | Low    | Keep LLM baseline local; only `clf/reg` on AML. Mention in docs.                                     |
| **Endpoint cost surprise**                                                                               | High   | Default `enable_inference=false`; gate deployment behind variable + manual `apply`; set budget alert |
| **HF cache / rate limits** inside AML job                                                                | Medium | Set `HF_HOME=/tmp`, `TRANSFORMERS_OFFLINE=0`, cache to blob if needed                                |

---

## 15. Terraform Snippets (Illustrative)

*All snippets are plan-level — not to be applied without review. Syntax validated against `hashicorp/azurerm >=3.100`.*

### `infra/terraform/modules/ml_workspace/main.tf`

```hcl
variable "name"              { type = string }
variable "resource_group_name" { type = string }
variable "location"          { type = string }
variable "storage_account_id" { type = string }
variable "key_vault_id"      { type = string }
variable "application_insights_id" { type = string }
variable "container_registry_id" { type = string }
variable "tags"              { type = map(string) }

resource "azurerm_machine_learning_workspace" "this" {
  name                    = var.name
  location                = var.location
  resource_group_name     = var.resource_group_name
  application_insights_id = var.application_insights_id
  key_vault_id            = var.key_vault_id
  storage_account_id      = var.storage_account_id
  container_registry_id   = var.container_registry_id

  identity { type = "SystemAssigned" }
  public_network_access_enabled = true
  tags = var.tags
}
```

### `infra/terraform/modules/ml_compute/main.tf`

```hcl
variable "workspace_id" { type = string }
variable "location"     { type = string }
variable "vm_size"      { type = string, default = "Standard_NC4as_T4_v3" }
variable "vm_priority"  { type = string, default = "LowPriority" }
variable "min_nodes"    { type = number, default = 0 }
variable "max_nodes"    { type = number, default = 2 }

resource "azurerm_machine_learning_compute_cluster" "gpu" {
  name                          = "cluster-gpu-spot"
  location                      = var.location
  vm_priority                   = var.vm_priority
  vm_size                       = var.vm_size
  machine_learning_workspace_id = var.workspace_id

  scale_settings {
    min_node_count                       = var.min_nodes
    max_node_count                       = var.max_nodes
    scale_down_nodes_after_idle_duration = "PT5M"
  }
  tags = { purpose = "training" }
}
```

### `infra/terraform/outputs.tf`

```hcl
output "resource_group_name"   { value = module.rg.name }
output "storage_account_name"  { value = module.storage.account_name }
output "workspace_name"        { value = module.ml_workspace.name }
output "mlflow_tracking_uri"   { value = module.ml_workspace.mlflow_tracking_uri }
output "acr_login_server"      { value = module.acr.login_server }
```

### IAM role assignments — `modules/ml_compute/main.tf` (excerpt)

All role assignments are Terraform-managed. The workspace MSI gets its roles in the same module that creates the compute, so `depends_on` ordering is implicit.

```hcl
data "azurerm_client_config" "current" {}

# Workspace system-assigned managed identity
data "azurerm_machine_learning_workspace" "this" {
  name                = var.workspace_name
  resource_group_name = var.resource_group_name
}

# Workspace MSI → Storage Blob Data Contributor (needed to read/write datasets + model artifacts)
resource "azurerm_role_assignment" "ws_msi_storage" {
  scope                = var.storage_account_id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = data.azurerm_machine_learning_workspace.this.identity[0].principal_id
}

# Workspace MSI → AcrPull (pull training/inference images)
resource "azurerm_role_assignment" "ws_msi_acr" {
  scope                = var.container_registry_id
  role_definition_name = "AcrPull"
  principal_id         = data.azurerm_machine_learning_workspace.this.identity[0].principal_id
}

# Signed-in user → AzureML Data Scientist on workspace (required for az ml jobs)
resource "azurerm_role_assignment" "user_ml_ds" {
  scope                = var.workspace_id
  role_definition_name = "AzureML Data Scientist"
  principal_id         = data.azurerm_client_config.current.object_id
}
```

---

## 16. Appendices

### A. Glossary

| Term                        | Meaning                                                                                                  |
| --------------------------- | -------------------------------------------------------------------------------------------------------- |
| **AML**                     | Azure Machine Learning (service). Workspace is the top-level object binding storage/KV/ACR/App Insights. |
| **Data Asset**              | AML-registered dataset (`azureml:twitter-splits:1`), versioned, backs lineage                            |
| **Datastore**               | AML pointer to a Storage Account container (WASBS)                                                       |
| **Command Job**             | Batch training job with `command:` + `code:` + `inputs:` + `compute:` (provisioned via Terraform)          |
| **MLflow Tracking URI**     | URI like `azureml://...` or `https://<ws>.ml.azure.com` that `mlflow` SDK uses to log                    |
| **Managed Online Endpoint** | AML-hosted HTTPS endpoint (Kubernetes-free) with auth key, auto-scaling                                  |

### B. File references inspected for this plan

- `compose.yaml`, `Dockerfile`, `AGENTS.md`, `requirements.txt`, `.gitignore`
- `twitter/main.py`, `twitter/data.py`, `twitter/models.py`, `twitter/modules.py`
- `configs/defaults.yaml`, `configs/data.yaml`, `configs/tasks/*.yaml`, `configs/encoders/*`
- `docs/improvement-plan.md`, `data/` listings (no `*.tf` found)

### C. Explicit non-goals (this plan does NOT do)

- No production hardening (VNet, Private Link, multi-env prod, SLA endpoint).
- No migration of `lightning_logs/` history — new runs go to MLflow/AML.
- No change to `gradio` label game or `topic analysis` — out of scope.

### D. References

- <https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs/resources/machine_learning_datastore_blobstorage>
