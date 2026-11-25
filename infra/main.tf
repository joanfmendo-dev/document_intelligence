terraform {
  required_version = ">= 1.6"
  required_providers {
    azurerm = { source = "hashicorp/azurerm", version = "~> 4.0" }
    random  = { source = "hashicorp/random", version = "~> 3.6" }
  }
}

provider "azurerm" {
  features {}
  subscription_id = var.subscription_id
}

resource "random_string" "suffix" {
  length  = 6
  upper   = false
  special = false
}

# signs the session cookie; a new value only signs everybody out
resource "random_password" "secret_key" {
  length  = 48
  special = false
}

resource "azurerm_resource_group" "rg" {
  name     = "rg-docbase"
  location = var.location
}

# The master copy of the database: upload a new .duckdb to this container and restart the app to replace the data.
resource "azurerm_storage_account" "data" {
  name                            = "docbase${random_string.suffix.result}"
  resource_group_name             = azurerm_resource_group.rg.name
  location                        = azurerm_resource_group.rg.location
  account_tier                    = "Standard"
  account_replication_type        = "LRS"
  allow_nested_items_to_be_public = false
  min_tls_version                 = "TLS1_2"
}

resource "azurerm_storage_container" "db" {
  name                  = "docbase"
  storage_account_id    = azurerm_storage_account.data.id
  container_access_type = "private"
}

resource "azurerm_log_analytics_workspace" "logs" {
  name                = "log-docbase"
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  sku                 = "PerGB2018"
  retention_in_days   = 30
  daily_quota_gb      = 0.1 # a cap, so the logs cannot become a cost
}

resource "azurerm_container_app_environment" "env" {
  name                       = "cae-docbase"
  resource_group_name        = azurerm_resource_group.rg.name
  location                   = azurerm_resource_group.rg.location
  log_analytics_workspace_id = azurerm_log_analytics_workspace.logs.id
}

locals {
  # Secrets never go in the image or in the Terraform plan output; the app reads them as environment variables.
  secrets = {
    "users"       = var.users
    "secret-key"  = random_password.secret_key.result
    "blob"        = azurerm_storage_account.data.primary_connection_string
    "llm-key"     = var.llm_api_key
    "embed-key"   = var.embedding_api_key
  }
  secret_env = {
    USERS                    = "users"
    SECRET_KEY               = "secret-key"
    BLOB_CONNECTION_STRING   = "blob"
    EXTRACTION_LLM_API_KEY   = "llm-key"
    COPILOT_API_KEY          = "llm-key"
    EMBEDDING_API_KEY        = "embed-key"
  }
  plain_env = {
    EXTRACTION_LLM_MODEL    = var.llm_model
    EXTRACTION_LLM_BASE_URL = var.llm_base_url
    COPILOT_MODEL           = var.llm_model
    COPILOT_BASE_URL        = var.llm_base_url
    EMBEDDING_MODEL         = var.embedding_model
    EMBEDDING_BASE_URL      = var.embedding_base_url
    BLOB_CONTAINER          = azurerm_storage_container.db.name
    SEMANTIC_MIN_SIMILARITY = var.semantic_min_similarity
    SEMANTIC_RELATIVE_MARGIN = var.semantic_relative_margin
    MAX_UPLOAD_MB           = "10"
    MAX_PAGES               = "10"
  }
}

resource "azurerm_container_app" "app" {
  name                         = "docbase"
  resource_group_name          = azurerm_resource_group.rg.name
  container_app_environment_id = azurerm_container_app_environment.env.id
  revision_mode                = "Single"

  dynamic "secret" {
    for_each = local.secrets
    content {
      name  = secret.key
      value = secret.value
    }
  }

  template {
    min_replicas = 0 # scale to zero: no cost while nobody uses it, a cold start for the first visit
    max_replicas = 1 # one instance only: two would overwrite each other's copy of the database

    container {
      name   = "docbase"
      image  = var.image
      cpu    = 1
      memory = "2Gi" # the OCR needs room

      dynamic "env" {
        for_each = local.plain_env
        content {
          name  = env.key
          value = env.value
        }
      }
      dynamic "env" {
        for_each = local.secret_env
        content {
          name        = env.key
          secret_name = env.value
        }
      }

      startup_probe {
        transport               = "HTTP"
        path                    = "/healthz"
        port                    = 8000
        initial_delay           = 5
        interval_seconds        = 5
        failure_count_threshold = 30
      }
    }
  }

  ingress {
    external_enabled = true
    target_port      = 8000
    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }
}

resource "azurerm_consumption_budget_resource_group" "alert" {
  name              = "docbase-budget"
  resource_group_id = azurerm_resource_group.rg.id
  amount            = var.budget_usd
  time_grain        = "Monthly"

  time_period {
    start_date = var.budget_start
  }

  notification {
    enabled        = true
    threshold      = 80
    operator       = "GreaterThan"
    threshold_type = "Actual"
    contact_emails = [var.alert_email]
  }
}
