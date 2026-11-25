variable "subscription_id" {
  type = string
}

variable "location" {
  type    = string
  default = "eastus2"
}

variable "image" {
  type        = string
  description = "e.g. ghcr.io/<owner>/docbase:latest (public repository: no registry credentials needed)"
}

variable "users" {
  type        = string
  sensitive   = true
  description = "name:hash pairs joined by commas (python -m docintel.web.passwd prints each one)"
}

variable "llm_model" {
  type    = string
  default = "azure_ai/gpt-5-nano-1"
}

variable "llm_base_url" {
  type        = string
  description = "e.g. https://<resource>.services.ai.azure.com/openai/v1"
}

variable "llm_api_key" {
  type      = string
  sensitive = true
}

variable "embedding_model" {
  type    = string
  default = "azure/text-embedding-3-small"
}

variable "embedding_base_url" {
  type        = string
  description = "e.g. https://<resource>.services.ai.azure.com"
}

variable "embedding_api_key" {
  type      = string
  sensitive = true
}

variable "budget_usd" {
  type    = number
  default = 10
}

variable "budget_start" {
  type        = string
  default     = "2026-12-01T00:00:00Z"
  description = "first day of a month"
}

variable "alert_email" {
  type = string
}

variable "semantic_min_similarity" {
  type        = string
  default     = "0.4"
  description = "floor of the semantic search; text-embedding-3-small scores its best hits 0.4-0.55 (0.36 gives more recall)"
}

variable "semantic_relative_margin" {
  type    = string
  default = "0.05"
}
