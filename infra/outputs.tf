output "url" {
  value = "https://${azurerm_container_app.app.ingress[0].fqdn}"
}

output "storage_account" {
  value       = azurerm_storage_account.data.name
  description = "upload your .duckdb to the container 'database' of this account, then restart the app"
}
