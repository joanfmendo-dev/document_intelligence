# Despliegue en Azure (Terraform)

Crea: grupo de recursos `rg-medical-records`, Storage (`stmedicalrecords<4 caracteres>`), Container App `ca-medical-records`, entorno `cae-…`, Log Analytics `log-…`, presupuesto `budget-…`. Todos derivan de la variable `project`. Incluye: grupo de recursos, Storage (copia maestra del `.duckdb`), Log Analytics (tope 0,1 GB/día), Container Apps (escala a cero, una sola réplica) y una alerta de presupuesto. **No** crea el recurso de Azure OpenAI: usa el que ya tienes (sus URL y claves van en `terraform.tfvars`).

    cp terraform.tfvars.example terraform.tfvars   # rellénalo
    az login
    terraform init && terraform apply
    # sube tu archivo y reinicia la app:
    az storage blob upload --account-name <storage_account> -c database -f medical_records.duckdb -n medical_records.duckdb --auth-mode key

El nombre del blob debe ser `medical_records.duckdb` (el de `DB_PATH` de la imagen). El estado de Terraform contiene secretos: no lo subas al repositorio (`.gitignore` ya excluye `*.tfstate` y `*.tfvars`).

## La imagen (GitHub Actions)

`.github/workflows/image.yml`: en cada push a `main` corre las pruebas y publica `ghcr.io/<tu-usuario>/medical-records:latest` (y una etiqueta con el commit). No necesita secretos propios, usa el `GITHUB_TOKEN`.

1. Sube el repositorio y espera a que termine el workflow.
2. En GitHub, *Packages → medical-records → Package settings → Change visibility → Public* (la primera vez el paquete nace privado; Container Apps lo descarga sin credenciales solo si es público). Pon el usuario en minúsculas en `image` de `terraform.tfvars`.
3. Para publicar una versión nueva: haz push; luego reinicia la revisión (`az containerapp revision restart ...`) o cambia `image` a la etiqueta del commit y vuelve a hacer `terraform apply`.
