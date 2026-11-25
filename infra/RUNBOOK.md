# Runbook: primer despliegue en Azure

PowerShell, desde la raíz del proyecto salvo que se indique. Sustituye `<usuario>` (tu usuario de GitHub, en minúsculas) y `<storage>` (sale en la salida de `terraform apply`).

## 0. Antes de hacer público el repositorio
1. Escanea el historial de git en busca de claves (gitleaks, trufflehog o `git log -p | Select-String "sk-|api_key"`).
2. **Rota** en el portal de Azure cualquier clave que haya estado en el historial.
3. Comprueba que `config.json`, `*.duckdb` y `*.tfvars` no estén en el repositorio: `git ls-files | Select-String "config.json|duckdb|tfvars"` no debe devolver nada.

## 1. Usuarios
Un comando por persona (pide la contraseña y la guarda en `config.json`):
```
python -m docintel.web.passwd maria
python -m docintel.web.passwd luis
```
Cada uno imprime una línea `For USERS: nombre:hash`. Júntalas con comas para `terraform.tfvars`: `users = "maria:salt$hash,luis:salt$hash"`.

## 2. Imagen
```
git push origin main
```
Espera a que termine el workflow `image` (pestaña *Actions*). Después, en GitHub: *Packages → docbase → Package settings → Change visibility → Public*.

## 3. Terraform
```
cd infra
Copy-Item terraform.tfvars.example terraform.tfvars   # rellénalo: image = "ghcr.io/<usuario>/docbase:latest", usuarios, URL y claves de Azure OpenAI, correo
az login
az account set --subscription <subscription_id>
terraform init
terraform plan
terraform apply
```
Anota `url` y `storage_account` de la salida.

## 4. Subir tu base **antes** de abrir la app
```
az storage blob upload --account-name <storage> -c docbase -f ..\mi_base.duckdb -n docbase.duckdb --auth-mode key --overwrite
```
La app no tiene réplicas hasta la primera visita (escala a cero), así que arrancará ya con tu base. Si la abres antes de subirla, arrancará con una base vacía y el primer guardado la subirá al Blob encima de la tuya. Si eso ocurre: sube tu archivo y reinicia (sección 7).

## 5. Comprobar
- `https://<url>/healthz` responde `ok` (la primera visita tarda: arranque en frío).
- `https://<url>/` pide usuario y contraseña. Entra, abre Records y prueba una pregunta en el copiloto.
- Guarda un cambio y comprueba en el portal que `docbase.duckdb` del contenedor `docbase` cambió de fecha.

## 6. Publicar una versión nueva
Haz push a `main`; el workflow publica `ghcr.io/<usuario>/docbase:<commit>`. Luego:
```
cd infra
terraform apply -var "image=ghcr.io/<usuario>/docbase:<commit>"
```
(una etiqueta nueva fuerza la revisión nueva; con `latest` Azure no vuelve a descargar la imagen).

## 7. Reemplazar los datos o reiniciar
Sube el `.duckdb` nuevo (paso 4) y reinicia:
```
az containerapp revision restart -n docbase -g rg-docbase --revision (az containerapp revision list -n docbase -g rg-docbase --query "[0].name" -o tsv)
```

## 8. Mirar el log y copiar la base
```
az containerapp logs show -n docbase -g rg-docbase --follow
az storage blob download --account-name <storage> -c docbase -n docbase.duckdb -f copia.duckdb --auth-mode key
```
En el log aparecen los errores del modelo (`The copilot's model could not be called: ...`) y de la subida (`could not upload the database to Blob Storage`).

## 9. Borrarlo todo
```
cd infra
terraform destroy
```
Elimina el grupo de recursos `rg-docbase` entero, incluida la base del Blob. Descarga una copia antes (sección 8).
