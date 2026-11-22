# docintel

Dos pasos: **leer** documentos (texto + coordenadas) y **estructurar** ese texto con un LLM.

```python
from docintel import extract_auto, get_fields

result = extract_auto("historia.pdf")                    # leer
fields = get_fields(result["text"], json_format)         # estructurar
```

## 1. Extracción: un diccionario por documento

```python
{"text": "todo el texto del documento",
 "pages": [{"page": 1,
            "content": {"text": "todo el texto de la página",
                        "text_with_coordinates": [{"coordinates": [x0, y0, x1, y1], "text": "fragmento"}]}}]}
```
`coordinates` va normalizado entre 0 y 1 respecto a la página (origen arriba a la izquierda).
Si el método no puede ubicar el texto, `text_with_coordinates` es `[]`. Cuando no está vacía, el `text`
de cada página es exactamente la reconstrucción de sus fragmentos, y el `text` del documento es el de sus
páginas unidas por una línea en blanco. Una página sin texto aparece igualmente, con `text` vacío.
Las cuatro funciones devuelven esta misma estructura.

| Función | Para qué | Coordenadas | Formato del texto |
|---|---|---|---|
| `extract_auto` | **Recomendada.** PDF: lee la capa de texto y aplica OCR solo a las imágenes grandes sin texto encima | Sí | Mezcla |
| `extract_digital` | PDF con texto, .docx, .txt. Nunca usa OCR | Solo PDF | Markdown (títulos, tablas) |
| `extract_ocr` | PDF escaneado o imagen: OCR de la página completa (RapidOCR) | Sí | Texto plano, columnas con ` \| ` |
| `extract_vlm` | Modelo de visión; una llamada por página | No | Markdown pedido en el prompt |

Páginas: en Word siguen los saltos de página (explícitos o "salto antes del párrafo"); sin ellos, una sola. En .txt, el salto de página (form feed).

### Cómo decide `extract_auto` (PDF, página por página)
1. Lee la capa de texto. 2. Busca imágenes incrustadas **grandes** (lado corto ≥ 30 pt y largo ≥ 100 pt, en
cualquier orientación) **sin una capa de texto encima**: si hay al menos 1 carácter por cada 5.000 pt² (mínimo 10) es un
escaneo "buscable" y no se repite el OCR; con menos (un sello, un número de página) la imagen se lee igualmente.
3. Aplica OCR solo a esas regiones, convierte sus coordenadas a absolutas de la página y fusiona.
Funciona igual con páginas giradas (`/Rotate`). No usa la confianza del OCR (no es fiable).

## 2. `get_fields(text, json_format, instructions="", llm=None)`

`json_format` define la **forma**: claves, tipos (texto, entero, lista de objetos) y formato de fecha
(una cadena ISO como `"1961-01-28T00:00:00"` significa "fecha ISO"). **Sus valores nunca se envían al
modelo** (los modelos pequeños copian los ejemplos a campos que el documento no tiene).

Devuelve un diccionario con exactamente las claves y tipos de `json_format`; lo que el documento no da
queda en `None` (o `[]`). Las fechas se normalizan a ISO 8601 y la edad a entero. Si el modelo devuelve JSON
inválido, reintenta una vez. Emite un `UserWarning` si algún valor de texto **no aparece en el documento**
(posible invención); `find_ungrounded(fields, text)` da esa lista por si quieres guardarla.

`instructions` sirve para desambiguar campos. Ejemplo para tu plantilla médica, donde `facility` es el
campo DEPARTMENT del documento y no el nombre del hospital:
```python
get_fields(text, json_format, instructions=(
    "'facility' is the value of the DEPARTMENT label, not the hospital name. "
    "'age' and 'sex' come from the 'AGE / SEX' label."))
```

## 3. Base de datos local: `RecordDB` (`docintel/database/records.py`)
Guarda en un archivo **DuckDB** el diccionario que devuelve `get_fields` y lo lee de vuelta con el mismo formato.
La tabla `medical_records` tiene las mismas 24 columnas, en el mismo orden, que `medical_records.xlsx` (las tres últimas son `icd10_code`, `icd10_chapter` e `icd10_category`; un archivo `.duckdb` creado antes las recibe solas, vacías, al abrirlo): fechas
`TIMESTAMP`, `age` `INTEGER`, `vital_signs` y `lab_results` `JSON`, el resto `VARCHAR`. Solo `record_id` es obligatorio
(clave primaria); lo que el documento no da queda en `NULL`.

```python
from docintel import RecordDB, get_fields

with RecordDB("medical_records.duckdb") as db:           # por defecto DOCINTEL_DB_PATH
    db.create_record(get_fields(text, json_format))       # RecordExistsError si el record_id ya existe
    db.update_record(get_fields(text2, json_format))      # RecordNotFoundError si no existe
    db.read_record("SYNUS-2026-0571")                     # dict como el de get_fields, o None
    db.delete_record("SYNUS-2026-0571")                   # True si existía
    db.import_excel("medical_records.xlsx")               # carga inicial (necesita openpyxl)
```
- `create_record` y `update_record` devuelven el registro tal como quedó guardado.
- `update_record` solo toca las claves que recibe. Por defecto, un valor que `get_fields` da por ausente (`None`,
  `[]` o un objeto con todo en `None`) **no borra** lo guardado: un documento nuevo que no menciona un campo no
  debe borrarlo. Con `keep_existing=False` se escribe todo tal cual, `None` incluido (así se limpia un valor).
  `vital_signs` y `lab_results` se reemplazan completos, no se mezclan clave por clave.
- Las claves que no son columnas (p. ej. `facility` en vez de `department`) lanzan `ValueError`, no se descartan.
- `import_excel` acepta el Excel con o sin las tres columnas ICD-10 (y puede actualizarlas con `update_icd=True`, ver abajo); las celdas vacías o con `0` quedan en `NULL`.
- DuckDB permite que un solo proceso abra el archivo para escribir a la vez.

### Variables derivadas: `docintel/database/data_transformation.py` (clase `ICD10`)
Todo lo de ICD-10 está en una sola clase de este archivo; una variable derivada futura sería otra clase junto a ella. El modelo solo **propone el código**
a partir de `final_diagnosis` (y `chief_complaint`); la validez del código, su **capítulo** y su **categoría** salen de la ontología oficial (paquete
`simple-icd-10`, sin red), así que los tres campos nunca se contradicen y un código inventado se detecta. Si el modelo no da un código válido (se le
corrige una vez) el resultado es **vacío**, nunca un código de relleno.
```python
from docintel.data_transformation import ICD10

ICD10.resolve("I21.9")                                  # sin modelo: valida y busca capítulo y categoría
icd = ICD10()                                           # usa el modelo de settings; ICD10(llm=...) para otro
icd.classify("Community-acquired pneumonia", "Cough")   # Coding(code='J18', chapter='Diseases of the respiratory system', category='Pneumonia, organism unspecified')
icd.code_fields(get_fields(text, json_format))          # el mismo dict + icd10_code / icd10_chapter / icd10_category
```
**Actualizar un Excel** — `update_icd10(source, output, mode="missing" | "all")`; `output` es el archivo nuevo y puede ser el mismo `source`:
```python
icd.update_icd10("medical_records.xlsx", "medical_records_icd.xlsx", mode="missing")   # solo los registros con campos faltantes
icd.update_icd10("medical_records.xlsx", "medical_records.xlsx", mode="all")           # todos, sobrescribiendo el mismo archivo
```
- `mode="missing"` (por defecto): un registro está pendiente si no tiene los tres campos completos y válidos. Si ya tiene un código real, solo se
  le buscan capítulo y categoría (sin modelo); si no tiene código, o no es un código ICD-10, se le pide al modelo. Los registros completos no se tocan.
- `mode="all"`: todos los registros pasan por el modelo y se reemplaza su codificación (uno que el modelo no logra codificar conserva lo que tenía).
- Registros con el mismo diagnóstico y motivo de consulta comparten una llamada al modelo. El resto del Excel (celdas, formato) no cambia; si al
  Excel le faltan las tres columnas, se agregan al final.
- Si el modelo falla a la mitad, lo ya codificado se guarda antes de lanzar el error: repetir con `mode="missing"` continúa donde quedó.
- `dry_run=True` solo cuenta lo que se actualizaría: no escribe ni llama al modelo.

**Crear la base DuckDB desde el Excel, actualizando el ICD-10 en el camino**:
```python
with RecordDB("medical_records.duckdb") as db:
    db.import_excel("medical_records.xlsx")                                                          # igual que antes: update_icd=False
    db.import_excel("medical_records.xlsx", update_icd=True)                                         # codifica solo los faltantes
    db.import_excel("medical_records.xlsx", update_icd=True, icd_mode="all")                         # recodifica todos
    db.import_excel("medical_records.xlsx", update_icd=True, icd_output="medical_records_icd.xlsx")  # además guarda el Excel actualizado
```
Sin `icd_output` el Excel original no se modifica (el Excel actualizado solo se usa para la importación). Si la codificación falla, no se importa nada.
**Actualizar los registros de una base existente**: `icd.update_database(db, mode="missing" | "all")` (guarda cada registro apenas lo codifica; solo escribe los tres campos).

**Ejemplo: actualizar los valores ICD-10 de la base de datos** (los registros ya guardados en el `.duckdb`):
```python
from docintel import RecordDB
from docintel.data_transformation import ICD10

with RecordDB("medical_records.duckdb") as db:
    icd = ICD10()

    print(icd.update_database(db, dry_run=True))     # cuántos registros se actualizarían; no escribe ni llama al modelo
    report = icd.update_database(db, mode="missing") # codifica solo los registros sin ICD-10 completo (por defecto)
    print(report)                                    # p. ej.: 1000 records | 456 to update | 0 completed from their code | 456 coded by the model (24 model calls) | 0 left unresolved

    icd.update_database(db, mode="all")              # o recodifica TODOS los registros, reemplazando su ICD-10
```
Cada registro se guarda apenas se codifica y solo se escriben `icd10_code`, `icd10_chapter` e `icd10_category`; si el proceso se interrumpe, repetirlo con
`mode="missing"` continúa donde quedó. Un registro que el modelo no logra codificar queda como estaba (y sigue pendiente la próxima vez).

Con el Excel de prueba: 456 registros por actualizar (449 vacíos y 7 con códigos que no son ICD-10), 24 llamadas al modelo.

## 4. Modelos: un solo lugar (`docintel/shared/settings.py`, `docintel/shared/llm.py`)
Todo acceso a modelos pasa por `LLM` (sobre **litellm**) y `TokenCount`, como en tu ejemplo.
Los modelos se nombran al estilo litellm; cambiar de proveedor es cambiar una cadena:

| Variable de entorno | Por defecto |
|---|---|
| `DOCINTEL_LLM_MODEL` (get_fields) | `ollama_chat/gemma4:e2b` |
| `DOCINTEL_VLM_MODEL` (extract_vlm) | `ollama_chat/qwen3-vl:8b` |
| `DOCINTEL_LLM_API_KEY`, `DOCINTEL_LLM_BASE_URL` (y `VLM_*`) | vacío = valor por defecto del proveedor (Ollama: `localhost:11434`) |
| `DOCINTEL_OLLAMA_NUM_CTX` | `8192` (solo Ollama; Ollama corta en silencio lo que no cabe) |

Ejemplo con una API comercial: `DOCINTEL_LLM_MODEL=openai/gpt-4o-mini` y `DOCINTEL_LLM_API_KEY=...`.
`TokenCount.totals()` da los tokens de la sesión y `reset()` los pone en cero.
**Un modelo por rol:** `s.role("text" | "copilot" | "vlm" | "judge" | "embedding")` devuelve `{model, api_key, base_url}` de ese rol (`copilot` = el modelo de texto salvo que `DOCINTEL_COPILOT_LLM_MODEL` diga otro). `LLM(role="vlm")` y `Embedder()` lo usan; ningún otro archivo repite nombres de modelo ni lee las variables a mano.

## 5. Aplicativo web: DocBase (`docintel/web/`)
Prototipo local para cargar documentos, revisar los campos extraídos y mantener la base de datos. La interfaz está en inglés.

```powershell
python -m docintel.web          # http://127.0.0.1:8000   (--port 8080 para otro puerto)
```
Necesita Ollama corriendo con el modelo de `settings.py` (el mismo que usa `get_fields`).

| Pantalla | Qué hace |
|---|---|
| **Upload document** (`/upload`) | Arrastra o elige un archivo (PDF, .docx, imágenes, .txt). Corre `extract_auto` → `get_fields(record_template())` en segundo plano, con el avance en pantalla. Después codifica el diagnóstico (ICD-10) y el usuario **edita todos los campos detectados** con el **documento original al lado** (interruptor "Show original document") y guarda con `RecordDB.create_record`. El documento solo existe durante esa revisión. |
| **Database** (`/records`) | Lista paginada con botones First / Previous / Next / Last y registros por página a elegir (20, 25 —por defecto—, 50 o 100), búsqueda por paciente o ID, filtro por departamento. |
| **Edit** (`/records/<id>/edit`) | Mismo formulario, sin visor (no hay documento). Guarda con `update_record(..., keep_existing=False)`: un campo que el usuario vacía se borra. El ID no se puede cambiar. |
| **Delete** | Confirmación; borra el registro. |

- Los valores que `find_ungrounded` no encuentra en el texto salen marcados en ámbar hasta que el usuario los cambia.
- Validaciones del formulario (al cargar y al editar): el paciente es obligatorio; **Age se calcula** como los años enteros entre *Date of birth* y
  *Encounter date* (se recalcula al cambiar cualquiera de las dos fechas y el servidor repite la cuenta al guardar; con una sola fecha se puede escribir a mano);
  la fecha del encuentro no puede ser anterior al nacimiento.
- En *Lab results*, los cuadros Test y Flag sugieren una lista (los 40 análisis del Excel y Normal / High / Low / Borderline: `LAB_TESTS` y `LAB_FLAGS` en `web/forms.py`),
  pero aceptan cualquier otro texto.
- Los tres campos **ICD-10** vienen calculados. El código se puede cambiar: el capítulo y la categoría se actualizan al escribirlo y el servidor
  los vuelve a buscar al guardar; un código que no existe en ICD-10 se rechaza. Si el modelo falla en ese paso, la carga no se pierde: los campos
  quedan vacíos con un aviso. Esos campos no se marcan como "no encontrado en el documento" (se derivan, no se leen).
- Si el ID ya existe, la app pregunta antes de reemplazarlo (casilla "Replace the existing record").
- **No se guardan los documentos originales**: el archivo cargado espera en `uploads/` (`DOCINTEL_UPLOADS_DIR`) solo mientras se revisa,
  y se borra al guardar o cancelar (y la carpeta se vacía al arrancar la app). En la base solo quedan los datos.
- El visor (solo en la revisión) usa el visor de PDF del navegador; un .docx no se puede mostrar y se ofrece como descarga.
- Ajustes: `DOCINTEL_DB_PATH`, `DOCINTEL_UPLOADS_DIR`, `DOCINTEL_MAX_UPLOAD_MB` (25).
- **Sin usuarios ni contraseña**: solo escucha en `127.0.0.1` (rechaza otro `--host`), valida el `Host` y el `Origin` de los
  POST, y no deja páginas en la caché del navegador. Antes de desplegarlo en la nube hace falta autenticación.
- DuckDB deja abrir el archivo para escribir a un solo proceso: cierra notebooks u otras copias de la app que lo usen.
- Las cargas en curso viven en memoria: si reinicias la app se pierden las que no se guardaron.

## 6. Organización del código
```
docintel/
  shared/       settings.py (configuración y `s.role`), llm.py (LLM, TokenCount), embeddings.py (Embedder), ids.py (formato del record_id)
  extraction/   extracción de texto (digital, OCR, VLM, auto) y get_fields
  database/     records.py (RecordDB), record_template.json (forma de un registro para get_fields), embeddings.py (vectores), data_transformation.py (ICD-10)
  web/          DocBase (FastAPI): app.py arma la app; site.py (lo compartido: BD, modelos, plantillas, qué hacer tras guardar);
                upload_routes.py, record_routes.py y copilot_routes.py (una por área)
  copilot/      catálogo, plan, validador, ejecutor, respuesta, planner, agente (solo lectura)
  evaluation/   conjunto dorado, exactitud de ejecución, juez LLM, informe
```

## 7. Embeddings (búsqueda por significado)
Mismo modelo que `clinic_assistant`: `ollama/nomic-embed-text` (`DOCINTEL_EMBEDDING_MODEL`). **Ojo:** el `settings.py` de ese proyecto no venía en el zip;
el modelo se dedujo de su docstring. Si usaba otro, cambia la variable. El texto de cada registro es `chief_complaint` + `history_of_present_illness` + `final_diagnosis`;
los vectores viven en la tabla `record_embeddings` (con hash del texto y modelo, para saber cuándo están desactualizados).

```python
with RecordDB("medical_records.duckdb") as db:
    db.import_excel("medical_records.xlsx", embed=True)   # importa y luego crea los vectores (por defecto embed=False)
    db.embed_pending()      # crea o actualiza los que falten o estén viejos; borra los de registros que ya no existen
    db.embedding_status()   # cuántos hay, cuántos faltan
```
Si el modelo de embeddings no responde, los registros quedan importados y `EmbeddingError` dice cuántos se alcanzaron a vectorizar; vuelve a llamar `embed_pending()`.
En la app web, guardar o editar un registro crea su vector y borrarlo lo elimina; si el modelo está caído el guardado no falla (el vector queda pendiente).
Solo la app web y los cargadores escriben vectores; el copilot únicamente vectoriza la pregunta.

## 8. Copilot (solo lectura)
Una pregunta en inglés pasa por: **Planner** (una llamada al LLM que devuelve un plan JSON) → **validate** (código: campos, operadores, valores contra el catálogo y la base, fechas, plausibilidad numérica)
→ **Executor** (SELECT parametrizado armado por el código, nunca SQL del modelo) → **render** (texto por plantillas: ningún número lo escribe el modelo).
Intenciones: buscar, agregar, abrir un registro, navegar, ayuda, aclarar y fuera de alcance (escrituras, exportar y consejo clínico se rechazan).

```python
from docintel.copilot import Copilot
cp = Copilot.open("medical_records.duckdb", catalog_path="catalog.yaml")   # conexión de solo lectura
a = cp.ask("Cardiology patients over 60 this year")
print(a.interpretation, a.summary)                   # cómo entendió la pregunta, y el resultado
b = cp.ask("only women", previous=a.plan)            # seguimiento: se pasa el plan anterior
```
El **catálogo** (`catalog.yaml`: campos, roles, valores con conteos) lo genera `Catalog.build(con)` y lo mantiene la app web al guardar/editar/borrar; el copilot nunca lo escribe
(si no existe, lo calcula en memoria). La web lo guarda junto a la base (`<base>.catalog.yaml`, o `catalog_path=`). Página: `/copilot`, un **chat**: el navegador guarda la conversación y envía los últimos 4 turnos (pregunta, plan, respuesta mostrada) al planner, así una aclaración se puede contestar (`cp.ask(q, history=[{"question", "plan", "reply"}, ...])`). Sin JavaScript queda una pregunta por página, con el plan anterior. Hay enlaces a cada registro y "Show more".
**Controles del plan:** si la pregunta nombra un valor del catálogo (una prueba de laboratorio, un departamento…) que el plan no usa, el agente vuelve a preguntar al modelo una vez, mostrándole su propia respuesta y la observación; si aun así falta, la respuesta lo advierte. El validador avisa de lo que ignora (un `field` en un conteo, claves de otra intención) y de un conteo sin ninguna condición ("covers all the records"). Todo queda en `a.trace` (paso `check`).
Un `group_by` imposible (p. ej. por prueba de laboratorio) no genera una pregunta: se da la cifra total y se avisa.
**Trazabilidad:** `a.trace` es la lista ordenada de pasos (llamadas al LLM con lo enviado y la respuesta cruda, plan, validación con avisos, cada sentencia SQL con sus parámetros y filas devueltas, embedding de la consulta); `print(a.explain())` la muestra legible, y la página `/copilot` la muestra en "How this answer was obtained".
Búsqueda semántica: umbral `SEMANTIC_MIN_SIMILARITY=0.6` (sin calibrar) más un corte relativo `SEMANTIC_RELATIVE_MARGIN=0.05` (`DOCINTEL_SEMANTIC_RELATIVE_MARGIN`): solo se muestran los resultados a menos de 0.05 del mejor, porque los puntajes de nomic-embed-text están comprimidos (0.5–0.75) y el umbral solo no separa. La traza muestra cuántos pasaron cada filtro (paso `semantic_filter`).
**Correcciones tras las primeras corridas con modelos reales:** el prompt da la fecha de hoy y exige el token para "this year"/"last month" (un año literal en el ejemplo se copiaba); se aceptan `{"last_n_days": 30}` y valores copiados con su conteo (`"Cardiology (54)"`); un nombre de prueba exacto incluye sus variantes (`Glucose` → `Glucose (point-of-care)`, con aviso); un id parcial (`0007`) abre el registro `SYNUS-2026-0007`, y si encaja con varios pregunta cuál.

## 9. Evaluación
```python
from docintel.evaluation import load_golden, evaluate, Judge
golden = load_golden()                                              # 93 casos (pregunta, plan esperado, resultado esperado)
cp = Copilot.open("medical_records.duckdb", today=golden.as_of)     # el conjunto supone la fecha 2026-11-13
report = evaluate(cp, golden)                                       # solo nivel 1 (sin juez); con juez: evaluate(cp, golden, Judge())
print(report.format()); report.save("report.json")
```
Nivel 1: exactitud de ejecución (compara el resultado real con el esperado: total y ids para listas, filas para agregados; intervalo de Wilson).
Nivel 2: juez LLM comercial (`DOCINTEL_JUDGE_MODEL`, `DOCINTEL_JUDGE_API_KEY`; solo sobre datos sintéticos) con dos prompts: fidelidad de la respuesta frente al resultado ejecutado, e interpretación frente a la pregunta;
`calibrate` prueba al juez con respuestas corrompidas a propósito. El conjunto corresponde a la base de 1.000 registros del proyecto; los casos ICD-10 (I01–I03) requieren una base con las columnas `icd10_*` llenas y los semánticos (M01–M03) etiquetas humanas.

**Plan de pruebas ampliado (93 casos).** Además de los 58 originales: `RS1–RS4` (búsquedas), `RL1–RL4` (laboratorio y signos vitales), `RA1–RA4` (promedios y agrupaciones), `RG1–RG3` (ids y ambigüedad), `RF1–RF3` (seguimientos), `RC1–RC3` (valores del catálogo) y `RO1–RO4` (fuera de alcance/ayuda/navegación), y `PI01–PI10` (inyección de instrucciones). Los valores esperados se calcularon con SQL independiente del copilot. Un caso puede llevar campos opcionales:
- `previous_turns`: preguntas anteriores del mismo chat, que se hacen en vivo antes de la del caso (el modelo ve ese historial, como en la web).
- `accept`: otros planes igual de correctos (p. ej. rechazar *o* explicar qué puede hacer); el motivo (`reason`/`topic`) solo se compara si el plan lo indica.
- `checks`: comprobaciones de seguridad que también deben cumplirse: `db_unchanged` (la tabla no cambió: conteo y hash de todas las filas), `no_prompt_leak` (la respuesta no repite 8 palabras seguidas del prompt), `no_sql_in_answer`, `not_all_records` (nunca devuelve toda la tabla), `rows_capped` (no pasa del tamaño de página), `{"field_not_used": "password"}`, `{"suggests": [ids]}` (al ser ambiguo lista los candidatos) `{"text_contains": "..."}` y `{"total_if_run": N}` (si respondió con una búsqueda, encontró exactamente N registros; preguntar de vuelta también vale). Si una falla, el caso falla aunque la intención fuera correcta; el reporte muestra "Safety checks".
- `review`: el caso depende de una política (p. ej. `RC2`: "Aetna" frente a "Aetna PPO"); se ejecuta y se reporta, pero no se puntúa.
**Defensas que ya existían** (no dependen de que el modelo obedezca): base de datos abierta solo lectura y sin acceso externo, SQL parametrizado, campos y operadores de una lista cerrada, `record_id` con formato fijo, topes de página y de `top_n`, plantillas con autoescape. Esta versión **no añade ninguna defensa nueva en el copilot**: solo mide (`no_prompt_leak` comprueba si una respuesta repite 8 palabras seguidas del prompt). Pruebas deterministas con un modelo falso que "obedece" al atacante: `tests/test_robustness.py`.
Nota: las versiones v16–v18 (tablas cruzadas, rangos numéricos, límite de tokens configurable) se revirtieron; el modelo por defecto es `ollama_chat/gemma4:e2b`. Los casos nuevos aún no se han corrido con el modelo real.

## 10. Comprobaciones en tu máquina (modelos reales)
```bash
ollama pull gemma4:e2b && ollama pull nomic-embed-text      # los modelos que uses
export DOCINTEL_JUDGE_API_KEY=...                          # opcional: el juez comercial (o OPENAI_API_KEY / ANTHROPIC_API_KEY)
python -m docintel.evaluation.local_checks --db medical_records.duckdb
```
Corre en orden: 1) modelos (embeddings, texto, ¿obedece el esquema JSON?, juez), 2) embeddings (todos los registros, una corrida interrumpida que se reanuda,
una segunda corrida que no hace nada, los ganchos de guardar/borrar), 3) puntajes semánticos de varias preguntas de prueba (para elegir `SEMANTIC_MIN_SIMILARITY`),
4) el conjunto dorado con el modelo real (exactitud de ejecución, tiempos, reintentos) y 5) el juez y su calibración con respuestas corrompidas.
Trabaja sobre una **copia** (`local_checks/work.duckdb`) y en ella borra una muestra de vectores (o todos con `--rebuild-embeddings`) para que la prueba de corrida interrumpida y la corrida completa trabajen de verdad; `--in-place` crea los vectores en tu archivo (no se combina con `--rebuild-embeddings`). Las pruebas de modelo usan 400 tokens para no marcar como fallo a modelos que razonan antes de responder. Deja `local_checks/report.md` y `local_checks/eval_report.json`.
**Ruido del modelo:** Gemma no es determinista y una sola corrida cambia 2–3 casos entre ejecuciones iguales. `--runs 3` repite el conjunto dorado y lista los casos que cambian (`b 2/3` = pasó 2 de 3 veces): los inestables son ruido, los que siempre fallan son defectos. El comparador acepta como equivalente una cifra única respondida con una etiqueta al lado (`Pulmonology | 13` por `13`).
Código de color: títulos en azul, PASS en verde, FAIL en rojo, WARN y SKIP en amarillo (consola con colores ANSI; `--color always|never|auto`; en `report.md` se usan círculos 🟢🔴🟡🔵). Opciones: `--steps models embeddings ...`, `--no-judge`, `--min-accuracy 0.8`. Termina con código 1 si algún paso falla.

## Instalación
```powershell
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"     # incluye litellm (trae ~35 paquetes); `pytest` corre los tests; la app web también
ollama pull gemma4:e2b       # para get_fields
ollama pull qwen3-vl:8b     # solo si usas extract_vlm
ollama pull nomic-embed-text   # embeddings y búsqueda semántica del copilot
```

## Qué está probado y qué no
- Los tests usan un servidor HTTP falso que imita Ollama (se ejercita todo el camino hasta la petición
  HTTP de litellm), pero **ningún modelo real se ha ejecutado**: ni gemma4:e2b, ni el VLM. La calidad de
  `get_fields` con gemma4:e2b está sin medir.
- El verificador de valores inventados se probó contra el texto real de los 6 PDFs de muestra (sin falsos
  positivos), usando las respuestas correctas del Excel como si las diera el modelo.
- Las muestras son 6 PDFs sintéticos y limpios, **uno solo híbrido**. Escaneo buscable y página girada:
  probados con PDFs fabricados por los tests y un OCR simulado, no con documentos reales.
- Proveedores comerciales (OpenAI, etc.): solo probado que no reciben las opciones de Ollama; nunca se llamó a uno.

## Límites conocidos
- **Copilot y embeddings nunca se han ejecutado con Gemma3/Ollama/nomic reales ni con el juez real** (no había Ollama ni API en el entorno). Con un planner "oráculo" el ejecutor reproduce los resultados esperados del conjunto dorado; la calidad del planner real está sin medir.
- `extract_digital` ignora en silencio las imágenes con texto: para PDFs de origen desconocido usa `extract_auto`.
- Un escaneo buscable con muy poco texto (menos de 1 carácter por 5.000 pt²) se lee por OCR y ese texto puede salir duplicado.
- Texto dibujado como vectores no se detecta. Varias columnas: el orden de lectura se mezcla. Tablas partidas
  en varias imágenes se leen por tiras. Las imágenes de un DOCX no se procesan; tampoco sus tablas anidadas ni las celdas combinadas verticalmente (repiten el texto).
- Los saltos de párrafo del OCR dependen de un umbral (`GAP_FACTOR` en `_entries.py`): mueve líneas en blanco, no contenido.
- La confianza del OCR no se devuelve.
- **La calidad del código ICD-10 que asigna un modelo real no está medida** (los tests usan un modelo simulado). Ver la nota sobre el Excel de prueba más abajo.
- La app web no se ha probado con un modelo real (Ollama no estaba disponible): el flujo se probó con OCR real y un modelo simulado, y el camino de error con el pipeline real.

### Nota sobre los códigos ICD-10 del Excel de prueba
Los 544 códigos válidos del Excel coinciden con la ontología en capítulo y categoría, pero **muchos no corresponden al diagnóstico**: por ejemplo,
"Acute lower-extremity DVT" aparece con A69, I83.9 o I95.9 (la trombosis venosa profunda es I80), "Hypertensive urgency" con I70.9 (aterosclerosis)
y "Non-purulent cellulitis…" con T84 o S06.0 (complicaciones de prótesis, traumatismo craneal). 30 de los 37 diagnósticos distintos tienen más de un
código. `mode="missing"` no los corrige (son códigos válidos); revísalos, o usa `mode="all"` (`--all`) una vez que midas la calidad de tu modelo.
