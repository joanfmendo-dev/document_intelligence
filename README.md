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

## 3. Modelos: un solo lugar (`docintel/settings.py`, `docintel/llm.py`)
Todo acceso a modelos pasa por `LLM` (sobre **litellm**) y `TokenCount`, como en tu ejemplo.
Los modelos se nombran al estilo litellm; cambiar de proveedor es cambiar una cadena:

| Variable de entorno | Por defecto |
|---|---|
| `DOCINTEL_LLM_MODEL` (get_fields) | `ollama_chat/gemma3:4b` |
| `DOCINTEL_VLM_MODEL` (extract_vlm) | `ollama_chat/qwen3-vl:8b` |
| `DOCINTEL_LLM_API_KEY`, `DOCINTEL_LLM_BASE_URL` (y `VLM_*`) | vacío = valor por defecto del proveedor (Ollama: `localhost:11434`) |
| `DOCINTEL_OLLAMA_NUM_CTX` | `8192` (solo Ollama; Ollama corta en silencio lo que no cabe) |

Ejemplo con una API comercial: `DOCINTEL_LLM_MODEL=openai/gpt-4o-mini` y `DOCINTEL_LLM_API_KEY=...`.
`TokenCount.totals()` da los tokens de la sesión; `TokenCount.print(in, out)` e `reset()` como en tu ejemplo.

## Instalación
```powershell
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"     # incluye litellm (trae ~35 paquetes); `pytest` corre los tests
ollama pull gemma3:4b       # para get_fields
ollama pull qwen3-vl:8b     # solo si usas extract_vlm
```

## Qué está probado y qué no
- Los tests usan un servidor HTTP falso que imita Ollama (se ejercita todo el camino hasta la petición
  HTTP de litellm), pero **ningún modelo real se ha ejecutado**: ni gemma3:4b, ni el VLM. La calidad de
  `get_fields` con gemma3:4b está sin medir.
- El verificador de valores inventados se probó contra el texto real de los 6 PDFs de muestra (sin falsos
  positivos), usando las respuestas correctas del Excel como si las diera el modelo.
- Las muestras son 6 PDFs sintéticos y limpios, **uno solo híbrido**. Escaneo buscable y página girada:
  probados con PDFs fabricados por los tests y un OCR simulado, no con documentos reales.
- Proveedores comerciales (OpenAI, etc.): solo probado que no reciben las opciones de Ollama; nunca se llamó a uno.

## Límites conocidos
- `extract_digital` ignora en silencio las imágenes con texto: para PDFs de origen desconocido usa `extract_auto`.
- Un escaneo buscable con muy poco texto (menos de 1 carácter por 5.000 pt²) se lee por OCR y ese texto puede salir duplicado.
- Texto dibujado como vectores no se detecta. Varias columnas: el orden de lectura se mezcla. Tablas partidas
  en varias imágenes se leen por tiras. Las imágenes de un DOCX no se procesan; tampoco sus tablas anidadas ni las celdas combinadas verticalmente (repiten el texto).
- Los saltos de párrafo del OCR dependen de un umbral (`GAP_FACTOR` en `_entries.py`): mueve líneas en blanco, no contenido.
- La confianza del OCR no se devuelve.
