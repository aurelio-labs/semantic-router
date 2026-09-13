# Auditoría técnica de `semantic-router`

## Conclusión ejecutiva

Este repositorio implementa un router de intenciones basado en similitud entre la consulta y ejemplos (`utterances`) agrupados por rutas (`Route`). No clasifica con un LLM durante la decisión normal: primero calcula embeddings, consulta un índice, agrega los mejores resultados por ruta y aplica un umbral. Un LLM solo aparece después, si la ruta seleccionada tiene `function_schemas` y necesita extraer argumentos.

La respuesta a las dudas concretas es:

| Pregunta | Respuesta comprobada en código |
|---|---|
| ¿Usa `nomic`? | Solo aparece en documentación como ejemplo de `OllamaEncoder`; no es default, dependencia ni configuración activa. |
| ¿Usa `bge-m3`? | No como default ni dependencia. Puede pasarse indirectamente a un encoder compatible, pero no está configurado aquí. |
| ¿Usa `bge-small`? | Sí, como default de `LocalEncoder` y `FastEmbedEncoder`: `BAAI/bge-small-en-v1.5`. |
| ¿Existe `HybridRouter`? | Sí. Combina un encoder denso y un sparse encoder. |
| ¿Usa BM25? | Sí: `HybridRouter` crea `BM25Encoder` por defecto. |
| ¿Usa TF-IDF? | Existe `TfidfEncoder`, pero no es el default de `HybridRouter`; debe inyectarse explícitamente. |
| ¿Existe `HybridLocalIndex`? | Sí. Es el índice local predeterminado de `HybridRouter`. |
| ¿Usa crate Rust `tokenizers`? | Sí, opcionalmente para `PretrainedTokenizer`, concretamente mediante el paquete Python `tokenizers`, que contiene bindings de Hugging Face/Rust. BM25 usa por defecto `google-bert/bert-base-uncased`. |

## Cómo decide una ruta

### Flujo común

1. Se definen rutas con nombre y ejemplos de utterances.
2. Al añadirlas, se generan embeddings de documento para cada utterance.
3. Al recibir texto, se genera un embedding de consulta.
4. El índice calcula similitud contra todos los ejemplos (localmente) o delega en un backend remoto.
5. Se toman los `top_k` resultados, cuyo valor predeterminado es `5`.
6. Se agrupan los scores por nombre de ruta.
7. Se agregan con `mean` por defecto; también existen `sum` y `max`.
8. Las rutas se ordenan por score agregado y se acepta una si supera `route.score_threshold` o el umbral global.
9. Si ninguna supera el umbral, devuelve un `RouteChoice` vacío.

Para `LocalIndex`, la similitud densa es cosine similarity implementada con NumPy. El índice almacena un vector por utterance, no un único vector por ruta; por eso la agregación de los mejores resultados es importante.

### `SemanticRouter`

`SemanticRouter` solo utiliza el encoder denso y `LocalIndex` si no se proporciona otro índice. El encoder por defecto del router es `OpenAIEncoder`, cuyo modelo por defecto es `text-embedding-3-small` (configurable con `OPENAI_MODEL_NAME`). Por tanto, el default general del paquete no es BGE.

### `HybridRouter`

`HybridRouter` exige un encoder denso en su constructor. Si no se entrega índice, crea `HybridLocalIndex`; si no se entrega sparse encoder, crea `BM25Encoder`.

Para una consulta, el router produce:

- vector denso de la consulta;
- vector sparse BM25 de la consulta;
- escala convexa: denso × `alpha`, sparse × `(1 - alpha)`;
- `alpha` predeterminado: `0.3`.

El índice local normaliza el score denso por norma y calcula:

```text
score_total(utterance) = cosine(dense_query, dense_document)
                       + dot(sparse_query, sparse_document)
```

Como ambos vectores ya fueron escalados, en la práctica la contribución densa está ponderada por `alpha=0.3` y la BM25 por `0.7`. El índice selecciona los mejores utterances y el router vuelve a agruparlos por ruta y aplicar la agregación (`mean` por defecto).

Importante: `HybridLocalIndex` no admite `route_filter` y emite advertencias si se entregan `function_schemas` o metadata; esas capacidades sí están contempladas en otros índices.

## Encoders y modelos

### Densos

El paquete expone múltiples backends, entre ellos OpenAI, Azure OpenAI, Cohere, Mistral, Voyage, Jina, NVIDIA NIM, Google, Bedrock, Ollama, LiteLLM, Hugging Face, FastEmbed, sentence-transformers local, CLIP y vision transformers.

Defaults relevantes:

- `OpenAIEncoder`: `text-embedding-3-small`.
- `LocalEncoder`: `BAAI/bge-small-en-v1.5`, usando `sentence-transformers`.
- `FastEmbedEncoder`: `BAAI/bge-small-en-v1.5`, usando `fastembed`.
- `HuggingFaceEncoder`: `sentence-transformers/all-MiniLM-L6-v2`.
- `OllamaEncoder`: `hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:F16`.
- otros defaults se centralizan en `semantic_router/utils/defaults.py` y pueden cambiar mediante variables de entorno.

Los encoders asimétricos pueden tener métodos separados para queries y documents; `SemanticRouter` y `HybridRouter` los respetan mediante `AsymmetricDenseMixin`.

### Sparse

- `BM25Encoder`: default del `HybridRouter`; es ajustable (`fit`) sobre todos los utterances de las rutas.
- `TfidfEncoder`: implementación alternativa clásica; tokeniza por palabras tras minúsculas y eliminación de puntuación. No es el default híbrido.
- `AurelioSparseEncoder`: backend sparse remoto/SDK.
- `LocalSparseEncoder`: sparse encoder de sentence-transformers, default `naver/splade-v3`; tampoco es el default híbrido.

## BM25 y tokenización

`BM25Encoder` calcula una representación sparse compatible con producto punto:

- para queries produce pesos IDF normalizados;
- para documentos produce la parte TF normalizada de BM25;
- usa `k1=1.5` y `b=0.75` por defecto;
- se ajusta al corpus de utterances mediante `fit`.

Si no se entrega tokenizer, crea `PretrainedTokenizer("google-bert/bert-base-uncased")`. Ese wrapper importa de forma diferida el paquete Python `tokenizers` y llama a `Tokenizer.from_pretrained`. El paquete `tokenizers` es el binding de Hugging Face sobre la implementación Rust; no hay un crate Rust propio ni una búsqueda BM25 basada en `nomic`.

El padding usa ID `0`, y BM25 excluye ese ID del conteo. También es posible entregar un tokenizer propio o serializar su configuración.

## Dependencias principales

### Runtime base

`pydantic`, `numpy`, `aurelio-sdk`, `pyyaml`, `regex`, `tiktoken`, `aiohttp`, `tornado`, `urllib3`, `litellm`, `openai`, `colorlog` y `colorama`.

### Opcionales

- `local`: `sentence-transformers`, `torch`, `transformers`, `tokenizers`, `llama-cpp-python`.
- `fastembed`: `fastembed`.
- `qdrant`: `qdrant-client`.
- `pinecone`: `pinecone[asyncio]`.
- `postgres`: `psycopg[binary]`.
- `cohere`, `mistralai`, `google`, `bedrock`, `ollama`, `vision`.
- `dev`: pytest, pytest-asyncio, coverage, ruff, mypy y utilidades de pruebas.

El paquete soporta índices `LocalIndex`, `HybridLocalIndex`, Pinecone, Qdrant y PostgreSQL. Los backends remotos no cambian la lógica de clasificación conceptual, pero sí delegan almacenamiento y búsqueda al proveedor.

## ONNX

No existe implementación propia de ONNX ni de `onnxruntime` en `semantic_router/`, tests o documentación técnica: no hay imports, sesiones, exportación de modelos ni código de inferencia ONNX. Sin embargo, `uv.lock` contiene `onnx` y `onnxruntime` como dependencias transitivas de `fastembed` (extra opcional). Por ello, al instalar `semantic-router[fastembed]`, FastEmbed puede utilizar ONNX Runtime internamente; esto no significa que `semantic-router` exponga o controle una ruta ONNX propia.

## Archivos de referencia

- `semantic_router/routers/base.py`: decisión, thresholds, `top_k`, agregación y ejecución de rutas.
- `semantic_router/routers/semantic.py`: flujo denso.
- `semantic_router/routers/hybrid.py`: combinación densa/sparse, `alpha` y defaults híbridos.
- `semantic_router/index/local.py`: cosine similarity local.
- `semantic_router/index/hybrid_local.py`: suma de score denso y sparse.
- `semantic_router/encoders/bm25.py`: entrenamiento y codificación BM25.
- `semantic_router/encoders/tfidf.py`: alternativa TF-IDF.
- `semantic_router/tokenizers.py`: wrapper para `tokenizers` de Hugging Face.
- `semantic_router/encoders/local.py`: BGE local y SPLADE local.
- `semantic_router/encoders/fastembed.py`: BGE vía FastEmbed.
- `semantic_router/utils/defaults.py`: modelos por defecto de proveedores.
- `pyproject.toml`: dependencias y extras instalables.

## Veredicto arquitectónico

La configuración híbrida que mejor describe el código es:

```python
HybridRouter(
    encoder=<encoder denso, por ejemplo LocalEncoder o OpenAIEncoder>,
    sparse_encoder=BM25Encoder(),
    index=HybridLocalIndex(),
    alpha=0.3,
)
```

No es correcto describir el default del proyecto como “nomic/bge-m3”. La descripción verificable es: **OpenAI `text-embedding-3-small` para `SemanticRouter` por defecto; BGE-small para los encoders locales/FastEmbed; y BM25 + `HybridLocalIndex` para `HybridRouter` por defecto**.
