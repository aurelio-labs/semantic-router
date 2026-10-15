---
title: Encoders
description: Dense and sparse encoders turn text and images into vectors for the IndexedRouter.
---

An encoder turns text (or images, or other data) into a vector — a list of numbers that captures meaning. Those vectors are what let Semantic Router measure how similar two inputs are. An `IndexedRouter` needs one; a `DecisionRouter` doesn't use them at all.

## What an encoder does

An encoder does two jobs:

1. **At setup**, it embeds the example utterances in your routes.
2. **At runtime**, it embeds each incoming query.

The router compares the two and picks the closest route — even when the wording is completely different.

## Dense vs. sparse

Semantic Router supports two kinds of encoder. They see text very differently, and each has its strengths.

### Dense encoders

Dense encoders fill every dimension of the vector. They:

- Produce fixed-size vectors (1536 dimensions for OpenAI's `text-embedding-3-small`, for example).
- Capture rich semantic relationships.
- Do well when context and meaning matter more than exact words.

```python
import os
from semantic_router.encoders import OpenAIEncoder

os.environ["OPENAI_API_KEY"] = "your-api-key"

encoder = OpenAIEncoder()
embeddings = encoder(["How's the weather today?", "Tell me about politics"])
```

### Sparse encoders

Sparse encoders leave most dimensions at zero, with only a few non-zero values. They:

- Key off specific words and tokens.
- Excel at exact keyword matching and term frequency.
- Are more interpretable — a non-zero dimension usually maps to a specific word.

```python
import os
from semantic_router.encoders import AurelioSparseEncoder

os.environ["AURELIO_API_KEY"] = "your-api-key"

encoder = AurelioSparseEncoder()
embeddings = encoder(["How's the weather today?", "Tell me about politics"])
```

## Using both: hybrid routing

You don't have to choose. Give an `IndexedRouter` a sparse encoder as well as a dense one, and you get semantic understanding *and* keyword precision. The `alpha` parameter sets the balance: it's the weight on the dense score, and the sparse score gets the rest.

```python
from semantic_router import IndexedRouter, Route
from semantic_router.encoders import BM25Encoder, FastEmbedEncoder

routes = [
    Route(name="weather", utterances=["How's the weather?", "Is it raining?"]),
    Route(name="politics", utterances=["Tell me about politics", "Who's the president?"]),
]

router = IndexedRouter(
    encoder=FastEmbedEncoder(name="BAAI/bge-small-en-v1.5"),
    sparse_encoder=BM25Encoder(),
    routes=routes,
    alpha=0.7,  # 1.0 = dense only, 0.0 = sparse only
)
```

## Supported encoders

Most encoders ship with the base install. The ones that run models locally need an extra: `[fastembed]` for FastEmbed, and `[local]` for Hugging Face and sentence-transformers models.

```bash
pip install -qU "semantic-router[fastembed]"
pip install -qU "semantic-router[local]"
```

### Dense encoders

| Encoder | What it uses | Install |
|---------|--------------|---------|
| [OpenAIEncoder](../../client-reference/encoders/openai) | OpenAI embedding models | base |
| [AzureOpenAIEncoder](../../client-reference/encoders/azure_openai) | Azure OpenAI embedding models | base |
| [CohereEncoder](../../client-reference/encoders/cohere) | Cohere embedding models | base |
| [MistralEncoder](../../client-reference/encoders/mistral) | Mistral embedding models | base |
| [GoogleEncoder](../../client-reference/encoders/google) | Google embedding models | base |
| [BedrockEncoder](../../client-reference/encoders/bedrock) | AWS Bedrock embedding models | base |
| [JinaEncoder](../../client-reference/encoders/jina) | Jina embedding models | base |
| [VoyageEncoder](../../client-reference/encoders/voyage) | Voyage embedding models | base |
| [NimEncoder](../../client-reference/encoders/nvidia_nim) | NVIDIA NIM embedding models | base |
| [LiteLLMEncoder](../../client-reference/encoders/litellm) | Any provider supported by LiteLLM | base |
| [OllamaEncoder](../../client-reference/encoders/ollama) | Embedding models served by a local Ollama instance | base |
| [HFEndpointEncoder](../../client-reference/encoders/huggingface) | Hugging Face Inference API | base |
| [HuggingFaceEncoder](../../client-reference/encoders/huggingface) | Hugging Face models, run locally | `local` |
| [LocalEncoder](../../client-reference/encoders/local) | Any sentence-transformers model, run locally | `local` |
| [FastEmbedEncoder](../../client-reference/encoders/fastembed) | FastEmbed models, run locally on CPU | `fastembed` |
| [VitEncoder](../../client-reference/encoders/vit) | Vision Transformer, for image embeddings | `vision` |
| [CLIPEncoder](../../client-reference/encoders/clip) | CLIP, for image and text embeddings | `vision` |

### Sparse encoders

| Encoder | What it uses | Install |
|---------|--------------|---------|
| [BM25Encoder](../../client-reference/encoders/bm25) | BM25 | base |
| [TfidfEncoder](../../client-reference/encoders/tfidf) | TF-IDF | base |
| [AurelioSparseEncoder](../../client-reference/encoders/aurelio) | Aurelio's API for BM25 sparse embeddings | base |
| [LocalSparseEncoder](../../client-reference/encoders/local) | Neural sparse models (SPLADE, CSR) via sentence-transformers, run locally | `local` |

`LocalSparseEncoder` is worth a closer look if you want sparse vectors without an API. It uses the sentence-transformers `SparseEncoder` API (v5+), runs on CPU, CUDA, or MPS, and works with any compatible model from the Hugging Face Hub:

```python
from semantic_router.encoders import LocalSparseEncoder

encoder = LocalSparseEncoder(name="naver/splade-v3")
embeddings = encoder(["How's the weather today?", "Tell me about politics"])
```

## AutoEncoder

If you'd rather pick an encoder by name at runtime, `AutoEncoder` resolves the right class for you:

```python
from semantic_router.encoders import AutoEncoder
from semantic_router.schema import EncoderType

encoder = AutoEncoder(type=EncoderType.OPENAI.value, name="text-embedding-3-small").model
embeddings = encoder(["How can I help you today?"])
```

## Choosing an encoder

A few questions to ask:

1. **Accuracy.** Dense encoders understand meaning better but can miss exact keywords. Sparse encoders are the reverse.
2. **Speed.** Local encoders avoid network round-trips, though cloud models are often more accurate.
3. **Cost.** Cloud encoders (OpenAI, Cohere, Aurelio) bill per call. Local ones don't.
4. **Privacy.** Local encoders keep your data on your machine. `FastEmbedEncoder` is the lightest way to start: a small model that runs on CPU, with no API key.
5. **Both?** When you're unsure, a hybrid setup often gives the best balance.

Each encoder's reference page covers its specific options.
