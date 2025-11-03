Semantic Router integrates with Aurelio AI's sparse encoding models through the `AurelioSparseEncoder` class. This integration provides high-quality sparse embeddings optimized for hybrid search and retrieval.

## Overview

The `AurelioSparseEncoder` enables sparse encoding using Aurelio's pretrained BM25 models. Sparse encoders are used alongside dense encoders in hybrid retrieval setups, combining semantic search with keyword-based matching for improved accuracy.

## Getting Started

### Prerequisites

1. Aurelio Platform API key ([https://platform.aurelio.ai/settings/api-keys](https://platform.aurelio.ai/settings/api-keys))
2. Semantic Router version 0.1.0 or later

### Installation

```bash
pip install "semantic-router>=0.1.0"
```

### Basic Usage

```python
import os
from semantic_router.encoders.aurelio import AurelioSparseEncoder

os.environ["AURELIO_API_KEY"] = "your-api-key"

sparse_encoder = AurelioSparseEncoder(name="bm25")
```

## Features

### Sparse Embeddings

Sparse encoders return dictionaries containing the indices and values of non-zero elements in the sparse matrix:

```python
result = sparse_encoder(["example text"])
# Returns: {"indices": [...], "values": [...]}
```

### Hybrid Search

Combine with dense encoders for improved retrieval accuracy:

```python
from semantic_router.encoders import OpenAIEncoder
from semantic_router.index.hybrid_local import HybridLocalIndex

# Dense encoder for semantic similarity
dense_encoder = OpenAIEncoder(name="text-embedding-3-small")

# Sparse encoder for keyword matching
sparse_encoder = AurelioSparseEncoder(name="bm25")

# Hybrid index supporting both sparse and dense
index = HybridLocalIndex()
```

## Integration with HybridRouter

The `AurelioSparseEncoder` is designed for use with `HybridRouter`:

```python
from semantic_router.routers import HybridRouter
from semantic_router.route import Route

routes = [
    Route(
        name="politics",
        utterances=[
            "isn't politics the best thing ever",
            "why don't you tell me about your political opinions"
        ]
    ),
    Route(
        name="chitchat",
        utterances=[
            "how's the weather today?",
            "lovely weather today"
        ]
    )
]

router = HybridRouter(
    encoder=dense_encoder,
    sparse_encoder=sparse_encoder,
    routes=routes,
    index=index
)
```

## Best Practices

1. **Hybrid Setup**: Always use sparse encoders with dense encoders in a `HybridRouter`

2. **Index Selection**: Use `HybridLocalIndex` for development or `PineconeIndex` (with hybrid support) for production

3. **BM25 Model**: The pretrained `bm25` model is optimized for general-purpose text retrieval

4. **API Key Management**: Store API keys securely using environment variables

5. **Performance**: Hybrid search typically provides 10-20% improvement over dense-only search

## Advantages

- **Improved Accuracy**: Hybrid search combines semantic and keyword matching
- **Complementary**: Sparse encoding catches exact matches dense encoders might miss
- **Pretrained**: Ready-to-use BM25 model, no training required
- **Flexible**: Works with any dense encoder

## Hybrid vs Dense-Only

Hybrid search (sparse + dense) provides:
- Better exact match recall
- Improved handling of rare terms and proper nouns
- More robust retrieval across diverse query types
- Typically 10-20% better accuracy than dense-only

## Example Usage

```python
from semantic_router.encoders import OpenAIEncoder
from semantic_router.encoders.aurelio import AurelioSparseEncoder
from semantic_router.routers import HybridRouter
from semantic_router.route import Route
from semantic_router.index.hybrid_local import HybridLocalIndex

# Initialize encoders
dense_encoder = OpenAIEncoder(name="text-embedding-3-small", score_threshold=0.3)
sparse_encoder = AurelioSparseEncoder(name="bm25")

# Initialize index
index = HybridLocalIndex()

# Define routes
routes = [
    Route(name="greeting", utterances=["hello", "hi", "hey"]),
    Route(name="goodbye", utterances=["bye", "goodbye", "see you"])
]

# Create hybrid router
router = HybridRouter(
    encoder=dense_encoder,
    sparse_encoder=sparse_encoder,
    routes=routes,
    index=index
)

print(router("hi there").name)  # -> greeting
```

## Example Notebook

For a complete example of using the Aurelio sparse encoder integration, see the [Aurelio BM25 Notebook](../encoders/aurelio-bm25.ipynb).
