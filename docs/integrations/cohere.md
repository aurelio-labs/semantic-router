Semantic Router integrates with Cohere's embedding models through the `CohereEncoder` class. This integration provides access to Cohere's high-quality embeddings optimized for semantic search and classification tasks.

## Overview

The `CohereEncoder` enables semantic routing using Cohere's embedding models. It calls the Cohere Python SDK directly — `cohere.ClientV2` for synchronous requests and `cohere.AsyncClientV2` for asynchronous ones — so it does not depend on LiteLLM.

## Getting Started

### Prerequisites

1. Cohere API key ([https://dashboard.cohere.ai/api-keys](https://dashboard.cohere.ai/api-keys))
2. Semantic Router version 0.1.8 or later

### Installation

```bash
pip install "semantic-router>=0.1.8"
```

### Basic Usage

```python
import os
from semantic_router.encoders import CohereEncoder

os.environ["COHERE_API_KEY"] = "your-api-key"

encoder = CohereEncoder(
    name="embed-english-v3.0",
    score_threshold=0.3
)
```

The key can also be passed explicitly with `cohere_api_key=...`. If neither the parameter
nor the `COHERE_API_KEY` environment variable is set, the constructor raises a
`ValueError`.

## Features

### Supported Models

The `CohereEncoder` supports Cohere's embedding models:
- `embed-english-v3.0` - Latest English embeddings (1024 dimensions)
- `embed-multilingual-v3.0` - Multilingual embeddings
- `embed-english-light-v3.0` - Lightweight English embeddings
- Previous versions (v2.0) also supported

### Input Types

Cohere models use different input types for optimal embedding generation. The encoder
selects the right one for you:
- `search_document` - used by `encode_documents()`, for indexing route utterances
- `search_query` - used by `encode_queries()`, `__call__()` and `acall()`, for incoming queries

### Asynchronous Support

Full async/await support for high-throughput applications:

```python
# Synchronous encoding
embeddings = encoder(["your text here"])

# Asynchronous encoding
embeddings = await encoder.acall(["your text here"])
```

### Passing Cohere Options

Extra keyword arguments are forwarded to the Cohere SDK, so model-specific options are
available directly:

```python
embeddings = encoder.encode_queries(["your text here"], output_dimension=256)
```

Note that `model`, `inputs`, `input_type` and `embedding_types` are set by the encoder
and cannot be overridden this way.

## Integration with Routers

The `CohereEncoder` works with both `SemanticRouter` and `HybridRouter`:

```python
from semantic_router.routers import SemanticRouter
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

router = SemanticRouter(
    encoder=encoder,
    routes=routes,
    auto_sync="local"
)
```

## Best Practices

1. **Model Selection**: Use `embed-english-v3.0` for best quality on English text, or `embed-multilingual-v3.0` for multilingual support

2. **Score Threshold**: Cohere embeddings often work well with lower thresholds (0.3-0.4) due to their quality

3. **API Key Management**: Store API keys securely using environment variables

4. **Batch Processing**: Process multiple texts in single API calls for better performance

5. **Input Type**: Consider using appropriate input types for asymmetric search scenarios

## Advantages

- **High Quality**: State-of-the-art embedding quality for semantic search
- **Multilingual**: Strong support for 100+ languages with multilingual models
- **Flexible**: Different model sizes for different performance/cost tradeoffs
- **Direct SDK Access**: Talks to the official Cohere SDK, with native async support and no extra abstraction layer

## Example Usage

```python
from semantic_router.encoders import CohereEncoder
from semantic_router.routers import SemanticRouter
from semantic_router.route import Route

encoder = CohereEncoder(name="embed-english-v3.0", score_threshold=0.3)

routes = [
    Route(name="greeting", utterances=["hello", "hi", "hey"]),
    Route(name="goodbye", utterances=["bye", "goodbye", "see you"])
]

router = SemanticRouter(encoder=encoder, routes=routes, auto_sync="local")

print(router("hi there").name)  # -> greeting
```

## Example Notebook

For a complete example of using the Cohere integration, see the [Cohere Encoder Notebook](../encoders/cohere.ipynb).
