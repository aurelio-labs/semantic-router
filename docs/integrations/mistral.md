Semantic Router integrates with Mistral AI's embedding models through the `MistralEncoder` class. This integration provides access to Mistral's high-quality embeddings for semantic search and classification tasks.

## Overview

The `MistralEncoder` enables semantic routing using Mistral's embedding models. It calls the
Mistral Python SDK directly — a single `mistralai.Mistral` client serving both
`embeddings.create()` and `embeddings.create_async()` — so it does not depend on LiteLLM.

## Getting Started

### Prerequisites

1. Mistral AI API key ([https://console.mistral.ai/](https://console.mistral.ai/))
2. Semantic Router version 0.1.8 or later

### Installation

```bash
pip install "semantic-router>=0.1.8"
```

### Basic Usage

```python
import os
from semantic_router.encoders import MistralEncoder

os.environ["MISTRAL_API_KEY"] = "your-api-key"

encoder = MistralEncoder(
    name="mistral-embed",
    score_threshold=0.4,
    mistralai_api_key=os.environ["MISTRAL_API_KEY"]
)
```

The key may come from the `mistralai_api_key` parameter, or from either the
`MISTRALAI_API_KEY` or `MISTRAL_API_KEY` environment variable. If none of them is set, the
constructor raises a `ValueError`.

## Features

### Supported Models

The `MistralEncoder` supports Mistral's embedding models:
- `mistral-embed` - High-quality embeddings (1024 dimensions)

### Asynchronous Support

Full async/await support for high-throughput applications:

```python
# Synchronous encoding
embeddings = encoder(["your text here"])

# Asynchronous encoding
embeddings = await encoder.acall(["your text here"])
```

Mistral embeddings are symmetric, so `encode_documents()` and `encode_queries()` issue the
same request — there is no separate query/document input type as there is for Cohere.

### Passing Mistral Options

Extra keyword arguments are forwarded to the Mistral SDK, so model-specific options are
available directly:

```python
embeddings = encoder.encode_queries(["your text here"], output_dimension=256)
```

Note that `model`, `inputs` and `encoding_format` are set by the encoder and cannot be
overridden this way.

## Integration with Routers

The `MistralEncoder` works with both `SemanticRouter` and `HybridRouter`:

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

1. **API Key Management**: Store API keys securely using environment variables

2. **Score Threshold**: Mistral embeddings typically work well with thresholds around 0.4

3. **Batch Processing**: Process multiple texts in single API calls for better performance

4. **European Option**: Mistral AI is a European company, providing GDPR-compliant AI services

## Advantages

- **High Quality**: State-of-the-art embedding quality
- **European Provider**: GDPR-compliant, European data residency options
- **Direct SDK Access**: Talks to the official Mistral SDK, with native async support and no extra abstraction layer
- **1024 Dimensions**: Balanced dimensionality for quality and efficiency

## Example Usage

```python
from semantic_router.encoders import MistralEncoder
from semantic_router.routers import SemanticRouter
from semantic_router.route import Route

encoder = MistralEncoder(name="mistral-embed", score_threshold=0.4)

routes = [
    Route(name="greeting", utterances=["hello", "hi", "hey"]),
    Route(name="goodbye", utterances=["bye", "goodbye", "see you"])
]

router = SemanticRouter(encoder=encoder, routes=routes, auto_sync="local")

print(router("hi there").name)  # -> greeting
```

## Example Notebook

For a complete example of using the Mistral integration, see the [Mistral Encoder Notebook](../encoders/mistral-encoder.ipynb).
