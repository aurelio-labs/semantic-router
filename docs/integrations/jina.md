Semantic Router integrates with Jina AI's embedding models through the `JinaEncoder` class. This integration allows you to use Jina's high-quality embeddings for semantic routing tasks.

## Overview

The `JinaEncoder` provides access to Jina's embedding models. Jina exposes an
OpenAI-compatible embeddings endpoint, so the encoder drives it with the OpenAI SDK
pointed at `https://api.jina.ai/v1` — it does not depend on LiteLLM. It supports both
synchronous and asynchronous encoding operations, making it suitable for various use
cases.

## Getting Started

### Prerequisites

1. A Jina AI API key (available at [https://jina.ai/api-keys/](https://jina.ai/api-keys/))
2. Semantic Router version 0.1.8 or later

### Installation

```bash
pip install semantic-router>=0.1.8
```

### Basic Usage

```python
import os
from semantic_router.encoders import JinaEncoder

# Set your Jina API key
os.environ["JINA_API_KEY"] = "your-api-key"

# Initialize the encoder
encoder = JinaEncoder(
    name="jina-embeddings-v3",  # or any other supported Jina model
    score_threshold=0.4,  # optional: set your desired similarity threshold
    api_key=os.environ["JINA_API_KEY"]
)
```

The key may come from the `api_key` parameter, or from either the `JINA_API_KEY` or
`JINA_AI_API_KEY` environment variable — the latter is still read for compatibility with
earlier releases. If none of them is set, the constructor raises a `ValueError`.

### Passing Jina Options

Standard OpenAI embedding parameters such as `dimensions` are forwarded as-is. Options
specific to Jina — `task`, `late_chunking` — travel in `extra_body`:

```python
embeddings = encoder.encode_queries(
    ["your text here"],
    dimensions=256,
    extra_body={"task": "retrieval.query"},
)
```

Note that `input` and `model` are set by the encoder and cannot be overridden this way.

### Custom Endpoint

Point the encoder at a different host with the `base_url` parameter, or the
`JINA_BASE_URL` / `JINA_AI_API_BASE` environment variable — the latter is still read for
compatibility with earlier releases.

## Features

### Model Selection

The `JinaEncoder` supports various Jina embedding models. By default, it uses the model specified in `EncoderDefault.JINA.value['embedding_model']`. You can specify a different model by providing the `name` parameter.

### Score Threshold

You can configure the similarity score threshold for routing decisions. This determines how similar two pieces of text need to be to be considered a match.

### Asynchronous Support

The encoder supports both synchronous and asynchronous operations:

```python
# Synchronous encoding
embeddings = encoder(["your text here"])

# Asynchronous encoding
embeddings = await encoder.acall(["your text here"])
```

## Integration with Routers

The `JinaEncoder` can be used with both `SemanticRouter` and `HybridRouter`:

```python
from semantic_router.routers import SemanticRouter

# Create routes
routes = [
    Route(
        name="support",
        utterances=["I need help", "Can you assist me?"]
    ),
    Route(
        name="sales",
        utterances=["I want to buy", "How much does it cost?"]
    )
]

# Initialize router with Jina encoder
router = SemanticRouter(
    encoder=encoder,
    routes=routes,
    auto_sync="local"
)
```

## Best Practices

1. **API Key Management**: Store your Jina API key securely using environment variables
2. **Model Selection**: Choose the appropriate Jina model for your use case
3. **Score Threshold**: Adjust the score threshold based on your routing requirements
4. **Batch Processing**: For multiple texts, use the list input format for better performance

## Example Notebook

For a complete example of using the Jina integration, see the [Jina Encoder Notebook](../encoders/jina-encoder.ipynb).
