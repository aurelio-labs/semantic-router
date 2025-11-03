Semantic Router integrates with Voyage AI's embedding models through the `VoyageEncoder` class. This integration provides access to Voyage's high-performance embeddings optimized for retrieval and semantic search tasks.

## Overview

The `VoyageEncoder` is a subclass of `LiteLLMEncoder` that enables semantic routing using Voyage AI's embedding models. It supports both synchronous and asynchronous operations with built-in cost tracking.

## Getting Started

### Prerequisites

1. Voyage AI API key ([https://www.voyageai.com/](https://www.voyageai.com/))
2. Semantic Router version 0.1.8 or later

### Installation

```bash
pip install "semantic-router>=0.1.8"
```

### Basic Usage

```python
import os
from semantic_router.encoders import VoyageEncoder

os.environ["VOYAGE_API_KEY"] = "your-api-key"

encoder = VoyageEncoder(
    name="voyage-3",
    score_threshold=0.4,
    api_key=os.environ["VOYAGE_API_KEY"]
)
```

## Features

### Supported Models

The `VoyageEncoder` supports Voyage AI's embedding models:
- `voyage-3` - Latest general-purpose model (1024 dimensions)
- `voyage-3-lite` - Lightweight version for faster inference
- `voyage-large-2` - Previous generation large model
- `voyage-code-2` - Optimized for code and technical content
- `voyage-law-2` - Specialized for legal documents

### Model Specialization

Voyage AI offers specialized models for different domains:
- **General**: `voyage-3` for most use cases
- **Code**: `voyage-code-2` for programming-related content
- **Legal**: `voyage-law-2` for legal documents and terminology
- **Lite**: `voyage-3-lite` for latency-sensitive applications

### Asynchronous Support

Full async/await support for high-throughput applications:

```python
# Synchronous encoding
embeddings = encoder(["your text here"])

# Asynchronous encoding
embeddings = await encoder.acall(["your text here"])
```

### Cost Tracking

Built-in cost tracking via LiteLLM integration:
- Automatic token counting
- Per-request cost calculation
- Model-specific pricing

## Integration with Routers

The `VoyageEncoder` works with both `SemanticRouter` and `HybridRouter`:

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

1. **Model Selection**: Use `voyage-3` for general-purpose tasks, or specialized models for domain-specific content

2. **Score Threshold**: Voyage embeddings typically work well with thresholds around 0.4

3. **API Key Management**: Store API keys securely using environment variables

4. **Batch Processing**: Process multiple texts in single API calls for better performance

5. **Domain-Specific Models**: Leverage specialized models (code, law) when working with domain-specific content

## Advantages

- **High Performance**: State-of-the-art retrieval performance on benchmarks
- **Domain Specialization**: Specialized models for code and legal content
- **Optimized Dimensions**: 1024-dimensional embeddings balance quality and efficiency
- **Cost Effective**: Competitive pricing with transparent cost tracking

## Example Usage

```python
from semantic_router.encoders import VoyageEncoder
from semantic_router.routers import SemanticRouter
from semantic_router.route import Route

encoder = VoyageEncoder(name="voyage-3", score_threshold=0.4)

routes = [
    Route(name="greeting", utterances=["hello", "hi", "hey"]),
    Route(name="goodbye", utterances=["bye", "goodbye", "see you"])
]

router = SemanticRouter(encoder=encoder, routes=routes, auto_sync="local")

print(router("hi there").name)  # -> greeting
```

## Example Notebook

For a complete example of using the Voyage integration, see the [Voyage Encoder Notebook](../encoders/voyage-encoder.ipynb).
