Semantic Router integrates with Ollama for local LLM and embedding model hosting through the `OllamaEncoder` class. This integration enables you to use Ollama-hosted models for semantic routing tasks while maintaining full control over your infrastructure.

## Overview

The `OllamaEncoder` allows you to use embedding models hosted on Ollama for semantic routing. All computation happens on your local Ollama instance, providing privacy, low latency, and full control over model selection.

## Getting Started

### Prerequisites

1. Ollama installed and running ([https://ollama.ai/](https://ollama.ai/))
2. An embedding model pulled in Ollama (e.g., `ollama pull nomic-embed-text`)
3. Semantic Router version 0.1.11 or later

### Installation

```bash
pip install semantic-router
```

### Basic Usage

```python
from semantic_router.encoders import OllamaEncoder

encoder = OllamaEncoder(
    name="nomic-embed-text",  # or any Ollama embedding model
    base_url="http://localhost:11434"  # default Ollama URL
)
```

## Features

### Supported Models

The `OllamaEncoder` supports any embedding model available in Ollama, including:
- `nomic-embed-text` - High-quality text embeddings
- `mxbai-embed-large` - Large embedding model
- `all-minilm` - Lightweight and fast
- Any other embedding model you've pulled to Ollama

### Custom Ollama Endpoint

You can specify a custom Ollama endpoint:

```python
encoder = OllamaEncoder(
    name="nomic-embed-text",
    base_url="http://my-ollama-server:11434"
)
```

### Asynchronous Support

The encoder supports both synchronous and asynchronous operations:

```python
# Synchronous encoding
embeddings = encoder(["your text here"])

# Asynchronous encoding
embeddings = await encoder.acall(["your text here"])
```

## Integration with Routers

The `OllamaEncoder` works with both `SemanticRouter` and `HybridRouter`:

```python
from semantic_router.routers import SemanticRouter
from semantic_router.route import Route

routes = [
    Route(
        name="technical",
        utterances=["How does this work?", "Explain the system"]
    ),
    Route(
        name="support",
        utterances=["I need help", "Can you assist?"]
    )
]

router = SemanticRouter(
    encoder=encoder,
    routes=routes,
    auto_sync="local"
)
```

## Best Practices

1. **Model Selection**: Choose an embedding model that balances quality and performance for your use case
2. **Ollama Setup**: Ensure Ollama is running before initializing the encoder
3. **Resource Management**: Monitor system resources when using larger models
4. **Model Availability**: Pull required models before deploying to production
5. **Network Configuration**: Use local endpoints when possible for lowest latency

## Advantages

- **Privacy**: All data stays on your infrastructure
- **Control**: Full control over model versions and updates
- **Cost**: No API costs or rate limits
- **Latency**: Low latency with local hosting
- **Offline**: Works without internet connection

## Example Usage

```python
from semantic_router.encoders import OllamaEncoder
from semantic_router.routers import SemanticRouter
from semantic_router.route import Route

# Initialize encoder
encoder = OllamaEncoder(name="nomic-embed-text")

# Define routes
routes = [
    Route(name="greeting", utterances=["hello", "hi", "hey"]),
    Route(name="goodbye", utterances=["bye", "goodbye", "see you"])
]

# Create router
router = SemanticRouter(encoder=encoder, routes=routes, auto_sync="local")

# Route queries
print(router("hi there").name)  # -> greeting
```
