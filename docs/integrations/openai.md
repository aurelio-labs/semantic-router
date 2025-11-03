Semantic Router integrates with OpenAI's embedding models through the `OpenAIEncoder` class. This integration provides access to OpenAI's high-quality text embeddings for semantic routing tasks.

## Overview

The `OpenAIEncoder` enables semantic routing using OpenAI's embedding models, including the latest `text-embedding-3` series. It supports both synchronous and asynchronous operations with customizable dimensions.

## Getting Started

### Prerequisites

1. OpenAI API key ([https://platform.openai.com/api-keys](https://platform.openai.com/api-keys))
2. Semantic Router installed

### Installation

```bash
pip install semantic-router
```

### Basic Usage

```python
import os
from semantic_router.encoders import OpenAIEncoder

os.environ["OPENAI_API_KEY"] = "your-api-key"

encoder = OpenAIEncoder(
    name="text-embedding-3-small",  # or text-embedding-3-large, text-embedding-ada-002
    openai_api_key=os.getenv("OPENAI_API_KEY")
)
```

## Features

### Supported Models

The `OpenAIEncoder` supports all OpenAI embedding models:
- `text-embedding-3-small` - Cost-effective, high performance (default dimensions: 1536)
- `text-embedding-3-large` - Highest quality embeddings (default dimensions: 3072)
- `text-embedding-ada-002` - Legacy model (dimensions: 1536)

### Customizable Dimensions

The `text-embedding-3` models support custom dimensions for optimized storage and performance:

```python
# Smaller, faster embeddings
encoder = OpenAIEncoder(
    name="text-embedding-3-large",
    dimensions=256  # Reduce from 3072 to 256
)
```

According to OpenAI, even 256-dimensional embeddings from `text-embedding-3-large` can outperform the 1536-dimensional `text-embedding-ada-002` embeddings.

### Truncation Support

Automatically truncate long texts to fit within token limits:

```python
embeddings = encoder(docs=documents, truncate=True)
```

### Asynchronous Support

Full async/await support for high-throughput applications:

```python
# Synchronous encoding
embeddings = encoder(["your text here"])

# Asynchronous encoding
embeddings = await encoder.acall(["your text here"])
```

## Integration with Routers

The `OpenAIEncoder` works with both `SemanticRouter` and `HybridRouter`:

```python
from semantic_router.routers import SemanticRouter
from semantic_router.route import Route

routes = [
    Route(
        name="technical",
        utterances=["How does this work?", "Explain the architecture"]
    ),
    Route(
        name="support",
        utterances=["I need help", "Can you assist me?"]
    )
]

router = SemanticRouter(
    encoder=encoder,
    routes=routes,
    auto_sync="local"
)
```

## Best Practices

1. **Model Selection**:
   - Use `text-embedding-3-small` for cost-effective general-purpose embeddings
   - Use `text-embedding-3-large` for highest quality when accuracy is critical

2. **Dimension Optimization**: Reduce dimensions to 256-512 for faster search and lower storage costs while maintaining quality

3. **API Key Management**: Store API keys in environment variables, never hardcode them

4. **Batch Processing**: Process multiple texts in a single call for better rate limit utilization

5. **Error Handling**: Implement retry logic for rate limits and network errors

## Cost Optimization

OpenAI charges per token processed. To optimize costs:
- Use `text-embedding-3-small` when possible (5x cheaper than 3-large)
- Reduce dimensions to minimum required for your use case
- Enable truncation to avoid processing unnecessary tokens
- Batch multiple texts into single API calls

## Example Notebook

For complete examples of using the OpenAI integration, see:
- [OpenAI Encoder Notebook](../encoders/openai-encoder.ipynb)
- [OpenAI Embed-3 Notebook](../encoders/openai-embed-3.ipynb)
