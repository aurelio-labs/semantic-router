Semantic Router integrates with Qdrant for high-performance vector storage and retrieval through the `QdrantIndex` class. This integration provides scalable semantic routing using Qdrant's efficient vector search capabilities.

## Overview

The `QdrantIndex` enables semantic routing backed by Qdrant's vector database. It supports both cloud and self-hosted deployments, with full synchronous and asynchronous operation support.

## Getting Started

### Prerequisites

1. Qdrant instance (cloud or self-hosted)
2. Semantic Router version 0.1.9 or later

### Installation

```bash
pip install "semantic-router[qdrant]"
```

### Basic Usage

```python
from semantic_router.index.qdrant import QdrantIndex

# Cloud Qdrant
index = QdrantIndex(
    url="https://your-cluster.qdrant.io",
    api_key="your-api-key",
    collection_name="semantic_router"
)

# Self-hosted Qdrant
index = QdrantIndex(
    url="http://localhost:6333",
    collection_name="semantic_router"
)
```

## Features

### Cloud and Self-Hosted

Supports both Qdrant Cloud and self-hosted deployments:

```python
# Cloud deployment
index = QdrantIndex(
    url="https://xyz-example.qdrant.io",
    api_key="your-api-key",
    collection_name="routes"
)

# Local/self-hosted
index = QdrantIndex(
    url="http://localhost:6333",
    collection_name="routes"
)
```

### Collection Management

Automatic collection creation and management:

```python
# Specify dimensions for new collection
index = QdrantIndex(
    url="http://localhost:6333",
    collection_name="my_routes",
    dimensions=1536  # Must match encoder dimensions
)
```

### Asynchronous Support

Full async/await support added in v0.1.10:

```python
import asyncio

async def main():
    result = await router.acall("hello")
    print(result.name)

asyncio.run(main())
```

## Integration with Routers

The `QdrantIndex` works with both `SemanticRouter` and `HybridRouter`:

```python
from semantic_router.encoders import OpenAIEncoder
from semantic_router.route import Route
from semantic_router.routers import SemanticRouter

encoder = OpenAIEncoder()

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
    index=index,
    auto_sync="local"
)
```

## Index Operations

### Query Routes

```python
# Route a query
result = router("How does the system work?")
print(result.name)  # -> technical
```

### Delete Routes

```python
# Delete a specific route
router.delete(route_name="support")
```

### Get Routes

```python
# Get all routes
all_routes = index.get_routes()
```

### Index Info

```python
# Get index configuration
config = index.describe()
print(f"Type: {config.type}, Dimensions: {config.dimensions}, Vectors: {config.vectors}")

# Check number of vectors
print(len(index))
```

## Best Practices

1. **Deployment Type**: Use Qdrant Cloud for managed infrastructure or self-host for full control
2. **Collection Names**: Use descriptive collection names for different use cases
3. **Dimensions**: Ensure collection dimensions match your encoder's output
4. **API Keys**: Store Qdrant Cloud API keys securely using environment variables
5. **Performance**: Use local Qdrant for development, cloud or self-hosted clusters for production

## Advantages

- **High Performance**: Fast vector search optimized for similarity queries
- **Flexible Deployment**: Cloud, self-hosted, or local options
- **Scalability**: Efficiently handles millions of vectors
- **Rich Filtering**: Support for metadata filtering (future Semantic Router feature)
- **Open Source**: Self-hosted option with no vendor lock-in

## Self-Hosting Qdrant

Quick start with Docker:

```bash
docker pull qdrant/qdrant
docker run -p 6333:6333 qdrant/qdrant
```

For production deployments, see [Qdrant documentation](https://qdrant.tech/documentation/).

## Upgrade Notes

### v0.1.10 Changes

- **Async Support**: Full async/await support for QdrantIndex
- **Consistent Behavior**: Aligned with other index implementations

### v0.1.9 Changes

- **Length Method**: Standardized `__len__` method returns 0 when uninitialized

## Example Notebook

For a complete example of using the Qdrant integration, see the [Qdrant Notebook](../indexes/qdrant.ipynb).
