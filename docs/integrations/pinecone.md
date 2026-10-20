Semantic Router integrates with Pinecone for scalable vector storage and retrieval through the `PineconeIndex` class. This integration enables efficient semantic routing at any scale using Pinecone's serverless or pod-based indexes.

## Overview

The `PineconeIndex` provides high-performance vector storage and similarity search for semantic routing. It supports both synchronous and asynchronous operations, making it suitable for production deployments requiring low latency and high throughput.

## Getting Started

### Prerequisites

1. Pinecone account and API key ([https://www.pinecone.io/](https://www.pinecone.io/))
2. Semantic Router version 0.1.12 or later (with Pinecone v7 support)

### Installation

```bash
pip install "semantic-router[pinecone]"
```

### Environment Variables

Set these environment variables for easier configuration:

- `PINECONE_API_KEY` (required): Your Pinecone API key
- `PINECONE_API_BASE_URL` (optional):
  - Cloud: `https://api.pinecone.io` (default)
  - Local emulator: `http://localhost:5080`
- `PINECONE_INDEX_NAME` (recommended): Name of existing index to reuse

**Why set `PINECONE_INDEX_NAME`?** Pinecone serverless has per-project index limits. Reusing a shared index avoids 403 quota errors. Semantic Router automatically isolates data using namespaces.

### Basic Usage (Cloud)

```python
import os
from semantic_router.encoders import OpenAIEncoder
from semantic_router.index.pinecone import PineconeIndex
from semantic_router.route import Route
from semantic_router.routers import SemanticRouter

# Required
os.environ["PINECONE_API_KEY"] = "your-api-key"

# Recommended: reuse existing index to avoid quota
os.environ["PINECONE_INDEX_NAME"] = "semantic-router-shared"

encoder = OpenAIEncoder(name="text-embedding-3-small")

# Use namespace for isolation
index = PineconeIndex(
    index_name="demo-index",
    namespace="demo",
    dimensions=1536
)

routes = [
    Route(name="greeting", utterances=["hello", "hi"]),
    Route(name="goodbye", utterances=["bye", "goodbye"])
]

router = SemanticRouter(encoder=encoder, routes=routes, index=index, auto_sync="local")

print(router(text="hi there").name)  # -> greeting
```

## Features

### Namespace Isolation

Namespaces provide logical separation within a single index:

```python
# Different namespaces for different use cases
index_prod = PineconeIndex(index_name="shared-index", namespace="production")
index_dev = PineconeIndex(index_name="shared-index", namespace="development")
```

### Index Reuse

If `PINECONE_INDEX_NAME` is set, Semantic Router reuses that index and writes route vectors under your specified namespace. This avoids hitting index creation quotas.

### Asynchronous Support

Full async support for high-throughput applications:

```python
import asyncio

async def main():
    result = await router.acall("hello")
    print(result.name)

asyncio.run(main())
```

### Local Emulator

For testing without cloud costs:

```bash
export PINECONE_API_KEY=pclocal
export PINECONE_API_BASE_URL=http://localhost:5080
```

The local emulator is useful for:
- Development and testing
- CI/CD pipelines
- Offline development

## Pinecone v7 Upgrade

Semantic Router 0.1.12+ uses Pinecone Python SDK v7, which includes:
- Improved data plane performance
- Better async support
- Simplified API structure
- Enhanced error handling

The upgrade is transparent - existing code should continue working without changes.

## Integration with Routers

The `PineconeIndex` works with both `SemanticRouter` and `HybridRouter`:

```python
from semantic_router.routers import SemanticRouter

router = SemanticRouter(
    encoder=encoder,
    routes=routes,
    index=index,
    auto_sync="local"  # or "remote" or None
)
```

### Auto-Sync Modes

- `auto_sync="local"`: Faster, syncs only when routes change
- `auto_sync="remote"`: Always syncs with remote index
- `auto_sync=None`: Manual sync control

## Best Practices

1. **Index Reuse**: Set `PINECONE_INDEX_NAME` in production to reuse indexes and avoid quota limits
2. **Namespaces**: Use namespaces to isolate different applications or environments
3. **Dimensions**: Ensure index dimensions match your encoder's output dimensions
4. **Error Handling**: Handle 403 (quota) and 404 (not found) errors appropriately
5. **Monitoring**: Track index usage and query latency in production

## Error Handling

Common errors and solutions:

- **403 (Quota)**: Set `PINECONE_INDEX_NAME` to reuse an existing index
- **404 (Not Found)**: Index or namespace doesn't exist yet (will be created)
- **Dimension Mismatch**: Ensure encoder dimensions match index dimensions

## Testing without Pinecone cloud

Pinecone's local emulator lets you run your own test suite against `PineconeIndex`
with no API key and no quota:

```bash
docker run -d --name pinecone-local \
  -e PORT=5080 -e PINECONE_HOST=localhost \
  -p 5080-5199:5080-5199 \
  ghcr.io/pinecone-io/pinecone-local:latest
```

```bash
export PINECONE_API_KEY=pclocal
export PINECONE_API_BASE_URL=http://localhost:5080
```

The emulator opens one extra port per index, counting up from 5081, so publish a
range wide enough for the number of indexes your tests create, and delete
indexes when you are done with them. `PineconeIndex` treats any `http://` base
URL as the emulator and skips the cloud-only calls.

## Example Notebooks

For complete examples, see:
- [Pinecone Sync Routes](../indexes/pinecone-sync-routes.ipynb)
- [Pinecone Async](../indexes/pinecone-async.ipynb)
- [Pinecone Local Emulator](../indexes/pinecone-local.ipynb)
