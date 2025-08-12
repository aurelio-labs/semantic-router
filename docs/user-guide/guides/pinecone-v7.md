## Pinecone v7 integration

This guide shows how to use Semantic Router with the Pinecone Python SDK v7+, including cloud vs local setup, shared-index reuse, and namespaces for isolation.

### Install

```bash
pip install "semantic-router[pinecone]"
```

### Environment variables

- `PINECONE_API_KEY` (required): Your Pinecone API key
- `PINECONE_API_BASE_URL` (optional):
  - Cloud: `https://api.pinecone.io` (default if a real API key is set)
  - Local emulator: `http://localhost:5080` or `http://pinecone:5080`
- `PINECONE_INDEX_NAME` (recommended on cloud): Name of an existing index to reuse
- `PINECONE_TRANSPORT` (optional): Set to `grpc` to enable gRPC data plane (default: HTTP)

Why set `PINECONE_INDEX_NAME`? Pinecone serverless has per-project index limits. Reusing a shared index avoids 403 quota errors. Semantic Router will automatically isolate data using namespaces.

### Basic usage (cloud)

```python
import os
from semantic_router.encoders import OpenAIEncoder
from semantic_router.index.pinecone import PineconeIndex
from semantic_router.route import Route
from semantic_router.routers import SemanticRouter

# Required
os.environ["PINECONE_API_KEY"] = "<your-key>"

# Strongly recommended: reuse an existing index to avoid quota
os.environ["PINECONE_INDEX_NAME"] = "semantic-router-shared"

encoder = OpenAIEncoder(name="text-embedding-3-small")

# Use a namespace for isolation (otherwise the router will use the requested
# index name internally as the namespace when reusing a shared index)
index = PineconeIndex(
    index_name="demo-index",
    namespace="demo",
    dimensions=1536,
    # Optional index creation options (cloud only):
    deletion_protection="disabled",
    tags={"team": "search", "env": "prod"},
    # Optional: use gRPC by argument instead of env
    transport="http",  # or "grpc"
)

routes = [
    Route(name="greeting", utterances=["hello", "hi"]),
    Route(name="goodbye", utterances=["bye", "goodbye"]),
]

router = SemanticRouter(encoder=encoder, routes=routes, index=index, auto_sync="local")

print(router(text="hi there").name)  # -> greeting
```

Notes:
- If the shared index exists, Semantic Router reuses it and writes route vectors under your `namespace`.
- If you do not set `PINECONE_INDEX_NAME`, creating a new index requires `dimensions`. If index creation is forbidden (quota), a clear error is raised asking you to set `PINECONE_INDEX_NAME`.
- You do not need to set `PINECONE_API_BASE_URL` for cloud; override it only when using the local emulator for testing.

### Local emulator

```bash
export PINECONE_API_KEY=pclocal
export PINECONE_API_BASE_URL=http://localhost:5080
```

In local mode, Semantic Router connects to the emulator at `http://localhost:5080` (or `http://pinecone:5080` in containerized CI) and adds a short delay after create to account for readiness.

### Async usage

```python
import asyncio
from semantic_router.routers import SemanticRouter

async def main():
    result = await router.acall("hello")
    print(result.name)

asyncio.run(main())
```

Internally, the library resolves the Pinecone v7 data-plane host and uses the correct `/vectors/query` endpoint for async queries.

### Query across namespaces

To query across multiple namespaces (useful when reusing a shared index), use the helpers:

```python
# Sync
merged = index.query_across_namespaces(
    vector=embedding,
    namespaces=["team-a", "team-b"],
    top_k=10,
)

# Async
merged_async = await index.aquery_across_namespaces(
    vector=embedding,
    namespaces=["team-a", "team-b"],
    top_k=10,
)
```

### Error handling and retries

- 403 (quota): The library attempts to reuse an existing index. If none is available, it raises an error advising you to set `PINECONE_INDEX_NAME`.
- 404 (eventual consistency): Readiness checks and upserts include brief bounded retries.

### CI tips (GitHub Actions)

- Set secrets:
  - `PINECONE_API_KEY`
  - `PINECONE_INDEX_NAME` (existing shared index)
- Ensure the environment uses cloud:

```yaml
env:
  PINECONE_API_KEY: ${{ secrets.PINECONE_API_KEY }}
  PINECONE_INDEX_NAME: ${{ secrets.PINECONE_INDEX_NAME }}

steps:
  - name: Run tests
    run: |
      PINECONE_API_BASE_URL="https://api.pinecone.io" \ 
      pytest -q
```

Tests that require Pinecone will automatically skip in cloud mode if `PINECONE_INDEX_NAME` isn’t provided, avoiding quota-based failures.

### Requirements recap

- Pinecone Python client v7+
- Semantic Router ≥ version including Pinecone v7 support (this branch)
- Recommended on cloud: `PINECONE_INDEX_NAME` pointing at an existing index

