Semantic Router integrates with Redis for high-performance vector storage and retrieval through the `RedisIndex` class. This integration uses [RediSearch](https://redis.io/docs/latest/develop/ai/search-and-query/) vector fields, so it needs a Redis server with the RediSearch module -- e.g. [Redis Stack](https://redis.io/docs/latest/operate/oss_and_stack/install/install-stack/), Redis Cloud, or Redis 8+ (which bundles it by default). Plain open-source Redis without the module has no vector index type and cannot be used.

## Overview

The `RedisIndex` enables semantic routing backed by Redis's vector search. It supports self-hosted, Docker, and managed (Redis Cloud) deployments, with full synchronous and asynchronous operation support.

## Getting Started

### Prerequisites

1. A Redis server with the RediSearch module (Redis Stack, Redis 8+, or Redis Cloud)

### Installation

```bash
pip install "semantic-router[redis]"
```

### Basic Usage

```python
from semantic_router.index.redis import RedisIndex

# Local Redis Stack
index = RedisIndex(
    index_name="semantic_router",
    host="localhost",
    port=6379,
)

# Or from a connection URL (also read from the REDIS_URL env var)
index = RedisIndex(
    index_name="semantic_router",
    connection_string="redis://default:password@your-redis-host:6379",
)
```

Quick start with Docker:

```bash
docker run -p 6379:6379 redis/redis-stack-server:latest
```

## Features

### Connection Options

`RedisIndex` accepts either a full connection string or individual fields, each falling back to an environment variable when unset:

| Field | Env var | Default |
|-------|---------|---------|
| `connection_string` | `REDIS_URL` | -- |
| `host` | `REDIS_HOST` | `localhost` |
| `port` | `REDIS_PORT` | `6379` |
| `username` | `REDIS_USERNAME` | -- |
| `password` | `REDIS_PASSWORD` | -- |

```python
index = RedisIndex(
    index_name="semantic_router",
    connection_string="rediss://default:password@your-redis-host:6379",  # TLS via rediss://
)
```

### Index Management

The RediSearch index is created automatically the first time routes are added, using the encoder's embedding dimensions:

```python
index = RedisIndex(
    index_name="my_routes",
    dimensions=1536,  # optional -- inferred from the encoder if omitted
)
```

### Distance Metrics

Redis supports three distance metrics for vector fields. `RedisIndex` defaults to cosine:

```python
from semantic_router.schema import Metric

index = RedisIndex(index_name="semantic_router", metric=Metric.COSINE)     # -> RediSearch COSINE
index = RedisIndex(index_name="semantic_router", metric=Metric.EUCLIDEAN)  # -> RediSearch L2
index = RedisIndex(index_name="semantic_router", metric=Metric.DOTPRODUCT) # -> RediSearch IP
```

`Metric.MANHATTAN` is not supported by Redis and raises a `ValueError`.

### FLAT vs HNSW

By default `RedisIndex` uses a `FLAT` index (exact search, no tuning needed). For large route sets, switch to `HNSW` and optionally tune it:

```python
index = RedisIndex(
    index_name="semantic_router",
    algorithm="HNSW",
    algorithm_params={"M": 40, "EF_CONSTRUCTION": 250},
)
```

### Namespaces

Like `QdrantIndex`, a `namespace` scopes all reads/writes to a single tenant within a shared index:

```python
index = RedisIndex(index_name="semantic_router", namespace="org-123")
```

### Asynchronous Support

Full async/await support, via `redis.asyncio`:

```python
import asyncio

async def main():
    result = await router.acall("hello")
    print(result.name)

asyncio.run(main())
```

## Integration with Routers

`RedisIndex` works with both `SemanticRouter` and `HybridRouter` (dense search only -- Redis does not support sparse/hybrid vectors):

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
    auto_sync="local",
)
```

## Index Operations

### Query Routes

```python
result = router("How does the system work?")
print(result.name)  # -> technical
```

### Delete Routes

```python
router.delete(route_name="support")
```

### Get Routes

```python
all_routes = index.get_routes()
```

### Index Info

```python
config = index.describe()
print(f"Type: {config.type}, Dimensions: {config.dimensions}, Vectors: {config.vectors}")

print(len(index))
```

## Best Practices

1. **Index name**: `index_name` also becomes the Redis key prefix for every stored record, so keep it unique per logical route set (like a Qdrant collection or Postgres table).
2. **Dimensions**: Ensure they match your encoder's output, or leave them unset to infer from the encoder.
3. **Algorithm**: Use `FLAT` for smaller route sets where exact recall matters; switch to `HNSW` once you have a large number of routes.
4. **Credentials**: Store Redis Cloud/production credentials in environment variables (`REDIS_URL` or `REDIS_PASSWORD`) rather than in code.
5. **Module requirement**: Confirm your Redis deployment includes RediSearch -- plain open-source Redis does not.
