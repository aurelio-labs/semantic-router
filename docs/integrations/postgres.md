Semantic Router integrates with PostgreSQL using the pgvector extension through the `PostgresIndex` class. This integration enables scalable vector storage and retrieval using your existing PostgreSQL infrastructure.

## Overview

The `PostgresIndex` leverages PostgreSQL's pgvector extension for vector similarity search. It supports multiple index types (FLAT, IVFFLAT, HNSW) and both synchronous and asynchronous operations, making it suitable for production deployments.

## Getting Started

### Prerequisites

1. PostgreSQL database with pgvector extension installed
2. Database credentials
3. Semantic Router version 0.1.9 or later (with psycopg v3)

### Installation

```bash
pip install "semantic-router[postgres]"
```

### Database Setup

Ensure pgvector is installed in your PostgreSQL database:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### Basic Usage

```python
import os
from semantic_router.index.postgres import PostgresIndex

# Set environment variables
os.environ["POSTGRES_HOST"] = "localhost"
os.environ["POSTGRES_PORT"] = "5432"
os.environ["POSTGRES_DB"] = "routes_db"
os.environ["POSTGRES_USER"] = "postgres"
os.environ["POSTGRES_PASSWORD"] = "password"

# Or use connection string directly
connection_str = (
    f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
    f"@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DB']}"
)

index = PostgresIndex(
    connection_string=connection_str,
    index_name="my_routes"
)
```

## Features

### Index Types

PostgreSQL index supports multiple index types via `IndexType`:

```python
from semantic_router.index.postgres import PostgresIndex, IndexType

# FLAT index (default) - exact search, slower at scale
index = PostgresIndex(connection_string=connection_str, index_type=IndexType.FLAT)

# IVFFLAT - approximate search, faster for medium scale
index = PostgresIndex(connection_string=connection_str, index_type=IndexType.IVFFLAT)

# HNSW - approximate search, best for large scale (millions of vectors)
index = PostgresIndex(connection_string=connection_str, index_type=IndexType.HNSW)
```

**Note**: Both `FLAT` and `IVFFLAT` map to pgvector's IVFFLAT index. `FLAT` is the default for consistency with other Semantic Router indexes.

### Asynchronous Support

Full async support added in v0.1.10:

```python
import asyncio

async def main():
    result = await router.acall("hello")
    print(result.name)

asyncio.run(main())
```

### Connection Management

The index automatically manages connections and provides cleanup:

```python
# Close connection when done
index.close()
```

## Integration with Routers

The `PostgresIndex` works with both `SemanticRouter` and `HybridRouter`:

```python
from semantic_router.encoders import OpenAIEncoder
from semantic_router.route import Route
from semantic_router.routers import SemanticRouter

encoder = OpenAIEncoder()

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
    index=index,
    auto_sync="local"
)
```

## Index Operations

### Query Routes

```python
# Route a query
result = router("I like voting. What do you think about the president?")
print(result.name)  # -> politics
```

### Delete Routes

```python
# Delete a specific route
router.delete(route_name="chitchat")
```

### Get Routes

```python
# Get all routes
all_routes = index.get_routes()

# Get specific route IDs
route_ids = index._get_route_ids(route_name="politics")
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

1. **Index Type Selection**:
   - Use `FLAT` for small datasets (< 100K vectors) requiring exact search
   - Use `IVFFLAT` for medium datasets (100K - 1M vectors)
   - Use `HNSW` for large datasets (> 1M vectors) - note higher memory requirements

2. **Connection Pooling**: Use connection pooling in production for better performance

3. **Index Maintenance**: Regularly vacuum and analyze PostgreSQL tables

4. **Monitoring**: Monitor query performance and index size

5. **Backup**: Include vector tables in your PostgreSQL backup strategy

## Advantages

- **Infrastructure Integration**: Use existing PostgreSQL infrastructure
- **ACID Compliance**: Full transactional support
- **Cost Effective**: No additional vector database costs
- **Familiar Tooling**: Use standard PostgreSQL tools and monitoring
- **Scalability**: Supports millions of vectors with HNSW indexing

## Upgrade Notes

### v0.1.9 Changes

- **psycopg2 → psycopg v3**: Better performance and Python 3.13 compatibility
- **IndexType Support**: Added `FLAT`, `IVFFLAT`, and `HNSW` index types
- **Standardized Methods**: Consistent sync methods across all indexes

### v0.1.10 Changes

- **Async Support**: Full async/await support for PostgresIndex

## Example Notebook

For a complete example of using the PostgreSQL integration, see the [PostgreSQL Sync Notebook](../indexes/postgres-sync.ipynb).
