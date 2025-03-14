Indexes are critical components in Semantic Router that store and retrieve embeddings efficiently. They act as the search backend that enables Semantic Router to find the most relevant routes for incoming queries.

## Understanding Indexes

In Semantic Router, an index serves several key purposes:

1. **Store embeddings** of route utterances
2. **Search for similar vectors** when routing queries
3. **Persist route configurations** across sessions
4. **Scale to handle large numbers** of routes and utterances

The choice of index can significantly impact the performance, scalability, and persistence capabilities of your semantic routing system.

## Local vs. Remote Indexes

Semantic Router supports both local (in-memory) and remote (cloud-based) indexes:

### Local Indexes

Local indexes store embeddings in memory, making them fast but ephemeral. They're perfect for development, testing, or applications with a small number of routes.

**Example usage**:

```python
from semantic_router.index import LocalIndex
from semantic_router.routers import SemanticRouter
from semantic_router.encoders import OpenAIEncoder
import os

# Set up API key
os.environ["OPENAI_API_KEY"] = "your-api-key"

# Create routes
from semantic_router import Route
routes = [
    Route(name="weather", utterances=["How's the weather?", "Is it raining?"]),
    Route(name="politics", utterances=["Tell me about politics", "Who's the president?"])
]

# Initialize the local index
index = LocalIndex()

# Create a router with the local index
router = SemanticRouter(
    encoder=OpenAIEncoder(),
    routes=routes,
    index=index
)

# Use the router
result = router("What's the weather like today?")
print(result.name)  # "weather"
```

### Remote Indexes

Remote indexes store embeddings in cloud-based vector databases, making them persistent and scalable. They're ideal for production applications or systems with many routes.

**Example usage with Pinecone**:

```python
from semantic_router.index import PineconeIndex
from semantic_router.routers import SemanticRouter
from semantic_router.encoders import OpenAIEncoder
import os

# Set up API keys
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
os.environ["PINECONE_API_KEY"] = "your-pinecone-api-key"

# Create routes
from semantic_router import Route
routes = [
    Route(name="weather", utterances=["How's the weather?", "Is it raining?"]),
    Route(name="politics", utterances=["Tell me about politics", "Who's the president?"])
]

# Initialize the Pinecone index
index = PineconeIndex(
    index_name="semantic-router",
    dimensions=1536  # Must match your encoder's dimension
)

# Create a router with the Pinecone index
router = SemanticRouter(
    encoder=OpenAIEncoder(),
    routes=routes,
    index=index,
    auto_sync="remote"  # Automatically sync routes to remote index
)

# Use the router
result = router("What's the weather like today?")
print(result.name)  # "weather"
```

## Hybrid Indexes

For advanced use cases, Semantic Router also provides a hybrid index that combines both dense and sparse embeddings:

```python
from semantic_router.index import HybridLocalIndex
from semantic_router.routers import HybridRouter
from semantic_router.encoders import OpenAIEncoder, AurelioSparseEncoder
import os

# Set up API keys
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
os.environ["AURELIO_API_KEY"] = "your-aurelio-api-key"

# Initialize dense and sparse encoders
dense_encoder = OpenAIEncoder()
sparse_encoder = AurelioSparseEncoder()

# Create a hybrid index
index = HybridLocalIndex()

# Create a hybrid router
router = HybridRouter(
    encoder=dense_encoder,
    sparse_encoder=sparse_encoder,
    routes=routes,
    index=index,
    alpha=0.5  # Balance between dense (0) and sparse (1) embeddings
)
```

## Supported Indexes

| Index | Description | Installation |
|-------|-------------|-------------|
| [LocalIndex](https://semantic-router.aurelio.ai/api/index/local) | In-memory index for development and testing | `pip install -qU semantic-router` |
| [HybridLocalIndex](https://semantic-router.aurelio.ai/api/index/hybrid_local) | In-memory index supporting hybrid search | `pip install -qU "semantic-router[hybrid]"` |
| [PineconeIndex](https://semantic-router.aurelio.ai/api/index/pinecone) | Pinecone vector database integration | `pip install -qU "semantic-router[pinecone]"` |
| [QdrantIndex](https://semantic-router.aurelio.ai/api/index/qdrant) | Qdrant vector database integration | `pip install -qU "semantic-router[qdrant]"` |
| [PostgresIndex](https://semantic-router.aurelio.ai/api/index/postgres) | PostgreSQL with pgvector extension | `pip install -qU "semantic-router[postgres]"` |

## Auto-Sync Feature

Semantic Router provides an auto-sync feature that keeps your routes in sync between local and remote indexes:

```python
# Initialize router with auto-sync to remote index
router = SemanticRouter(
    encoder=encoder,
    routes=routes,
    index=remote_index,
    auto_sync="remote"  # Options: "local", "remote", None
)

# Add a new route - it will be automatically synced to the remote index
new_route = Route(name="greetings", utterances=["Hello there", "Hi, how are you?"])
router.add(new_route)
```

Auto-sync modes:
- `"local"`: Sync from remote to local (pull)
- `"remote"`: Sync from local to remote (push)
- `None`: No automatic syncing

## Considerations for Choosing an Index

When selecting an index for your application, consider:

1. **Persistence**: Local indexes are lost when your application restarts; remote indexes persist
2. **Scalability**: Remote indexes can handle millions of vectors; local indexes are limited by memory
3. **Latency**: Local indexes have lower latency; remote indexes add network overhead
4. **Setup complexity**: Local indexes require no setup; remote indexes require account creation and configuration
5. **Cost**: Local indexes are free; remote indexes may incur usage costs
6. **Hybrid search**: Only certain indexes support combined dense and sparse search

## Index Methods

All indexes in Semantic Router inherit from `BaseIndex` and implement these key methods:

- `add()`: Add embeddings to the index
- `query()`: Search for similar vectors
- `delete()`: Remove routes from the index
- `describe()`: Get information about the index
- `is_ready()`: Check if the index is initialized and ready for use

For detailed information on specific indexes and their configuration options, refer to their respective documentation pages. 