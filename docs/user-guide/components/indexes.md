An index is where Semantic Router stores your route embeddings and searches them. It's the search backend behind every routing decision.

## What an index does

An index handles four things:

1. **Stores** the embeddings of your route utterances.
2. **Searches** for the most similar vectors when a query comes in.
3. **Persists** your routes across sessions (remote indexes only).
4. **Scales** to large numbers of routes and utterances.

Your choice of index shapes performance, scale, and whether your routes survive a restart.

## Local vs. remote

### Local indexes

A local index keeps embeddings in memory. It's fast and needs zero setup, but it's gone when your process exits. Ideal for development, testing, and small route sets.

```python
import os
from semantic_router import Route, SemanticRouter
from semantic_router.encoders import OpenAIEncoder
from semantic_router.index import LocalIndex

os.environ["OPENAI_API_KEY"] = "your-api-key"

routes = [
    Route(name="weather", utterances=["How's the weather?", "Is it raining?"]),
    Route(name="politics", utterances=["Tell me about politics", "Who's the president?"]),
]

router = SemanticRouter(
    encoder=OpenAIEncoder(),
    routes=routes,
    index=LocalIndex(),
)

result = router("What's the weather like today?")
print(result.name)  # "weather"
```

### Remote indexes

A remote index stores embeddings in a vector database. Your routes persist, and you can scale to millions of vectors. This is what you want in production.

Here's Pinecone:

```python
import os
from semantic_router import Route, SemanticRouter
from semantic_router.encoders import OpenAIEncoder
from semantic_router.index import PineconeIndex

os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
os.environ["PINECONE_API_KEY"] = "your-pinecone-api-key"

routes = [
    Route(name="weather", utterances=["How's the weather?", "Is it raining?"]),
    Route(name="politics", utterances=["Tell me about politics", "Who's the president?"]),
]

index = PineconeIndex(
    index_name="semantic-router",
    dimensions=1536,  # must match your encoder
)

router = SemanticRouter(
    encoder=OpenAIEncoder(),
    routes=routes,
    index=index,
    auto_sync="remote",  # push local routes to the remote index
)

result = router("What's the weather like today?")
print(result.name)  # "weather"
```

## Hybrid indexes

For hybrid routing, you need an index that stores both dense and sparse vectors. `HybridLocalIndex` does that in memory:

```python
import os
from semantic_router.routers import HybridRouter
from semantic_router.encoders import OpenAIEncoder, AurelioSparseEncoder
from semantic_router.index import HybridLocalIndex

os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
os.environ["AURELIO_API_KEY"] = "your-aurelio-api-key"

router = HybridRouter(
    encoder=OpenAIEncoder(),
    sparse_encoder=AurelioSparseEncoder(),
    routes=routes,
    index=HybridLocalIndex(),
    alpha=0.5,  # 0 = all dense, 1 = all sparse
)
```

## Supported indexes

| Index | What it is | Install |
|-------|------------|---------|
| [LocalIndex](../../client-reference/index/local) | In-memory, for development and testing | base |
| [HybridLocalIndex](../../client-reference/index/hybrid_local) | In-memory, dense and sparse | base |
| [PineconeIndex](../../client-reference/index/pinecone) | Pinecone vector database | `pip install -qU "semantic-router[pinecone]"` |
| [QdrantIndex](../../client-reference/index/qdrant) | Qdrant vector database | `pip install -qU "semantic-router[qdrant]"` |
| [PostgresIndex](../../client-reference/index/postgres) | PostgreSQL with pgvector | `pip install -qU "semantic-router[postgres]"` |
| [RedisIndex](../../client-reference/index/redis) | Redis with RediSearch vector search | `pip install -qU "semantic-router[redis]"` |

## Keeping local and remote in sync

With a remote index, your in-memory routes and the stored ones can drift. The `auto_sync` parameter tells the router how to reconcile them at startup:

```python
router = SemanticRouter(
    encoder=encoder,
    routes=routes,
    index=remote_index,
    auto_sync="remote",  # "local", "remote", or None
)

# adding a route syncs it automatically
router.add(Route(name="greetings", utterances=["Hello there", "Hi, how are you?"]))
```

- `"local"` — local is the source of truth; push it to remote.
- `"remote"` — remote is the source of truth; pull it into local.
- `None` — don't sync.

There are more strategies, including merges. The [sync guide](../features/sync) covers them all.

## Choosing an index

1. **Persistence.** Local indexes vanish on restart. Remote ones don't.
2. **Scale.** Local is bounded by memory. Remote handles millions of vectors.
3. **Latency.** Local is fastest. Remote adds a network hop.
4. **Setup.** Local needs nothing. Remote needs an account and config.
5. **Cost.** Local is free. Remote may bill for usage.
6. **Hybrid search.** Only some indexes store both dense and sparse vectors.

## Index methods

Every index inherits from `BaseIndex` and implements:

- `add()` — add embeddings.
- `query()` — search for similar vectors.
- `delete()` — remove a route.
- `describe()` — get index info.
- `is_ready()` — check it's initialized.

Each index's reference page covers its configuration options.
