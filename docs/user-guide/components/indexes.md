---
title: Indexes
description: Where an IndexedRouter stores its route vectors, and how to share one index between routers.
---

An index is where an `IndexedRouter` stores the vectors of your route utterances and searches them. It's the search backend behind every decision the router makes. (A `DecisionRouter` doesn't use one.)

## What an index does

1. **Stores** the embedding of every utterance, with its route.
2. **Searches** for the stored vectors closest to each query.
3. **Persists** your routes across restarts, if it's a remote index.
4. **Scales** to large numbers of routes and utterances.

## The index is the source of truth

In version 1 the index holds your routes, and the router is a client of it:

- **Building a router writes its routes to the index.** Each record's id comes from its route and utterance, so writing the same routes again changes nothing. Two processes can start up with the same routes and the index ends up the same.
- **Passing `routes=` adds or updates, and never deletes.** Routes already in the index that aren't in your list stay there. Remove one with `router.remove(name=...)`.
- **`router.routes` is a view of the index.** The router fills it from the index when it's built, after its own `add()` and `remove()`, and whenever you refresh it. It never writes routes anywhere else. See [sharing an index](#sharing-an-index).

There is no sync step, and no `auto_sync`: there's only one copy of your routes to keep right. (0.x kept a second copy in the router and reconciled the two. The [migration guide](../guides/migration-to-v1#sync-is-gone) covers moving off it.)

## Local indexes

A local index keeps the vectors in memory. It's fast and needs no setup, but it's gone when your process exits. It's the default when you don't pass an index, and ideal for development, tests and small route sets.

```python
from semantic_router import IndexedRouter, Route
from semantic_router.encoders import FastEmbedEncoder
from semantic_router.index import LocalIndex

routes = [
    Route(name="weather", utterances=["How's the weather?", "Is it raining?"]),
    Route(name="politics", utterances=["Tell me about politics", "Who's the president?"]),
]

router = IndexedRouter(
    encoder=FastEmbedEncoder(name="BAAI/bge-small-en-v1.5"),
    routes=routes,
    index=LocalIndex(),
)

router("What's the weather like today?").name
```

```
[Out]: 'weather'
```

## Remote indexes

A remote index stores the vectors in a vector database. Your routes persist, many processes can share them, and you can scale to millions of vectors. This is what you want in production.

Here's Pinecone:

```python
import os

from semantic_router import IndexedRouter
from semantic_router.encoders import OpenAIEncoder
from semantic_router.index import PineconeIndex

os.environ["OPENAI_API_KEY"] = "<YOUR_OPENAI_API_KEY>"
os.environ["PINECONE_API_KEY"] = "<YOUR_PINECONE_API_KEY>"

index = PineconeIndex(
    index_name="semantic-router",
    dimensions=1536,  # must match your encoder
)

router = IndexedRouter(
    encoder=OpenAIEncoder(),
    routes=routes,
    index=index,
)
```

Qdrant can run in memory, which is handy for trying a remote-style index without a server:

```python
from semantic_router.index import QdrantIndex

router = IndexedRouter(
    encoder=FastEmbedEncoder(name="BAAI/bge-small-en-v1.5"),
    routes=routes,
    index=QdrantIndex(index_name="routes", location=":memory:"),
)
```

Postgres works through pgvector:

```python
from semantic_router.index import PostgresIndex

index = PostgresIndex(
    connection_string="postgresql://user:password@localhost:5432/routes",
    index_name="routes",
    dimensions=384,  # must match your encoder
)
```

MongoDB Atlas works through Atlas Vector Search. The index creates its collection and vector search index on the first `add`, so there's nothing to set up in Atlas first:

```python
from semantic_router.index import MongoDbIndex

index = MongoDbIndex(
    connection_uri="mongodb+srv://user:password@cluster.mongodb.net/",  # or set MONGO_CONNECTION_URI
    database_name="semantic_router",
    index_name="routes",
)
```

Atlas makes new records searchable a moment after they're written, so `add` waits until they are (up to `sync_timeout`, 30 seconds by default). For local development, the `mongodb/mongodb-atlas-local` Docker image supports the same vector search.

## Sharing an index

Because the index holds the routes, several routers can share one. To connect to an index that's already filled, pass it without `routes`: the router reads them from the index.

```python
shared = QdrantIndex(index_name="support", location=":memory:")
encoder = FastEmbedEncoder(name="BAAI/bge-small-en-v1.5")

writer = IndexedRouter(encoder=encoder, routes=routes, index=shared)
reader = IndexedRouter(encoder=encoder, index=shared)

sorted(r.name for r in reader.routes)
```

```
[Out]: ['politics', 'weather']
```

Each router keeps its own cached copy of the route list, so routing a query never re-reads every route from the index. That read grows with the number of utterances, and on Pinecone it can take seconds. The cache fills when the router is built and after its own `add()` and `remove()`. Changes another router writes show up when you refresh:

```python
writer.add(routes=Route(name="greetings", utterances=["hello there", "hi, how are you?"]))

reader.refresh()
sorted(r.name for r in reader.routes)
```

```
[Out]: ['greetings', 'politics', 'weather']
```

Use `await router.arefresh()` in async code. To pick up other writers' changes without calling it yourself, set `refresh_interval`, in seconds. When the cache is older than that, the next call re-reads the routes while it embeds the query:

```python
reader = IndexedRouter(encoder=encoder, index=shared, refresh_interval=60)
```

With the default, `refresh_interval=None`, a call never reads routes from the index.

The search itself always runs against the index, so a stale cache never hides a route: if another writer adds a route, the reader returns it for a matching query even before it refreshes. What the cache holds is the route list (`router.routes`) and each route's settings. Until a refresh, a route the reader hasn't cached is held to the router's `confidence_threshold`, because its own isn't known yet. So refresh after another writer changes routes' settings, such as a route's threshold.

Every router on a shared index must use the same encoder: the stored vectors only mean something to the model that made them.

## Hybrid indexes

For [hybrid search](routers#hybrid-search), the index has to store sparse vectors as well as dense ones. Two do:

- `HybridLocalIndex`, in memory. A hybrid `IndexedRouter` uses it when you don't pass an index.
- `PineconeIndex`, remote.

`LocalIndex`, `QdrantIndex`, `PostgresIndex` and `MongoDbIndex` are dense only: give a router one of them along with a `sparse_encoder` and it raises an error.

## Supported indexes

| Index | What it is | Hybrid | Install |
|-------|------------|--------|---------|
| [LocalIndex](../../client-reference/index/local) | In memory, for development and testing | no | base |
| [HybridLocalIndex](../../client-reference/index/hybrid_local) | In memory, dense and sparse | yes | base |
| [PineconeIndex](../../client-reference/index/pinecone) | Pinecone vector database | yes | `pip install -qU "semantic-router[pinecone]"` |
| [QdrantIndex](../../client-reference/index/qdrant) | Qdrant vector database | no | `pip install -qU "semantic-router[qdrant]"` |
| [PostgresIndex](../../client-reference/index/postgres) | PostgreSQL with pgvector | no | `pip install -qU "semantic-router[postgres]"` |
| [MongoDbIndex](../../client-reference/index/mongodb) | MongoDB Atlas Vector Search | no | `pip install -qU "semantic-router[mongodb]"` |

## Choosing an index

1. **Persistence.** Local indexes vanish on restart. Remote ones don't.
2. **Sharing.** Only a remote index can be shared between processes.
3. **Scale.** Local is bounded by memory. Remote handles millions of vectors.
4. **Latency.** Local is fastest. Remote adds a network hop to every query.
5. **Setup and cost.** Local needs nothing and costs nothing. Remote needs a service.
6. **Hybrid search.** Only `HybridLocalIndex` and `PineconeIndex` store sparse vectors.

## Index methods

You rarely call an index directly; the router does it for you. Every index implements:

- `add()` and `aadd()`: write records.
- `query()` and `aquery()`: find the stored vectors closest to a query.
- `delete()` and `adelete()`: remove a route's records.
- `get_routes()` and `aget_routes()`: rebuild the routes from the stored records.
- `delete_index()`: drop everything.

Each index's reference page covers its own options.
