> **You're reading the docs for Semantic Router 0.x.** Version 1.0 is out: read [this page for v1](https://docs.aurelio.ai/docs/semantic-router/v1/user-guide/concepts/architecture), or the [migration guide](https://docs.aurelio.ai/docs/semantic-router/v1/user-guide/guides/migration-to-v1) to upgrade.

# Semantic Router: Technical Architecture

## System overview

Three components do the work. An **encoder** turns the input into a vector. An **index** stores your route vectors. A **router** compares the two and picks a match.

```mermaid
graph TD
    A[Input Query] --> B[Encoder]
    B --> C[Vector Embedding]
    C --> D[Router]
    E[Routes] --> D
    F[Index] <--> D
    D --> G[Matched Route]
    G --> H[Response Handler]
```

## Core components

### 1. Encoders

Encoders map inputs into semantic space.

```mermaid
classDiagram
    class BaseEncoder {
        +encode(text: List[str]) -> Any
        +aencode(text: List[str]) -> Any
    }
    class DenseEncoder {
        +encode() -> dense vectors
    }
    class SparseEncoder {
        +encode() -> sparse vectors
    }
    BaseEncoder <|-- DenseEncoder
    BaseEncoder <|-- SparseEncoder
    DenseEncoder <|-- OpenAIEncoder
    DenseEncoder <|-- HuggingFaceEncoder
    DenseEncoder <|-- CLIPEncoder
    SparseEncoder <|-- AurelioSparseEncoder
    SparseEncoder <|-- BM25Encoder
    SparseEncoder <|-- TFIDFEncoder
```

- **Dense encoders** produce continuous vectors (OpenAI, Hugging Face, …).
- **Sparse encoders** produce mostly-zero vectors (BM25, TF-IDF, AurelioSparse, …).
- **Multimodal encoders** handle images as well as text (CLIP, ViT).

### 2. Routes

A route is a pattern to match, defined by example inputs that should trigger it.

```mermaid
classDiagram
    class Route {
        +name: str
        +utterances: List[str]
        +description: Optional[str]
        +function_schemas: Optional[List[Dict]]
        +score_threshold: Optional[float]
        +metadata: Optional[Dict]
    }
```

- **name** — the route's identifier.
- **utterances** — example inputs that should match.
- **function_schemas** — optional. Set this and the route becomes *dynamic*, able to call functions.
- **score_threshold** — the minimum similarity needed to match.

### 3. Indexes

Indexes store route vectors and search them efficiently.

```mermaid
classDiagram
    class BaseIndex {
        +add(embeddings, routes, utterances)
        +query(vector, top_k) -> matches
        +delete(route_name)
    }
    BaseIndex <|-- LocalIndex
    BaseIndex <|-- PostgresIndex 
    BaseIndex <|-- PineconeIndex
    BaseIndex <|-- QdrantIndex
    LocalIndex <|-- HybridLocalIndex
```

- **LocalIndex** — in-memory, dense vectors.
- **HybridLocalIndex** — in-memory, dense *and* sparse.
- **PineconeIndex / QdrantIndex** — cloud vector databases.
- **PostgresIndex** — SQL storage via pgvector.

## Data flow

```mermaid
sequenceDiagram
    participant User
    participant Router
    participant Encoder
    participant Index
    User->>Router: send query
    Router->>Encoder: encode query
    Encoder->>Router: return vector
    Router->>Index: search for similar routes
    Index->>Router: return matches
    Router->>User: return best matched route
```

1. An input arrives (text, an image).
2. The encoder turns it into a vector.
3. The router searches the index for similar route vectors.
4. The best match above threshold is selected.
5. The matched route comes back, ready for your handler.

## Router types

```mermaid
classDiagram
    class BaseRouter {
        +__call__(query) -> RouteChoice
        +acall(query) -> RouteChoice
        +add(routes)
        +route(query) -> RouteChoice
    }
    BaseRouter <|-- SemanticRouter
    BaseRouter <|-- HybridRouter
```

- **SemanticRouter** — dense embeddings, pure semantic matching.
- **HybridRouter** — dense *and* sparse, for better accuracy when exact keywords matter.

## Putting it together

```python
from semantic_router import Route, SemanticRouter
from semantic_router.encoders import OpenAIEncoder

# 1. Define routes
weather_route = Route(name="weather", utterances=["What's the weather like?"])
greeting_route = Route(name="greeting", utterances=["Hello there!", "Hi!"])

# 2. Pick an encoder
encoder = OpenAIEncoder()

# 3. Build the router
router = SemanticRouter(encoder=encoder, routes=[weather_route, greeting_route])

# 4. Route a query
result = router("What's the forecast for tomorrow?")
print(result.name)  # "weather"
```

## Performance notes

- **In-memory vs. vector DB** — pick by scale and latency needs.
- **Encoder choice** — trade accuracy against speed for your use case.
- **Batching** — use the batch methods for higher throughput.
- **Async** — available for high-concurrency and network-heavy workloads.
