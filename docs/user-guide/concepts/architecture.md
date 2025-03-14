# Semantic Router: Technical Architecture

## System Overview

Semantic Router is built around three core components that work together to enable intelligent routing of inputs based on semantic meaning.

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

## Core Components

### 1. Encoders

Encoders transform inputs into vector representations in semantic space.

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

**Types of Encoders:**
- **Dense encoders**: Generate continuous vectors (OpenAI, HuggingFace, etc.)
- **Sparse encoders**: Generate sparse vectors (BM25, TFIDF, AurelioSparse, etc.)
- **Multimodal encoders**: Handle images and text (CLIP, ViT)

### 2. Routes

Routes define patterns to match against, with examples of inputs that should trigger them.

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

**Key properties:**
- **name**: Identifier for the route
- **utterances**: Example inputs that should match this route
- **function_schemas**: Optional specifications for function calling
- **score_threshold**: Minimum similarity score required to match

### 3. Indexing Systems

Indexes store and retrieve route vectors efficiently.

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

**Index types:**
- **LocalIndex**: In-memory vector storage for dense embeddings
- **HybridLocalIndex**: In-memory storage supporting both dense and sparse vectors
- **PineconeIndex/QdrantIndex**: Cloud-based vector DBs
- **PostgresIndex**: SQL-based vector storage

## Data Flow

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

1. **Input Reception**: The system receives an input (text, image)
2. **Encoding**: The input is transformed into a vector representation
3. **Retrieval**: The vector is compared against stored route vectors
4. **Matching**: The best matching route is selected based on similarity
5. **Response**: The system returns the matched route, enabling appropriate handling

## Router Types

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

- **SemanticRouter**: Uses dense vector embeddings for semantic matching
- **HybridRouter**: Combines both dense and sparse vectors for enhanced accuracy

## Integration Example

```python
from semantic_router import Route, SemanticRouter
from semantic_router.encoders import OpenAIEncoder

# 1. Define routes
weather_route = Route(name="weather", utterances=["What's the weather like?"])
greeting_route = Route(name="greeting", utterances=["Hello there!", "Hi!"])

# 2. Initialize encoder
encoder = OpenAIEncoder()

# 3. Create router with routes
router = SemanticRouter(encoder=encoder, routes=[weather_route, greeting_route])

# 4. Route an incoming query
result = router("What's the forecast for tomorrow?")
print(result.name)  # "weather"
```

## Performance Considerations

- **In-memory vs. Vector DB**: Choose based on scale and latency requirements
- **Encoder selection**: Balance accuracy vs. speed based on use case
- **Batch processing**: Use batch methods for higher throughput
- **Async support**: Available for high-concurrency environments and applications relying
on heavy network use
