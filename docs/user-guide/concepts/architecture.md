---
title: Architecture
description: The router contract, the router types, calibration, and how a query flows through them.
---

## System overview

Every router does one thing: given a query, it scores every route. Everything after that is shared. The base class turns the scores into confidences (calibrated, if you've fitted it), applies the thresholds, and hands back a `RouteChoice`.

```mermaid
graph TD
    Q[Query] --> S["Score every route<br/>(each router type, its own way)"]
    S --> C["Calibrate<br/>raw score → probability"]
    C --> T["Apply thresholds<br/>call, route, router"]
    T --> R[RouteChoice]
```

Routers differ only in how they score:

```mermaid
classDiagram
    class BaseRouter {
        +routes
        +confidence_threshold
        +calibrator
        +__call__(query, limit, threshold) RouteChoice
        +acall(query, limit, threshold) RouteChoice
        +fit(X, y, method) FitReport
        +evaluate(X, y) float
        +add(routes)
        +remove(name)
        #_scores(query)* dict
    }
    BaseRouter <|-- IndexedRouter
    BaseRouter <|-- DecisionRouter
    BaseRouter <|-- Chain
    BaseRouter <|-- Merge
```

- **`IndexedRouter`** scores by embedding similarity: an encoder plus an index.
- **`DecisionRouter`** scores by asking a decision model, with one HTTP call.
- **`Chain`** and **`Merge`** score by asking other routers.

A router of your own is one more subclass: implement `_scores` and the rest comes free. See [routers](../components/routers#build-your-own-router).

## Core components

### Routes

A route is one decision a router can make. The same `Route` works with every router type.

```mermaid
classDiagram
    class Route {
        +name: str
        +utterances: List[str]
        +description: Optional[str]
        +confidence_threshold: Optional[float]
        +metadata: Dict
    }
```

`IndexedRouter` embeds the `utterances`; `DecisionRouter` reads the `description`. See [routes](../components/routes).

### Encoders

Encoders map inputs into semantic space. `IndexedRouter` uses them; `DecisionRouter` doesn't.

```mermaid
classDiagram
    class DenseEncoder
    class SparseEncoder
    DenseEncoder <|-- OpenAIEncoder
    DenseEncoder <|-- FastEmbedEncoder
    DenseEncoder <|-- HuggingFaceEncoder
    DenseEncoder <|-- CLIPEncoder
    SparseEncoder <|-- BM25Encoder
    SparseEncoder <|-- TfidfEncoder
    SparseEncoder <|-- AurelioSparseEncoder
```

- **Dense encoders** produce continuous vectors (OpenAI, FastEmbed, Hugging Face, ...).
- **Sparse encoders** produce mostly-zero vectors keyed on terms (BM25, TF-IDF, ...). Add one to an `IndexedRouter` for hybrid search.
- **Asymmetric encoders** embed queries and documents differently; the router calls the right side for each.

See [encoders](../components/encoders).

### Indexes

Indexes store utterance vectors and search them. The index is the single source of truth for an `IndexedRouter`'s routes: building a router writes to it, and `router.routes` is a cached view of it, which `refresh()` re-reads.

```mermaid
classDiagram
    class BaseIndex {
        +add(embeddings, routes, utterances)
        +aquery(vector, top_k) matches
        +adelete(route_name)
        +get_routes() List[Route]
    }
    BaseIndex <|-- LocalIndex
    LocalIndex <|-- HybridLocalIndex
    BaseIndex <|-- PineconeIndex
    BaseIndex <|-- QdrantIndex
    BaseIndex <|-- PostgresIndex
```

See [indexes](../components/indexes).

### Calibrator

A `Calibrator` maps a raw score to the probability that the top route is right. `fit()` builds one from labelled queries, with Platt scaling or isotonic regression. Both are monotone, so calibration never changes which route ranks first, only the number attached to it. See [calibration](../features/calibration).

## Data flow

An `IndexedRouter` call:

```mermaid
sequenceDiagram
    participant User
    participant Router as IndexedRouter
    participant Encoder
    participant Index
    User->>Router: router("query")
    Router->>Encoder: embed the query
    Encoder->>Router: vector
    Router->>Index: top_k nearest utterances
    Index->>Router: (route, score) hits
    Router->>Router: aggregate per route, calibrate, apply thresholds
    Router->>User: RouteChoice(name, confidence, calibrated)
```

A `Chain` of an `IndexedRouter` and a `DecisionRouter`:

```mermaid
sequenceDiagram
    participant User
    participant Chain
    participant Fast as IndexedRouter
    participant Smart as DecisionRouter
    User->>Chain: chain("query")
    Chain->>Fast: score
    Fast->>Chain: confidences
    alt top confidence clears its bar
        Chain->>User: RouteChoice from Fast
    else not sure
        Chain->>Smart: score
        Smart->>Chain: probabilities
        Chain->>User: RouteChoice from Smart
    end
```

## Sync and async

The router logic is async all the way down. Every sync method (`__call__`, `fit`, `add`, ...) runs its async twin on one background event loop that all routers share. That keeps HTTP connections pooled between calls, so repeated calls to a remote encoder, index or decision model don't pay to reconnect. It also means sync calls work inside notebooks, which already run a loop.

In async code, use the `a`-methods with `await`.

## Performance notes

- **Encoding dominates.** For an `IndexedRouter`, nearly all of a call's time is embedding the query. A local encoder or a fast embedding API matters more than the index.
- **Batch where you can.** `fit()` and `evaluate()` embed all their queries in one encoder call.
- **A Chain pays only when it must.** Confident queries stop at the first router; only ambiguous ones pay for the decision model too.
- **Warm up remote models.** The first call to a decision model can be much slower than the rest. Make one call at startup.
