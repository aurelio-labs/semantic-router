---
title: Changelog
description: What changed in each Semantic Router release.
---

### v1.0.1

`1.0.1` makes the decision model a component you pass to `DecisionRouter`, the way you pass an encoder to `IndexedRouter`. It changes `DecisionRouter`'s arguments, so code and files written for `1.0.0` need a small edit.

#### Breaking: decision models are a component

- `DecisionRouter(model=TypeSafeDecisionModel(), routes=routes)` replaces `DecisionRouter(routes=routes)`. The connection settings move to the model: `api_key`, `base_url`, `timeout`, `retries` and `concurrency`, and `model` becomes the model's `name`.
- `SystemOneDecisionModel(name=..., base_url=...)` talks to any `/v1/systemone` server (Laya, Von or your own) and sends a key only if you pass one. `TypeSafeDecisionModel` is the same with TypeSafe's defaults, and reads `JEV_API_KEY`.
- Subclass `DecisionModel` and implement `aask` to route with another kind of model.
- `choice`, `noul` and `score` move to `semantic_router.decision_models`.
- Files save the model as a block, `model: {type: typesafe, name: jev-latest}`. A `1.0.0` file, which wrote `model: jev-latest`, fails to load with an error that shows the new form.

### v1.0.0

The `1.0.0` release rebuilds the routing layer. A router is now anything that scores every route; the base class turns scores into calibrated confidence, and routers compose. It's a breaking release: the [migration guide](guides/migration-to-v1) covers every change, with before and after code.

#### Feature: Router types

- `IndexedRouter` replaces `SemanticRouter`. Passing a `sparse_encoder` makes it hybrid, which replaces `HybridRouter`. The hybrid `alpha` (the weight on the dense score) defaults to 0.7, not 0.3, and now weights the dense side on `HybridLocalIndex` too.
- `DecisionRouter` routes with a decision model over TypeSafe's `/v1/systemone` API (Jev, Laya, Von), and `aask()` answers several `choice`, `noul` and `score` questions about one message in a single call.
- `Chain` falls through from a fast router to a slower one when the fast one isn't confident. `Merge` takes a weighted average of several routers.
- Subclass `BaseRouter` and implement one async method, `_scores`, to build your own router.
- `DecisionRouter(concurrency=8)` sets how many requests a batch keeps in flight.

#### Feature: Calibration

- `fit()` learns a map from raw score to the probability that the top route is right (Platt or isotonic), and returns a `FitReport` with accuracy and calibration error. It replaces the random threshold search.
- `confidence_threshold` replaces `score_threshold`, on the router and on each route. A per-call `threshold=` overrides both.
- `RouteChoice` is now `name`, `confidence` and `calibrated`.

#### Feature: The index is the source of truth

- `router.routes` is a cached view of the index, filled on build and after the router's own `add`/`remove`. `refresh()` / `arefresh()` re-read it, and `refresh_interval` re-reads it during calls once it's that many seconds old. A query always searches the index itself.
- An `IndexedRouter` can open an index that 0.x filled: `IndexedRouter(encoder=..., index=...)` with no routes reads them back.

#### Feature: Config files

- `router.to_yaml(path=)`, `router.to_json(path=)` and `load_router(path=)` save and load any router type, keyed on `router_type`, with the calibrator and without secrets.

#### Removed

- Index sync: `auto_sync`, `sync()`, `is_synced()`, `get_utterance_diff()`, and the hash, lock and config records. The index is the single source of truth.
- Dynamic routes and the `semantic_router.llms` package, `function_schemas`, `RouteChoice.function_call` and `similarity_score`.
- `route_filter`, `vector=` and `simulate_static=` on router calls.
- The default `OpenAIEncoder()`: the encoder is now required.
- `score_threshold` on encoders, and `llama-cpp-python` from the `[local]` extra.
- Default thresholds: with no `confidence_threshold`, the best route always wins.

---

### v0.1.12

The `0.1.12` release brings Pinecone v7 support and AWS Bedrock boto3 client integration, improving compatibility and expanding cloud deployment options.

#### Feature: Pinecone v7 Upgrade

Semantic Router now supports Pinecone Python SDK v7, providing:
- Improved data plane performance and reliability
- Enhanced async support for high-throughput applications
- Simplified API structure for better developer experience
- Better error handling and retry logic

The upgrade is backward compatible - existing code continues to work without changes. For shared index deployments, set `PINECONE_INDEX_NAME` to reuse existing indexes and avoid quota limits. Semantic Router automatically isolates data using namespaces.

#### Feature: Bedrock boto3 Client Support

Added support for custom boto3 clients in the `BedrockEncoder`, enabling:
- Advanced AWS credential management (IAM roles, cross-account access)
- Custom retry policies and timeout configurations
- VPC endpoint support for private deployments
- Better integration with existing AWS infrastructure

This provides greater flexibility for enterprise AWS deployments while maintaining backward compatibility with the existing credential-based approach.

#### Chore: Relaxed OpenAI SDK Dependency

Updated OpenAI SDK dependency constraints from `<2.0.0` to `<3.0.0`, allowing compatibility with OpenAI SDK v2.x releases. This provides access to newer OpenAI features and improvements while maintaining stability through the v2 major version.

---

### v0.1.11

The `0.1.11` release introduces new encoder options for local and self-hosted deployments, alongside sparse encoding improvements and CI/CD enhancements.

#### Feature: Local Encoder

Added `LocalEncoder` for fully local embedding generation using sentence-transformers:
- No API keys or internet connection required
- Automatic device selection (CUDA, MPS, CPU)
- Support for any sentence-transformers model
- Privacy-first design - all data stays on your machine

Perfect for offline deployments, privacy-sensitive applications, or development environments.

#### Feature: Local Sparse Encoder 

Added sparse encoder support using sentence transformers with `LocalSparseEncoder`.
- Same offline advantages as the Local Encoder above.
- Better handling of sentence boundaries
- Improved BM25 and TF-IDF implementations
- Enhanced compatibility with `HybridRouter`
#### Feature: Ollama Encoder

Introduced `OllamaEncoder` for using Ollama-hosted embedding models:
- Works with any Ollama embedding model (nomic-embed-text, mxbai-embed-large, etc.)
- Full control over model versions and hosting
- Low latency with local Ollama instances
- Both sync and async support

Ideal for organizations wanting to self-host embedding models while maintaining API-like convenience.


#### Feature: Dagger CI

Implemented Dagger for CI/CD pipeline, providing:
- Faster, more reliable builds
- Better caching and parallelization
- Consistent local and remote execution
- Improved developer experience

#### Chore: Torch Dependency Optimization

Removed torch from main dependencies to reduce installation size. PyTorch is now only installed when needed via `semantic-router[local]` extras, reducing default installation from ~2GB to <100MB.

---

### v0.1.10

The `0.1.10` release was primarily focused on expanding async support for `QdrantIndex`, `PostgresIndex`, and `HybridRouter`, alongside many synchronization and testing improvements.

#### Feature: Expanded Async Support

- **QdrantIndex**: Async methods have been brought inline with our other indexes, ensuring consistent behavior.
- **PostgresIndex**: Async methods have been added to the `PostgresIndex` for improved performance in async environments.
- **HybridRouter**: Async support for the `HybridRouter` is now aligned with the `SemanticRouter`, providing a more consistent experience.

#### Fixes and Optimizations

- **LocalIndex Bug Fix**: Added a `metadata` attribute to the local index. This fixes a bug where `LocalIndex` embeddings would always be recomputed, as reported in [issue #585](https://github.com/aurelio-labs/semantic-router/issues/585).
- Various other bug fixes and optimizations have been included in this release.
- The `urllib3` library has been upgraded.
- Test compatibility and synchronization have been optimized.

---

### v0.1.9

The `0.1.9` update focuses on improving support for our local deployment options. We have standardized the `PostgresIndex` to bring it in line with other index options and prepare it for future feature releases. For `local` extras (including `transformers` and `llama_cpp` support) and `postgres` extras we have resolved issues making those extras unusable with Python 3.13.

Continue reading below for more detail.

#### Feature: Improvements to `postgres` Support

- **Upgrade from psycopg2 to v3 (psycopg)**: We've upgraded our PostgreSQL driver from psycopg2 to the newer psycopg v3, which provides better performance and modern Python compatibility.

- **Standardization of sync methods**: The `PostgresIndex` class now has standardized synchronous methods that integrate seamlessly into our standard pytest test suite, ensuring better testing coverage and reliability.

- **Addition of IndexType support**: We've introduced `IndexType` which includes `FLAT`, `IVFFLAT`, and `HNSW` index types. Both `FLAT` and `IVFFLAT` map to the `IVFFLAT` index in pgvector (pgvector does not currently support `FLAT` and so `IVFFLAT` is the closest approximation - but our other indexes do). `FLAT` is now the default index type. We recommend `HNSW` for high-scale use cases with many millions of utterances due to its higher memory requirements and complexity.

#### Feature: Configurable Logging

- **Environment variable log level control**: Logger now supports configurable log levels via environment variables. Set `SEMANTIC_ROUTER_LOG_LEVEL` or `LOG_LEVEL` to control logging verbosity (e.g., `DEBUG`, `INFO`, `WARNING`, `ERROR`). Defaults to `INFO` if not specified.

#### Fix: Local Installation for Python 3.13 and up

Python 3.13 and up had originally been incompatible with our local installations due to the lack of compatability with PyTorch and the (then) new version of Python. We have now added the following version support for `local` libraries:

```python
local = [
    "torch>=2.6.0 ; python_version >= '3.13'",
    "transformers>=4.50.0 ; python_version >= '3.13'",
    "tokenizer>=0.20.2 ; python_version >= '3.13'",
    "llama-cpp-python>=0.3.0 ; python_version >= '3.13'",
]
```

#### Fix: Consistent Index Length Behavior

- **Standardized `__len__` method across all index implementations**: All index types now consistently return `0` when uninitialized, preventing potential `AttributeError` exceptions that could occur in `PineconeIndex` and `QdrantIndex` when checking the length of uninitialized indexes.

#### Chore: Broader and More Useful Tests

We have broken our tests apart into strict unit and integration test directories. Now, when incoming PRs are raised we will no longer trigger integration tests that require API keys to successfully run. To ensure we're still covering all components of the library we have broadened our testing suite to extensively test `LocalIndex`, `PineconeIndex` (via Pinecone Local), `PostgresIndex`, and `QdrantIndex` within those unit tests.
