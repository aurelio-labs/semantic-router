> **You're reading the docs for Semantic Router 0.x.** Version 1.0 is out: read [this page for v1](https://docs.aurelio.ai/docs/semantic-router/v1/user-guide/changelog), or the [migration guide](https://docs.aurelio.ai/docs/semantic-router/v1/user-guide/guides/migration-to-v1) to upgrade.

### v0.3.0

`0.3.0` brings in the features from pull requests that were already open when `0.2.0` shipped, as a one-off exception. It is now the last feature release of the 0.x line: from here, 0.x gets bug fixes only, and new work goes into 1.x. Releases also follow strict [semantic versioning](https://semver.org) from now on: new features raise the minor version, and fixes raise the patch version. These are the changes since `0.2.0`, and none of them is breaking.

#### Features

- **`RedisIndex`** stores routes in Redis with RediSearch vector search, sync and async. It needs a server with the search module, such as Redis Stack, Redis 8 or Redis Cloud. Install `semantic-router[redis]`, and see the [Redis guide](https://github.com/aurelio-labs/semantic-router/blob/v0/docs/integrations/redis.md).
- **`OpenAICompatibleLLM`** calls any OpenAI-compatible chat completions API. Pass its `base_url`, and either `api_key` or the name of the environment variable holding the key (`api_key_var_name`).
- **Presets** built on `OpenAICompatibleLLM`, each with its own default `base_url`:

| LLM | Key variable | Default model |
|---|---|---|
| `RequestyLLM` | `REQUESTY_API_KEY` | `openai/gpt-4o-mini` |
| `OrcaRouterLLM` | `ORCAROUTER_API_KEY` | `orcarouter/auto` |
| `AtlasCloudLLM` | `ATLASCLOUD_API_KEY` | `deepseek-ai/deepseek-v4-flash` |
| `MiniMaxLLM` | `MINIMAX_API_KEY` | `MiniMax-M3.1-Flash` |

A preset reads its model from `<PROVIDER>_CHAT_MODEL_NAME` (`REQUESTY_CHAT_MODEL_NAME`, for example) when you don't pass `name`. `MiniMaxLLM` strips MiniMax-M3's `<think>` blocks from its replies, and its default `max_tokens` is 4096 to leave room for thinking.

#### Limits

**`RedisIndex` on Redis 8 with search workers.** With the query engine's worker threads on, Redis 8 can return incomplete results while it re-indexes a key. Redis 8.10 turns them on by default (`search-workers 16`); 8.0 and 8.2 leave them off. A query that runs while the same keys are being rewritten can miss routes, and so can the scans behind `sync` and `delete`. `RedisIndex` waits for a new or reopened index to finish indexing the keys already in the database (up to `index_ready_timeout`, 60 seconds by default), but it can't guard against concurrent updates. If routes are updated while the router serves queries, use Redis Stack 7.4 or set `search-workers 0`. On Redis 8.2 with workers on, a search can also return a record without its score, and `RedisIndex` raises an error rather than return incomplete results.

`OpenAICompatibleLLM` and its presets have two limits in 0.x:

- Dynamic routes extract function inputs only when called synchronously. `BaseLLM` has no async extraction, so an async call returns the matched route with `function_call=None` and logs an error.
- A router config file saves only each LLM's class and model name. A router using `OpenAICompatibleLLM` can't be reloaded from one, because it needs a `base_url`. A preset reloads with its default `base_url` and reads its key from its environment variable, so a custom `base_url` or a key passed in code is lost.

### v0.2.0

`0.2.0` is the last feature release of the 0.x line: from here, 0.x gets bug fixes only, and new work goes into 1.x (see the [migration guide](https://docs.aurelio.ai/docs/semantic-router/v1/user-guide/guides/migration-to-v1)). These are the changes since `0.1.16`.

#### Breaking changes

- **Python 3.10 or later.** Python 3.9 is no longer supported, and dependency floors are raised to patched versions.
- **`litellm` is optional.** Install `semantic-router[litellm]` to use `LiteLLMEncoder`. `CohereEncoder`, `MistralEncoder`, `VoyageEncoder`, `JinaEncoder` and `NimEncoder` call their providers directly and no longer need LiteLLM.
- **`BM25Encoder` scores change.** The ATIRE BM25 formula is corrected: document length normalisation, the `(k1 + 1)` numerator, query IDF weighted by term frequency, and query weights scaled to `[0, 1]`. Both the document and the query weights change, so vectors stored by 0.1.x don't match queries encoded by 0.2.0: re-encode your documents and rebuild any sparse or hybrid index, then re-check the thresholds you tuned.

#### Feature

- `FastEmbedEncoder` supports async calls.

#### Fixes

- `OpenAIEncoder` rejects an empty list, an empty string, non-string items and a bare `str` with a clear `ValueError` before calling the API. It used to retry the API's 400 before failing. Other input is sent unchanged.
- `LocalIndex` returns no results for a route filter that matches no routes, like the other indexes, instead of raising.
- `LocalIndex.add` checks that `metadata_list` is as long as the other inputs.
- `PostgresIndex.aadd` uses deterministic record ids, as `add` does, so adding a stored utterance again doesn't duplicate it.
- `PostgresIndex` no longer fails when two connections create the `vector` extension at the same moment.
- `PineconeIndex` retries reading the stats of a freshly created index while its host comes up.
- `QdrantIndex`: a failed scroll raises instead of reading as an empty index.
- `NimEncoder` embeds queries with `input_type="query"` instead of `"passage"`.
- `CLIPEncoder` and `VitEncoder` work with transformers 5.

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
