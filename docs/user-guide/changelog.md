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

The `0.1.9` update focuses on improving support for our local deployment options. We have standardized the `PostgresIndex` to bring it in line with other index options and prepare it for future feature releases. For `local` extras (inclusing `transformers` and `llama_cpp` support) and `postgres` extras we have resolved issues making those extras unusable with Python 3.13.

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
