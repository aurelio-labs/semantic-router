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