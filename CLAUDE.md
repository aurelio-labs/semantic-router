# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Semantic Router is a high-performance decision-making layer for LLMs and agents that uses semantic vector space to make routing decisions instead of waiting for slow LLM generations. It's maintained by Aurelio AI and provides both static and dynamic routing capabilities.

## Development Commands

### Package Management
This project uses `uv` for dependency management. All commands should be prefixed with `uv run`.

### Testing
```bash
# Run all tests with coverage
make test

# Run specific test categories
make test_unit        # Unit tests only
make test_functional  # Functional tests only  
make test_integration # Integration tests only

# Run a single test file
uv run pytest tests/unit/test_route.py -vv

# Run a single test function
uv run pytest tests/unit/test_route.py::test_function_name -vv
```

### Code Quality
```bash
# Run all linting and formatting
make lint

# Auto-fix and format code
make format

# Lint only files changed from main branch
make lint_diff

# Run type checking
uv run mypy semantic_router/
```

### Running Examples
Most examples are in Jupyter notebooks in the `docs/` directory. To run them:
```bash
uv run jupyter notebook docs/00-introduction.ipynb
```

## Architecture Overview

### Core Components

1. **Routes** (`semantic_router/route.py`)
   - Basic unit of routing logic
   - Contains name, utterances, optional function schemas and LLM
   - Supports both static (simple matching) and dynamic (parameter extraction) routes

2. **Routers** (`semantic_router/routers/`)
   - `BaseRouter`: Abstract base with core routing logic, training, and sync
   - `SemanticRouter`: Standard dense embedding-based router
   - `HybridRouter`: Combines dense and sparse embeddings for better accuracy

3. **Encoders** (`semantic_router/encoders/`)
   - `DenseEncoder`: Base class for semantic embeddings (OpenAI, Cohere, HuggingFace, etc.)
   - `SparseEncoder`: Base for keyword-based encodings (BM25, TF-IDF)
   - Supports both sync and async operations
   - Some encoders are asymmetric (different encoding for queries vs documents)

4. **Indexes** (`semantic_router/index/`)
   - `BaseIndex`: Abstract interface for vector storage
   - Implementations: `LocalIndex`, `PineconeIndex`, `PostgresIndex`, `QdrantIndex`
   - `HybridLocalIndex`: Supports both dense and sparse vectors

### Data Flow
```
User Query → Router → Encoder → Embeddings → Index → Similarity Search → Route Selection → Response
```

### Key Patterns
- **Strategy Pattern**: Swappable encoders and indexes
- **Template Method**: BaseRouter defines algorithm, subclasses implement specifics
- **Async Support**: Full async/await support throughout
- **Configuration**: Routes can be imported/exported as JSON/YAML

## Important Considerations

### When Adding New Features
1. Check existing patterns in similar components (e.g., look at other encoders when adding a new encoder)
2. Ensure both sync and async versions are implemented where applicable
3. Add appropriate type hints and follow existing naming conventions
4. Add tests in the appropriate test directory (unit/functional/integration)

### Testing Guidelines
- Unit tests go in `tests/unit/` and test individual components in isolation
- Functional tests go in `tests/functional/` and test component interactions
- Integration tests go in `tests/integration/` and test with real external services
- Mock external API calls in unit tests
- Use `pytest-mock` for mocking

### Common Gotchas
1. Many encoders require API keys set as environment variables (e.g., `OPENAI_API_KEY`)
2. The project supports Python 3.9-3.13, some features are disabled for 3.13+ due to dependency constraints
3. Local models require the `[local]` extra: `pip install "semantic-router[local]"`
4. Hybrid routing requires the `[hybrid]` extra
5. When working with indexes, be aware of synchronization between local and remote states

### Performance Considerations
- Route encoding happens once during initialization
- Query encoding happens on every request
- Use `auto_sync="local"` for better performance when routes don't change frequently
- HybridRouter is more accurate but slightly slower than SemanticRouter