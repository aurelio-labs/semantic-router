Semantic Router provides several types of integrations with other AI frameworks and services. These integrations fall into four main categories:

## 1. Encoder Integrations

Encoder integrations allow Semantic Router to use various embedding models for semantic similarity calculations. These integrations power the core routing functionality by converting text into vector representations that can be compared for semantic similarity.

### Cloud Encoder Providers

- [Jina AI](jina.md) - High-quality embeddings via Jina's API
- [OpenAI](openai.md) - OpenAI's text-embedding models
- [Cohere](cohere.md) - Cohere's embedding models
- [Voyage](voyage.md) - Voyage AI embeddings
- [Mistral](mistral.md) - Mistral AI embeddings
- [NVIDIA](nvidia.md) - NVIDIA NIM embedding models
- [Bedrock](bedrock.md) - AWS Bedrock embedding models (Titan, Cohere)
- [Aurelio](aurelio.md) - Aurelio's semantic embeddings

### Local and Self-Hosted Encoders

- [Local](local.md) - Run sentence-transformers models locally (no API required)
- [Ollama](ollama.md) - Use Ollama-hosted embedding models
- Hugging Face - Self-hosted or Inference API models
- FastEmbed - Fast local embedding models

Each encoder integration provides access to high-quality embedding models that can be used with both the `SemanticRouter` and `HybridRouter` classes. All encoders support both synchronous and asynchronous operations.

## 2. Index Integrations

Index integrations enable efficient storage and retrieval of vector embeddings. These integrations are crucial for scaling semantic routing to handle large numbers of routes and queries.

### Vector Database Integrations

- [Pinecone](pinecone.md) - Serverless or pod-based vector database with namespace support
- [Qdrant](qdrant.md) - High-performance open-source vector database
- [PostgreSQL](postgres.md) - PostgreSQL with pgvector extension (FLAT, IVFFLAT, HNSW)
- **Local** - In-memory index for development and testing

Each index integration supports:
- Both synchronous and asynchronous operations
- Millions of vectors with efficient similarity search
- Full CRUD operations (create, read, update, delete)
- Route isolation via namespaces or collections

## 3. LLM Integrations

LLM integrations enable dynamic route generation and decision-making using large language models. These integrations are particularly useful for:
- Generating route responses
- Creating dynamic routes based on context
- Handling complex routing logic
- Parameter extraction from queries

Currently supported LLM providers:
- **OpenAI** - GPT-4, GPT-3.5, and other OpenAI models
- **Azure OpenAI** - OpenAI models via Azure
- **Mistral** - Mistral AI language models
- **Bedrock** - AWS Bedrock LLMs (Claude, Titan, etc.)
- **LiteLLM** - Unified interface to 100+ LLM providers

## 4. Framework Integrations

While not built-in integrations, Semantic Router works well with many popular AI frameworks and libraries. We provide example notebooks and documentation showing how to use Semantic Router with:

- [Pydantic AI](pydantic-ai.md) - Type-safe AI agent framework
- [LiteLLM](litellm.md) - Unified LLM interface
- Agent frameworks (LangChain, LlamaIndex, etc.)
- Custom applications

These examples demonstrate how to combine Semantic Router's routing capabilities with other tools to build powerful AI applications.

## Getting Started

To get started with any integration:
1. Install Semantic Router with the appropriate extras (e.g., `pip install "semantic-router[pinecone]"`)
2. Follow the integration-specific guide
3. Refer to the example notebooks in the repository

For detailed information on each integration, click the links above to view the integration-specific documentation. 