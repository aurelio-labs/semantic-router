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

Each encoder plugs into an `IndexedRouter`: dense encoders as its `encoder`, and sparse encoders (BM25, TF-IDF, Aurelio) as its `sparse_encoder` for hybrid routing. All encoders support both synchronous and asynchronous operations.

## 2. Index Integrations

Index integrations enable efficient storage and retrieval of vector embeddings. These integrations are crucial for scaling semantic routing to handle large numbers of routes and queries.

### Vector Database Integrations

- [Pinecone](pinecone.md) - Serverless or pod-based vector database with namespace support
- [Qdrant](qdrant.md) - High-performance open-source vector database
- [PostgreSQL](postgres.md) - PostgreSQL with pgvector extension (FLAT, IVFFLAT, HNSW)
- **MongoDB Atlas** - Atlas Vector Search, with `MongoDbIndex` (`pip install "semantic-router[mongodb]"`)
- **Local** - In-memory index for development and testing

Each index integration supports:
- Both synchronous and asynchronous operations
- Millions of vectors with efficient similarity search
- Full CRUD operations (create, read, update, delete)
- Route isolation via namespaces or collections

## 3. Decision Model Integrations

A `DecisionRouter` routes by asking a decision model instead of comparing embeddings. `TypeSafeDecisionModel` calls TypeSafe's **Jev**, `NaceDecisionModel` calls nace.ai's **Drex**, and `SystemOneDecisionModel(base_url=...)` calls any other server that speaks the same `/v1/systemone` API, such as **Laya** or **Von**. It needs no encoder and no index. See [decision models](../user-guide/features/decision-models.md), and for background, [Understanding Jev and Where It Fits in the AI Stack](https://jamesbriggs.io/understanding-jev) and [the video](https://youtu.be/P-t4F_UlxkI).

Semantic Router 1.0 no longer wraps LLMs (OpenAI, Mistral, Bedrock and so on) for dynamic routes: route first, then call your LLM with whichever client you already use.

## 4. Framework Integrations

While not built-in integrations, Semantic Router works well with many popular AI frameworks and libraries. We provide example notebooks and documentation showing how to use Semantic Router with:

- [Pydantic AI](pydantic-ai.md) - Type-safe AI agent framework
- [LiteLLM](litellm.md) - One encoder for 100+ embedding providers
- Agent frameworks (LangChain, LlamaIndex, etc.)
- Custom applications

These examples demonstrate how to combine Semantic Router's routing capabilities with other tools to build powerful AI applications.

## Getting Started

To get started with any integration:
1. Install Semantic Router with the appropriate extras (e.g., `pip install "semantic-router[pinecone]"`)
2. Follow the integration-specific guide
3. Refer to the example notebooks in the repository

For detailed information on each integration, click the links above to view the integration-specific documentation. 