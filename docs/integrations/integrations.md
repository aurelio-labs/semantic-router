Semantic Router provides several types of integrations with other AI frameworks and services. These integrations fall into four main categories:

## 1. Encoder Integrations

Encoder integrations allow Semantic Router to use various embedding models for semantic similarity calculations. These integrations power the core routing functionality by converting text into vector representations that can be compared for semantic similarity.

Supported encoder providers include:
- Jina AI
- OpenAI
- Cohere
- Hugging Face
- Google AI
- Mistral
- Voyage
- And more...

Each encoder integration provides access to high-quality embedding models that can be used with both the `SemanticRouter` and `HybridRouter` classes.

## 2. Index Integrations

Index integrations enable efficient storage and retrieval of vector embeddings. These integrations are crucial for scaling semantic routing to handle large numbers of routes and queries.

Supported index providers include:
- Pinecone
- Qdrant
- PostgreSQL (with vector extension)
- Local (in-memory)

Each index integration supports both synchronous and asynchronous operations, making them suitable for various deployment scenarios.

## 3. LLM Integrations

LLM integrations enable dynamic route generation and decision-making using large language models. These integrations are particularly useful for:
- Generating route responses
- Creating dynamic routes based on context
- Handling complex routing logic

Currently supported LLM providers:
- OpenAI
- Azure OpenAI
- Mistral
- Bedrock

## 4. Example Integrations

While not built-in integrations, Semantic Router works well with many popular AI frameworks and libraries. We provide example notebooks and documentation showing how to use Semantic Router with:

- Pydantic AI
- Agent SDKs
- Other vector databases
- And more...

These examples demonstrate how to combine Semantic Router's routing capabilities with other tools to build powerful AI applications. 