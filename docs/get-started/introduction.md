Semantic Router is a superfast decision-making layer for LLMs and agents. Instead of waiting for slow LLM generations to make tool-use decisions, it uses semantic vector space to route requests based on meaning.

## What is Semantic Router?

Semantic Router enables:

- **Faster decisions**: Make routing decisions in milliseconds rather than seconds
- **Lower costs**: Avoid expensive LLM inference for simple routing tasks
- **Better control**: Direct conversations, queries, and agent actions with precision
- **Full flexibility**: Use cloud APIs or run everything locally

## Key Features

- **Simple API**: Set up routes with just a few lines of code
- **Dynamic routes**: Generate parameters and trigger function calls
- **Multiple integrations**: Works with Cohere, OpenAI, Hugging Face, FastEmbed, and more
- **Vector store support**: Integrates with Pinecone and Qdrant for persistence
- **Multi-modal capabilities**: Route based on image content, not just text
- **Local execution**: Run entirely on your machine with no API dependencies

## Version 0.1 Released

Semantic Router v0.1 is now available! If you're migrating from an earlier version, please see our [migration guide](../user-guide/guides/migration-to-v1).

## Getting Started

For a quick introduction to using Semantic Router, check out our [quickstart guide](quickstart).

## Execution Options

Semantic Router supports multiple execution modes:

- **Cloud-based**: Using OpenAI, Cohere, or other API-based embeddings
- **Hybrid**: Combining local embeddings with API-based LLMs
- **Fully local**: Run everything on your machine with models like Llama and Mistral

## Resources

- [Documentation](https://docs.aurelio.ai/semantic-router/index.html)
- [GitHub Repository](https://github.com/aurelio-labs/semantic-router)
- [Online Course](https://www.aurelio.ai/course/semantic-router) 