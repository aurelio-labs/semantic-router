Semantic Router uses LiteLLM as the foundation for many encoder integrations, providing unified access to 100+ LLM providers with built-in cost tracking and error handling.

## Overview

LiteLLM is integrated throughout Semantic Router to provide:
- Unified interface to multiple embedding providers
- Automatic cost tracking and token counting
- Standardized error handling and retries
- Support for both synchronous and asynchronous operations

## LiteLLM-Based Encoders

Many Semantic Router encoders are built on `LiteLLMEncoder`:
- `CohereEncoder` - Cohere embeddings
- `VoyageEncoder` - Voyage AI embeddings
- `MistralEncoder` - Mistral AI embeddings
- `NimEncoder` - NVIDIA NIM embeddings
- `JinaEncoder` - Jina AI embeddings
- Custom LiteLLM encoders

## Features

### Unified Interface

All LiteLLM-based encoders share a consistent interface:

```python
from semantic_router.encoders import CohereEncoder

encoder = CohereEncoder(name="embed-english-v3.0", score_threshold=0.3)
embeddings = encoder(["your text here"])
```

### Cost Tracking

Automatic cost tracking for all supported providers:
- Per-request token counting
- Model-specific pricing
- Total cost calculation
- Usage logging

### Error Handling

Built-in error handling and retries:
- Automatic retry on rate limits
- Fallback to alternative models
- Clear error messages
- Timeout management

### Provider Support

LiteLLM provides access to 100+ providers including:
- OpenAI, Azure OpenAI
- Anthropic, Google
- Cohere, Mistral
- AWS Bedrock, Vertex AI
- And many more

## Direct LiteLLM Usage

You can also use `LiteLLMEncoder` directly for any LiteLLM-supported model:

```python
from semantic_router.encoders import LiteLLMEncoder

# Use any LiteLLM-supported model
encoder = LiteLLMEncoder(
    name="text-embedding-ada-002",  # or any litellm model name
    score_threshold=0.4
)
```

## Integration with Routers

All LiteLLM-based encoders work seamlessly with Semantic Router:

```python
from semantic_router.routers import SemanticRouter
from semantic_router.route import Route

routes = [
    Route(name="support", utterances=["help", "assist"]),
    Route(name="sales", utterances=["buy", "purchase"])
]

router = SemanticRouter(
    encoder=encoder,
    routes=routes,
    auto_sync="local"
)
```

## Best Practices

1. **Model Selection**: Choose the appropriate model for your use case and budget
2. **Cost Monitoring**: Monitor LiteLLM's cost tracking output for budget control
3. **Rate Limits**: LiteLLM handles rate limits automatically, but consider batch sizing
4. **API Keys**: Set provider-specific API keys via environment variables
5. **Logging**: Enable LiteLLM logging for debugging and monitoring

## Advantages

- **Provider Flexibility**: Easy to switch between providers
- **Cost Transparency**: Clear visibility into API costs
- **Reliability**: Built-in retry logic and error handling
- **Future-Proof**: New providers added to LiteLLM work automatically
- **Consistency**: Same interface across all providers

## Environment Variables

LiteLLM respects standard provider environment variables:
- `OPENAI_API_KEY` - OpenAI models
- `COHERE_API_KEY` - Cohere models
- `MISTRAL_API_KEY` - Mistral models
- `VOYAGE_API_KEY` - Voyage models
- And more...

## Example Usage

```python
from semantic_router.encoders import CohereEncoder
from semantic_router.routers import SemanticRouter
from semantic_router.route import Route

# LiteLLM-based encoder with cost tracking
encoder = CohereEncoder(name="embed-english-v3.0", score_threshold=0.3)

routes = [
    Route(name="greeting", utterances=["hello", "hi"]),
    Route(name="goodbye", utterances=["bye", "goodbye"])
]

router = SemanticRouter(encoder=encoder, routes=routes, auto_sync="local")

# LiteLLM automatically tracks costs
result = router("hi there")
print(result.name)  # -> greeting
```

## Learn More

- [LiteLLM Documentation](https://docs.litellm.ai/)
- [Supported Models](https://docs.litellm.ai/docs/providers)
- [Cost Tracking](https://docs.litellm.ai/docs/completion/cost_tracking)
