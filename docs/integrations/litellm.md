Semantic Router can use [LiteLLM](https://github.com/BerriAI/litellm) to reach 100+ LLM providers through a single interface, with built-in cost tracking and error handling.

## Installation

LiteLLM is an **optional** dependency. Install it with the `litellm` extra:

```bash
pip install "semantic-router[litellm]"
```

Without it, `semantic_router` imports and every provider-specific encoder works as
normal, but constructing a `LiteLLMEncoder` raises an `ImportError` telling you to
install the extra.

## Overview

When installed, LiteLLM provides:
- Unified interface to multiple embedding providers
- Automatic cost tracking and token counting
- Standardized error handling and retries
- Support for both synchronous and asynchronous operations

## Native Encoders or LiteLLM?

Most providers can be reached two ways, and the choice is a real trade-off.

The **native encoders** below call their provider's own SDK or REST API. They need no extra dependency, but they also get none of LiteLLM's cross-provider machinery — in particular **they do not report cost or token usage**. Every provider in this table is also reachable through LiteLLM, so for these the choice is yours:

| Encoder | Provider docs |
| --- | --- |
| `AzureOpenAIEncoder` | — |
| `BedrockEncoder` | [AWS Bedrock](bedrock.md) |
| `CohereEncoder` | [Cohere](cohere.md) |
| `GoogleEncoder` | [Google notebook](../encoders/google.ipynb) |
| `JinaEncoder` | [Jina](jina.md) |
| `MistralEncoder` | [Mistral](mistral.md) |
| `NimEncoder` | [NVIDIA NIM](nvidia.md) |
| `OpenAIEncoder` | [OpenAI](openai.md) |
| `VoyageEncoder` | [Voyage](voyage.md) |

The remaining encoders — the local, sparse and self-hosted ones such as `BM25Encoder`, `HuggingFaceEncoder`, `LocalEncoder`, `OllamaEncoder` and `TfidfEncoder` — have no LiteLLM route, so this trade-off does not apply to them.

`LiteLLMEncoder` is the only encoder that requires the `litellm` extra, and the only one that gives you cost tracking.

### Reaching the Same Provider Through LiteLLM

If you want cost tracking for one of the providers above, you do not have to give up that provider — configure its credentials for LiteLLM and use `LiteLLMEncoder` with the provider-prefixed model name instead of the native encoder.

For example, Jina embeddings via LiteLLM rather than via `JinaEncoder`:

```python
from semantic_router.encoders import LiteLLMEncoder

# LiteLLM names models as "<provider>/<model>"; Jina's provider id is "jina_ai"
encoder = LiteLLMEncoder(
    name="jina_ai/jina-embeddings-v3",
    api_key="your-jina-api-key",  # or set JINA_AI_API_KEY
    score_threshold=0.4,
)
embeddings = encoder(["your text here"])
```

The same pattern applies to the other providers — the prefixes are `azure/`, `bedrock/`, `cohere/`, `mistral/`, `nvidia_nim/`, `openai/`, `vertex_ai/` and `voyage/`. Note that the model name for `LiteLLMEncoder` always carries this prefix, whereas the native encoders take a bare model name.

Two caveats when switching:

- Embeddings are not guaranteed to be byte-identical between the two routes, so re-index your routes rather than mixing vectors produced by both.
- The native encoders expose provider-specific options the LiteLLM route may not, such as Jina's `task` or NIM's `input_type`.

## Features

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
- And many more see: https://docs.litellm.ai/docs/providers

## Direct LiteLLM Usage

`LiteLLMEncoder` works with any LiteLLM-supported embedding model:

```python
from semantic_router.encoders import LiteLLMEncoder

# names must be "<provider>/<model>" - a bare model name raises a ValueError
encoder = LiteLLMEncoder(
    name="openai/text-embedding-ada-002",
    score_threshold=0.4
)
```

The API key is taken from the `api_key` argument, or from `<PROVIDER>_API_KEY` in the environment — `OPENAI_API_KEY` for the example above.

## Integration with Routers

`LiteLLMEncoder` plugs into a router like any other encoder:

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

`LiteLLMEncoder` reads `<PROVIDER>_API_KEY`, where the provider is the prefix of the model
name:
- `OPENAI_API_KEY` - `openai/...` models
- `COHERE_API_KEY` - `cohere/...` models
- `MISTRAL_API_KEY` - `mistral/...` models
- `VOYAGE_API_KEY` - `voyage/...` models
- `JINA_AI_API_KEY` - `jina_ai/...` models
- `NVIDIA_NIM_API_KEY` - `nvidia_nim/...` models
- And more...

These are LiteLLM's own variable names. A native encoder may accept a different one — for
instance `JinaEncoder` also reads `JINA_API_KEY`.

## Example Usage

```python
from semantic_router.encoders import LiteLLMEncoder
from semantic_router.routers import SemanticRouter
from semantic_router.route import Route

# routed through LiteLLM, so requests are cost-tracked
encoder = LiteLLMEncoder(name="openai/text-embedding-3-small", score_threshold=0.3)

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
