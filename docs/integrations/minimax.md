Semantic Router integrates with MiniMax's embedding and language models through the `MiniMaxEncoder` and `MiniMaxLLM` classes. This integration uses MiniMax's OpenAI-compatible API for high-quality embeddings and chat completions.

## Overview

MiniMax provides both embedding and language models accessible via an OpenAI-compatible API at `https://api.minimax.io/v1`. The `MiniMaxEncoder` generates dense embeddings for semantic routing, while `MiniMaxLLM` enables dynamic routes with parameter extraction.

## Getting Started

### Prerequisites

1. MiniMax API key from [https://platform.minimaxi.com/](https://platform.minimaxi.com/)
2. Semantic Router installed

### Installation

```bash
pip install "semantic-router"
```

### Basic Usage

#### Encoder

```python
import os
from semantic_router.encoders import MiniMaxEncoder

os.environ["MINIMAX_API_KEY"] = "your-api-key"

encoder = MiniMaxEncoder()
# or with a custom model
encoder = MiniMaxEncoder(name="embo-01", score_threshold=0.3)
```

#### LLM

```python
from semantic_router.llms import MiniMaxLLM

llm = MiniMaxLLM()
# or with a custom model
llm = MiniMaxLLM(name="MiniMax-M2.5-highspeed")
```

## Features

### Supported Models

**Embedding Models:**
- `embo-01` — 1536-dimensional embeddings (default)

**Language Models:**
- `MiniMax-M2.5` — Latest flagship model (default)
- `MiniMax-M2.5-highspeed` — Fast inference variant

### Asynchronous Support

Full async/await support for both encoder and LLM:

```python
# Synchronous
embeddings = encoder(["your text here"])

# Asynchronous
embeddings = await encoder.acall(["your text here"])
```

### Temperature Clamping

MiniMax requires temperature in the range (0.0, 1.0]. The `MiniMaxLLM` automatically clamps the temperature to this range.

### Think-Tag Stripping

MiniMax models may include `<think>...</think>` tags in responses. The `MiniMaxLLM` automatically strips these to return clean output.

## Integration with Routers

### Semantic Router with MiniMax Encoder

```python
import os
from semantic_router import Route
from semantic_router.encoders import MiniMaxEncoder
from semantic_router.routers import SemanticRouter

os.environ["MINIMAX_API_KEY"] = "your-api-key"

encoder = MiniMaxEncoder()

routes = [
    Route(
        name="greeting",
        utterances=["hello", "hi there", "good morning"],
    ),
    Route(
        name="farewell",
        utterances=["goodbye", "see you later", "bye"],
    ),
]

router = SemanticRouter(encoder=encoder, routes=routes, auto_sync="local")
result = router("hey, how are you?")
print(result.name)  # "greeting"
```

### Dynamic Routes with MiniMax LLM

```python
from semantic_router import Route
from semantic_router.llms import MiniMaxLLM

llm = MiniMaxLLM()

weather_route = Route(
    name="weather",
    utterances=[
        "what's the weather like",
        "tell me the forecast",
        "is it going to rain",
    ],
    function_schemas=[{
        "name": "get_weather",
        "description": "Get weather for a location",
        "signature": "(location: str)",
        "output": "<class 'str'>",
    }],
    llm=llm,
)
```

## AutoEncoder Support

MiniMax is also available through the `AutoEncoder` factory:

```python
from semantic_router.encoders import AutoEncoder

encoder = AutoEncoder(type="minimax", name="embo-01")
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `MINIMAX_API_KEY` | MiniMax API key (required) |
| `MINIMAX_MODEL_NAME` | Default embedding model name |
| `MINIMAX_CHAT_MODEL_NAME` | Default chat model name |
