Semantic Router integrates with MiniMax's language models through the `MiniMaxLLM` class, for dynamic routes with parameter extraction.

## Overview

MiniMax serves its language models through an OpenAI-compatible API at `https://api.minimax.io/v1`, so `MiniMaxLLM` is a preset of `OpenAICompatibleLLM`: a default base URL, the `MINIMAX_API_KEY` environment variable and a default model.

## Getting Started

### Prerequisites

1. MiniMax API key from [https://platform.minimax.io](https://platform.minimax.io). A key from MiniMax's China platform ([https://platform.minimaxi.com](https://platform.minimaxi.com)) needs `base_url="https://api.minimaxi.com/v1"`.
2. Semantic Router installed

### Installation

```bash
pip install "semantic-router"
```

### Basic Usage

```python
import os
from semantic_router.llms import MiniMaxLLM

os.environ["MINIMAX_API_KEY"] = "your-api-key"

llm = MiniMaxLLM()
# or with a custom model
llm = MiniMaxLLM(name="MiniMax-M3")
```

## Features

### Supported Models

- `MiniMax-M3.1-Flash` (default): always thinks, and returns its thinking in a separate field
- `MiniMax-M3`: thinks by default, inside `<think>` tags in the reply

Any model name MiniMax's OpenAI-compatible API accepts works through `name=` or `MINIMAX_CHAT_MODEL_NAME`. MiniMax's M2.5 and M2.7 models have been unavailable to new users since 19 October 2026 and are scheduled to retire on 18 November 2026; existing pay-as-you-go users can keep using them until then (see [MiniMax's model list](https://platform.minimax.io/docs/faq/models)).

### Asynchronous Support

```python
from semantic_router.schema import Message

messages = [Message(role="user", content="Hi, how are you?")]

# Synchronous
reply = llm(messages=messages)

# Asynchronous
reply = await llm.acall(messages=messages)
```

### Thinking

MiniMax models think before they answer. `MiniMaxLLM` returns only the answer: it strips `<think>...</think>` blocks from the reply and ignores thinking returned in its own field.

Thinking counts towards `max_tokens`, so `MiniMaxLLM` defaults to `max_tokens=4096` rather than the 200 the other LLMs use. If a reply still runs out of tokens while the model is thinking, `MiniMaxLLM` raises an error rather than returning the thinking; raise `max_tokens` if that happens.

### Temperature

Current MiniMax models ignore `temperature`. `MiniMaxLLM` still clamps it to (0.0, 1.0], the range older MiniMax models required.

## Dynamic Routes with MiniMax LLM

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

## Limits

`MiniMaxLLM` inherits two limits from semantic-router 0.x:

- Dynamic routes extract function inputs only when called synchronously. `BaseLLM` has no async extraction, so an async call returns the matched route with `function_call=None` and logs an error. `MiniMaxLLM.acall` itself works; routes just don't use it for extraction.
- A router config file saves only the LLM's class and model name. Reloading one uses the default `base_url` and `MINIMAX_API_KEY`, so a custom `base_url` (such as the China platform's) or a key passed in code is lost.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `MINIMAX_API_KEY` | MiniMax API key (required) |
| `MINIMAX_CHAT_MODEL_NAME` | Default chat model name |
