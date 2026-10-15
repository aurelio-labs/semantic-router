---
title: Local execution
description: Run routing entirely on your own machine, with local encoders and an in-memory index.
---

There are plenty of reasons to keep routing on your own hardware: cost, latency, privacy, compliance, or just working offline. An `IndexedRouter` with a local encoder and a local index makes no network calls at all once the model is downloaded.

## Pick a local encoder

| Encoder | Install | Good for |
|---|---|---|
| `FastEmbedEncoder` | `pip install -qU "semantic-router[fastembed]"` | The lightest start: small ONNX models that run fast on CPU. |
| `HuggingFaceEncoder` | `pip install -qU "semantic-router[local]"` | Any Hugging Face embedding model, on CPU, CUDA or Apple silicon. |
| `LocalEncoder` | `pip install -qU "semantic-router[local]"` | Any sentence-transformers model. |
| `TfidfEncoder` | base install | A sparse encoder fitted on your own utterances, for hybrid search. |
| `LocalSparseEncoder` | `pip install -qU "semantic-router[local]"` | Neural sparse models such as SPLADE. |

## A fully local router

FastEmbed for meaning, TF-IDF for keywords, and the default in-memory index:

```python
from semantic_router import IndexedRouter, Route
from semantic_router.encoders import FastEmbedEncoder, TfidfEncoder

routes = [
    Route(
        name="get_time",
        utterances=["what time is it in rome?", "what's the time in new york?", "tell me the time in tokyo", "current time in london"],
    ),
    Route(
        name="weather",
        utterances=["how's the weather today?", "is it going to rain?", "what's the forecast for tomorrow?", "is it cold outside?"],
    ),
]

router = IndexedRouter(
    encoder=FastEmbedEncoder(name="BAAI/bge-small-en-v1.5"),
    sparse_encoder=TfidfEncoder(),
    routes=routes,
    alpha=0.7,  # mostly meaning, some keywords
)

router("what's the time in sydney right now?").name
```

```
[Out]: 'get_time'
```

The first `FastEmbedEncoder` call downloads the model (about 70 MB) and caches it. After that, nothing leaves your machine.

Calibrate it the same way as any router, with labelled queries it hasn't seen, and save it to a file so the next start-up skips the refit. See [calibration](../features/calibration) and [save and load](save-load-from-file).

## Hugging Face models

`HuggingFaceEncoder` runs any embedding model from the Hugging Face Hub with PyTorch, on whatever hardware you have:

```python
from semantic_router.encoders import HuggingFaceEncoder

encoder = HuggingFaceEncoder(name="sentence-transformers/all-MiniLM-L6-v2")
```

Bigger models are usually more accurate and slower. Since every query is embedded on the hot path, start small and move up only if your [evaluation](../features/calibration#measure-it) says you need to.

## Pairing with a local LLM

Semantic Router doesn't run LLMs any more: it decides, and your code acts. To pair it with a local model, route first, then call the model only when a route needs one. With [Ollama](https://ollama.com), for example:

```python
from datetime import datetime
from zoneinfo import ZoneInfo

import ollama


def get_time(timezone: str) -> str:
    """Get the current time in an IANA timezone, such as "Australia/Sydney"."""
    return datetime.now(ZoneInfo(timezone)).strftime("%H:%M")


query = "what's the time in sydney right now?"
choice = router(query)
if choice.name == "get_time":
    # the router picked the tool; the local LLM fills in its arguments
    reply = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": query}], tools=[get_time])
```

The LLM never has to choose between tools, and it doesn't run at all for queries that don't need it.
