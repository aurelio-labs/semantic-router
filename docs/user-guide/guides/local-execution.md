> **You're reading the docs for Semantic Router 0.x.** Version 1.0 is out: read [this page for v1](https://docs.aurelio.ai/docs/semantic-router/v1/user-guide/guides/local-execution), or the [migration guide](https://docs.aurelio.ai/docs/semantic-router/v1/user-guide/guides/migration-to-v1) to upgrade.

Plenty of reasons to run your own LLM instead of calling an API — cost, privacy, compliance. Semantic Router supports local LLMs through `llama.cpp`.

`llama.cpp` also runs quantized GGUF models, which shrink memory use enough that even a 13B-parameter model runs with hardware acceleration on an Apple M1 Pro.

## Full example

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-labs/semantic-router/blob/main/docs/05-local-execution.ipynb)
[![Open nbviewer](https://raw.githubusercontent.com/pinecone-io/examples/master/assets/nbviewer-shield.svg)](https://nbviewer.org/github/aurelio-labs/semantic-router/blob/main/docs/05-local-execution.ipynb)

We'll use **Mistral-7B-Instruct**, quantized to 4-bit to keep the memory footprint small.

## Install

> For hardware acceleration (BLAS, CUDA, Metal, and so on), see the [llama-cpp-python README](https://github.com/abetlen/llama-cpp-python#installation-with-specific-hardware-acceleration-blas-cuda-metal-etc).

```python
pip install -qU "semantic-router[local]"
```

On Apple silicon, compile with Metal acceleration:

```bash
CMAKE_ARGS="-DLLAMA_METAL=on" pip install -qU "semantic-router[local]"
```

## Download the model

Mistral 7B Instruct as a 4-bit GGUF is a good balance of quality and consumer-hardware friendliness:

```python
!curl -L "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_0.gguf?download=true" -o ./mistral-7b-instruct-v0.2.Q4_0.gguf
!ls mistral-7b-instruct-v0.2.Q4_0.gguf
```

## Define the routes

We'll include a dynamic route so the local LLM has some function calling to do:

```python
from datetime import datetime
from zoneinfo import ZoneInfo

from semantic_router import Route
from semantic_router.utils.function_call import get_schema


def get_time(timezone: str) -> str:
    """Finds the current time in a specific timezone.

    :param timezone: The timezone to find the current time in, should
        be a valid timezone from the IANA Time Zone Database like
        "America/New_York" or "Europe/London". Do NOT put the place
        name itself like "rome", or "new york", you must provide
        the IANA format.
    :type timezone: str
    :return: The current time in the specified timezone."""
    now = datetime.now(ZoneInfo(timezone))
    return now.strftime("%H:%M")


time_schema = get_schema(get_time)
time = Route(
    name="get_time",
    utterances=[
        "what is the time in new york city?",
        "what is the time in london?",
        "I live in Rome, what time is it?",
    ],
    function_schemas=[time_schema],
)

politics = Route(
    name="politics",
    utterances=[
        "isn't politics the best thing ever",
        "why don't you tell me about your political opinions",
        "don't you just love the president",
        "don't you just hate the president",
        "they're going to destroy this country!",
        "they will save the country!",
    ],
)
chitchat = Route(
    name="chitchat",
    utterances=[
        "how's the weather today?",
        "how are things going?",
        "lovely weather today",
        "the weather is horrendous",
        "let's go to the chippy",
    ],
)

routes = [politics, chitchat, time]
```

## Encoder

To stay fully local we'll use `HuggingFaceEncoder`, which defaults to `sentence-transformers/all-MiniLM-L6-v2`:

```python
from semantic_router.encoders import HuggingFaceEncoder

encoder = HuggingFaceEncoder()
```

## The `llama.cpp` LLM

Create a `llama_cpp.Llama` and wrap it in `LlamaCppLLM`. Three parameters worth knowing:

- `n_gpu_layers` — how many layers to offload to the GPU. `-1` for the whole model, `0` for CPU only.
- `n_ctx` — the context window. Capped by the model's own limit (8000 tokens for Mistral-7B-Instruct).
- `verbose` — set `False` to quiet `llama.cpp`'s output.

> The [llama-cpp-python API reference](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/) covers the rest.

```python
from semantic_router import SemanticRouter

from llama_cpp import Llama
from semantic_router.llms.llamacpp import LlamaCppLLM

enable_gpu = True  # offload to GPU (the model must fit in memory)

_llm = Llama(
    model_path="./mistral-7b-instruct-v0.2.Q4_0.gguf",
    n_gpu_layers=-1 if enable_gpu else 0,
    n_ctx=2048,
)
_llm.verbose = False
llm = LlamaCppLLM(name="Mistral-7B-v0.2-Instruct", llm=_llm, max_tokens=None)

router = SemanticRouter(encoder=encoder, routes=routes, llm=llm)
```

## Try it

A static route first:

```python
router("how's the weather today?")
```

```
RouteChoice(name='chitchat', function_call=None, similarity_score=None)
```

Now a time question, which triggers the dynamic route and the local LLM:

```python
out = router("what's the time in New York right now?")
print(out)
get_time(**out.function_call[0])
```

```
name='get_time' function_call=[{'timezone': 'America/New_York'}] similarity_score=None
'07:50'
```

A couple more:

```python
out = router("what's the time in Rome right now?")
print(out)
get_time(**out.function_call[0])
```

```
name='get_time' function_call=[{'timezone': 'Europe/Rome'}] similarity_score=None
'13:51'
```

```python
out = router("what's the time in Bangkok right now?")
print(out)
get_time(**out.function_call[0])
```

```
name='get_time' function_call=[{'timezone': 'Asia/Bangkok'}] similarity_score=None
'18:51'
```

All local, no API calls.

## Cleanup

Delete the model when you're done:

```bash
rm ./mistral-7b-instruct-v0.2.Q4_0.gguf
```
