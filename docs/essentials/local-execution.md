There are many reasons users might choose to roll their own LLMs rather than use a third-party service. Whether it's due to cost, privacy or compliance, Semantic Router supports the use of "local" LLMs through `llama.cpp`.

Using `llama.cpp` also enables the use of quantized GGUF models, reducing the memory footprint of deployed models, allowing even 13-billion parameter models to run with hardware acceleration on an Apple M1 Pro chip.

## Full Example

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-labs/semantic-router/blob/main/docs/05-local-execution.ipynb)
[![Open nbviewer](https://raw.githubusercontent.com/pinecone-io/examples/master/assets/nbviewer-shield.svg)](https://nbviewer.org/github/aurelio-labs/semantic-router/blob/main/docs/05-local-execution.ipynb)

Below is an example of using semantic router with **Mistral-7B-Instruct**, quantized to reduce memory footprint.

## Installing the library

> Note: if you require hardware acceleration via BLAS, CUDA, Metal, etc. please refer to the [abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python#installation-with-specific-hardware-acceleration-blas-cuda-metal-etc) repository README.md

```python
pip install -qU "semantic-router[local]"
```

If you're running on Apple silicon you can run the following to compile with Metal hardware acceleration:

```bash
CMAKE_ARGS="-DLLAMA_METAL=on" pip install -qU "semantic-router[local]"
```

## Download the Mistral 7B Instruct 4-bit GGUF files

We will be using Mistral 7B Instruct, quantized as a 4-bit GGUF file, a good balance between performance and ability to deploy on consumer hardware

```python
!curl -L "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_0.gguf?download=true" -o ./mistral-7b-instruct-v0.2.Q4_0.gguf
!ls mistral-7b-instruct-v0.2.Q4_0.gguf
```

## Initializing Dynamic Routes

Similar to dynamic routes in other examples, we will be initializing some dynamic routes that make use of LLMs for function calling

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
        "don't you just love the president" "don't you just hate the president",
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

## Encoders

You can use alternative Encoders, however, in this example we want to showcase a fully-local Semantic Router execution, so we are going to use a `HuggingFaceEncoder` with `sentence-transformers/all-MiniLM-L6-v2` (the default) as an embedding model.

```python
from semantic_router.encoders import HuggingFaceEncoder

encoder = HuggingFaceEncoder()
```

## `llama.cpp` LLM

From here, we can go ahead and instantiate our `llama-cpp-python` `llama_cpp.Llama` LLM, and then pass it to the `semantic_router.llms.LlamaCppLLM` wrapper class.

For `llama_cpp.Llama`, there are a couple of parameters you should pay attention to:

- `n_gpu_layers`: how many LLM layers to offload to the GPU (if you want to offload the entire model, pass `-1`, and for CPU execution, pass `0`)
- `n_ctx`: context size, limit the number of tokens that can be passed to the LLM (this is bounded by the model's internal maximum context size, in this case for Mistral-7B-Instruct, 8000 tokens)
- `verbose`: if `False`, silences output from `llama.cpp`

> For other parameter explanation, refer to the `llama-cpp-python` [API Reference](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/)

```python
# In semantic-router v0.1.0, RouteLayer has been replaced with SemanticRouter
from semantic_router import SemanticRouter

from llama_cpp import Llama
from semantic_router.llms.llamacpp import LlamaCppLLM

enable_gpu = True  # offload LLM layers to the GPU (must fit in memory)

_llm = Llama(
    model_path="./mistral-7b-instruct-v0.2.Q4_0.gguf",
    n_gpu_layers=-1 if enable_gpu else 0,
    n_ctx=2048,
)
_llm.verbose = False
llm = LlamaCppLLM(name="Mistral-7B-v0.2-Instruct", llm=_llm, max_tokens=None)

# Initialize SemanticRouter with our encoder, routes, and LLM
router = SemanticRouter(encoder=encoder, routes=routes, llm=llm)
```

Let's test our router with some queries:

```python
router("how's the weather today?")
```

This should output:
```
RouteChoice(name='chitchat', function_call=None, similarity_score=None)
```

Now let's try a time-related query that will trigger our function calling:

```python
out = router("what's the time in New York right now?")
print(out)
get_time(**out.function_call[0])
```

This should output something like:
```
name='get_time' function_call=[{'timezone': 'America/New_York'}] similarity_score=None
'07:50'
```

Let's try more examples:

```python
out = router("what's the time in Rome right now?")
print(out)
get_time(**out.function_call[0])
```

Output:
```
name='get_time' function_call=[{'timezone': 'Europe/Rome'}] similarity_score=None
'13:51'
```

```python
out = router("what's the time in Bangkok right now?")
print(out)
get_time(**out.function_call[0])
```

Output:
```
name='get_time' function_call=[{'timezone': 'Asia/Bangkok'}] similarity_score=None
'18:51'
```

## Cleanup

Once done, if you'd like to delete the downloaded model you can do so with the following:

```bash
rm ./mistral-7b-instruct-v0.2.Q4_0.gguf
``` 