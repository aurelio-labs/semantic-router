# Quickstart

To get started with *semantic-router* we install it like so:

```bash
pip install -qU semantic-router
```

> **Warning**
> If wanting to use a fully local version of semantic router you can use `HuggingFaceEncoder` and `LlamaCppLLM` (`pip install -qU "semantic-router[local]"`, see [here](https://github.com/aurelio-labs/semantic-router/blob/main/docs/05-local-execution.ipynb)). To use the `HybridRouteLayer` you must `pip install -qU "semantic-router[hybrid]"`.

We begin by defining a set of `Route` objects. These are the decision paths that the semantic router can decide to use, let's try two simple routes for now — one for talk on *politics* and another for *chitchat*:

```python
from semantic_router import Route

# we could use this as a guide for our chatbot to avoid political conversations
politics = Route(
    name="politics",
    utterances=[
        "isn't politics the best thing ever",
        "why don't you tell me about your political opinions",
        "don't you just love the president",
        "they're going to destroy this country!",
        "they will save the country!",
    ],
)

# this could be used as an indicator to our chatbot to switch to a more
# conversational prompt
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

# we place both of our decisions together into single list
routes = [politics, chitchat]
```

We have our routes ready, now we initialize an embedding / encoder model. We currently support a `CohereEncoder` and `OpenAIEncoder` — more encoders will be added soon. To initialize them we do:

```python
import os
from semantic_router.encoders import CohereEncoder, OpenAIEncoder

# for Cohere
os.environ["COHERE_API_KEY"] = "<YOUR_API_KEY>"
encoder = CohereEncoder()

# or for OpenAI
os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"
encoder = OpenAIEncoder()
```

With our `routes` and `encoder` defined we now create a `RouteLayer`. The route layer handles our semantic decision making.

```python
from semantic_router.layer import RouteLayer

rl = RouteLayer(encoder=encoder, routes=routes)
```

We can now use our route layer to make super fast decisions based on user queries. Let's try with two queries that should trigger our route decisions:

```python
rl("don't you love politics?").name
```

```
[Out]: 'politics'
```

Correct decision, let's try another:

```python
rl("how's the weather today?").name
```

```
[Out]: 'chitchat'
```

We get both decisions correct! Now lets try sending an unrelated query:

```python
rl("I'm interested in learning about llama 2").name
```

```
[Out]:
```

In this case, no decision could be made as we had no matches — so our route layer returned `None`!