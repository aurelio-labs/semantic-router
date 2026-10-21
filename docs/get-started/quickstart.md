> **You're reading the docs for Semantic Router 0.x.** Version 1.0 is out: read [this page for v1](https://docs.aurelio.ai/docs/semantic-router/v1/get-started/quickstart), or the [migration guide](https://docs.aurelio.ai/docs/semantic-router/v1/user-guide/guides/migration-to-v1) to upgrade.

Semantic Router sends text to the right handler based on *meaning*, not keywords. That makes it a good fit for chatbots, intent classification, guardrails — anything that needs to understand what a user is actually asking.

Install it:

```bash
pip install -qU semantic-router
```

Want everything to run locally, with no API calls? Install the `local` extra and use `HuggingFaceEncoder` with `LlamaCppLLM`. The [local execution guide](../user-guide/guides/local-execution) walks through it.

```bash
pip install -qU "semantic-router[local]"
```

## Define your routes

A `Route` is a topic or intent you want to detect. You describe it with example utterances — a few phrases a user might say. Those examples become the route's semantic reference point.

Let's start with two: one for *politics*, one for *chitchat*.

```python
from semantic_router import Route

# use this to steer a chatbot away from political conversation
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

# and this to switch the chatbot into a more conversational mode
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

routes = [politics, chitchat]
```

## Pick an encoder

An encoder turns text into a vector, so we can measure how similar two pieces of text are. Semantic Router supports a lot of them — OpenAI, Cohere, Hugging Face, FastEmbed, and more. The [encoders guide](../user-guide/components/encoders) has the full list.

We'll use Cohere or OpenAI here:

```python
import os
from semantic_router.encoders import CohereEncoder, OpenAIEncoder

# Cohere
os.environ["COHERE_API_KEY"] = "<YOUR_API_KEY>"
encoder = CohereEncoder()

# or OpenAI
os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"
encoder = OpenAIEncoder()
```

## Create the router

The `SemanticRouter` is the decision engine. Hand it your encoder and routes:

```python
from semantic_router import SemanticRouter

sr = SemanticRouter(encoder=encoder, routes=routes)
```

## Route a query

Now call it. Under the hood, the router embeds your query and finds the closest route.

```python
sr("don't you love politics?").name
```

```
[Out]: 'politics'
```

Right answer. One more:

```python
sr("how's the weather today?").name
```

```
[Out]: 'chitchat'
```

Both correct. Notice the queries don't match any utterance word for word — they just mean the same thing. That's the whole point.

## When nothing matches

Send something unrelated:

```python
sr("I'm interested in learning about llama 2").name
```

```
[Out]:
```

No route was close enough, so the router returns `None`. Treat that as your fallback signal: pass the query through, hand it to a default handler, whatever fits your app.
