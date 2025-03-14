*Semantic-router* is a lightweight library that helps you intelligently route text to the right handlers based on meaning rather than exact keyword matching. It's perfect for building chatbots, classification systems, or any application that needs to understand user intent.

To get started with *semantic-router* we install it like so:

```bash
pip install -qU semantic-router
```

> **Warning**
> If wanting to use a fully local version of semantic router you can use `HuggingFaceEncoder` and `LlamaCppLLM` (`pip install -qU "semantic-router[local]"`, see [here](../user-guide/guides/local-execution)). To use the `HybridRouteLayer` you must `pip install -qU "semantic-router[hybrid]"`.

## Defining Routes

We begin by defining a set of `Route` objects. A Route represents a specific topic or intent that you want to detect in user input. Each Route is defined by example utterances that serve as a semantic reference point.

Let's try two simple routes for now — one for talk on *politics* and another for *chitchat*:

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

## Setting Up an Encoder

With our routes ready, now we initialize an embedding / encoder model. The encoder converts text into numerical vectors, allowing the system to measure semantic similarity. We currently support `CohereEncoder` and `OpenAIEncoder` — more encoders will be added soon.

To initialize them:

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

## Creating a RouteLayer

With our `routes` and `encoder` defined we now create a `RouteLayer`. The RouteLayer is the decision-making engine that compares incoming text against your routes to find the best semantic match.

```python
from semantic_router.layer import RouteLayer

rl = RouteLayer(encoder=encoder, routes=routes)
```

## Making Routing Decisions

We can now use our route layer to make super fast routing decisions based on user queries. Behind the scenes, the system converts both your example utterances and the incoming query into vectors and finds the closest match.

Let's try with two queries that should trigger our route decisions:

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

We get both decisions correct! The power of semantic routing is that it works even when queries don't exactly match your examples but are similar in meaning.

## Handling Unmatched Queries

Now let's try sending an unrelated query:

```python
rl("I'm interested in learning about llama 2").name
```

```
[Out]:
```

In this case, no decision could be made as we had no semantic matches — so our route layer returned `None`! This feature is useful for creating fallback behavior or passthroughs in your applications when no intent is clearly matched.