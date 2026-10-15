---
title: Quickstart
description: Build a router, route queries, and calibrate its confidence in a few minutes, with no API keys.
---

Semantic Router sends text to the right handler based on *meaning*, not keywords. That makes it a good fit for chatbots, intent classification and guardrails: anything that needs to understand what a user is actually asking.

This quickstart runs entirely on your machine. It uses a small local embedding model through FastEmbed, so you don't need any API keys.

```bash
pip install -qU "semantic-router[fastembed]>=1.0.0"
```

## Define your routes

A `Route` is a decision you want the router to make. You describe it with example utterances: a few phrases a user might say. Those examples become the route's reference point in semantic space.

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

An encoder turns text into a vector, so the router can measure how close two pieces of text are in meaning. We'll use a small local model. The first call downloads it (about 70 MB).

```python
from semantic_router.encoders import FastEmbedEncoder

encoder = FastEmbedEncoder(name="BAAI/bge-small-en-v1.5")
```

Semantic Router supports plenty of others: OpenAI, Cohere, Hugging Face, and more. The [encoders guide](../user-guide/components/encoders) has the full list.

## Create the router

An `IndexedRouter` embeds every utterance once, stores them in an index, and matches each query against them. Hand it your encoder and routes:

```python
from semantic_router import IndexedRouter

router = IndexedRouter(encoder=encoder, routes=routes)
```

With no index given, it keeps the vectors in memory. The [indexes guide](../user-guide/components/indexes) shows how to use Pinecone, Qdrant or Postgres instead.

## Route a query

Call the router with a query. It embeds the query and scores it against every route.

```python
router("don't you love politics?")
```

```
[Out]: RouteChoice(name='politics', confidence=0.6717691933713713, calibrated=False)
```

You get back a `RouteChoice`: the route's `name`, a `confidence`, and `calibrated`, which says whether that confidence is a probability. Right now it isn't: it's the raw similarity between the query and the route's utterances.

```python
router("how's the weather today?")
```

```
[Out]: RouteChoice(name='chitchat', confidence=0.8048876076635829, calibrated=False)
```

Both correct, even though neither query matches an utterance word for word.

## When nothing should match

Now send something unrelated:

```python
router("I'm interested in learning about llama 2")
```

```
[Out]: RouteChoice(name='politics', confidence=0.46913750545106614, calibrated=False)
```

It picked a route anyway. That's the default: **with no threshold set, the best route always wins**, however weak the match. To let the router say "none of these", give it a bar to clear. You could set a raw similarity bar by hand, but the right number depends on the encoder, the routes and your data. Calibration finds it for you.

## Calibrate the confidence

`fit()` takes some labelled example queries and learns to turn the router's raw scores into probabilities. Use queries the router hasn't seen, *not* the utterances you gave the routes. Label the ones that shouldn't match any route with `None`.

```python
X = [
    "who should I vote for in the election?",
    "what do you think of the prime minister?",
    "is the government doing a good job?",
    "tell me your opinion on the new tax law",
    "are you left wing or right wing?",
    "the opposition party is a joke",
    "nice day out, isn't it?",
    "it's freezing outside",
    "how was your weekend?",
    "fancy grabbing some fish and chips?",
    "what a lovely sunny afternoon",
    "how have you been?",
    "how do I reset my password?",
    "what's the capital of australia?",
    "write me a python function to sort a list",
    "what is the boiling point of water?",
    "recommend a good sci-fi book",
    "how many legs does a spider have?",
]
y = ["politics"] * 6 + ["chitchat"] * 6 + [None] * 6

router.fit(X=X, y=y)
```

```
[Out]: FitReport(method='platt', n=18, accuracy=0.9166666666666666, ece=0.23736602563869935)
```

The report gives the top-route accuracy on your examples and the calibration error (`ece`, lower is better). Eighteen examples is enough to see it work. For a real app, use a few dozen per route.

Now confidences are probabilities:

```python
router("don't you love politics?")
```

```
[Out]: RouteChoice(name='politics', confidence=0.9295308430490237, calibrated=True)
```

## Set a threshold

Set a `confidence_threshold`, and a route only wins if its confidence clears it. Because the router is calibrated, 0.5 means "only answer when it's more likely right than wrong".

```python
router.confidence_threshold = 0.5

router("I'm interested in learning about llama 2")
```

```
[Out]: RouteChoice(name=None, confidence=None, calibrated=True)
```

No route cleared the bar, so `name` is `None`. Treat that as your fallback signal: pass the query through, send it to a default handler, whatever fits your app. The on-topic queries still route:

```python
router("how's the weather today?")
```

```
[Out]: RouteChoice(name='chitchat', confidence=0.9937006111667317, calibrated=True)
```

You can also pass `confidence_threshold=` when you build the router, and give any single route its own bar with `Route(confidence_threshold=...)`. The [calibration guide](../user-guide/features/calibration) covers both.

## Next steps

- [Routes](../user-guide/components/routes) and [routers](../user-guide/components/routers): every option, including hybrid search.
- [Calibration](../user-guide/features/calibration): how `fit()` works, and choosing between Platt and isotonic.
- [Decision models](../user-guide/features/decision-models) and [Chain and Merge](../user-guide/features/composition): when an embedding match isn't enough.
- Coming from 0.x? Read the [migration guide](../user-guide/guides/migration-to-v1).
