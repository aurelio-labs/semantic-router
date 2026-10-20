> **You're reading the docs for Semantic Router 0.x.** `SemanticRouter` is now `IndexedRouter` in version 1.0: read [the v1 page](https://docs.aurelio.ai/docs/semantic-router/v1/user-guide/components/routers), or the [migration guide](https://docs.aurelio.ai/docs/semantic-router/v1/user-guide/guides/migration-to-v1) to upgrade.

The `SemanticRouter` is the heart of the library. Give it a query and it decides which route to take. It's built from three things: an `encoder`, an `index`, and a list of `routes`. If any of your routes are dynamic (they produce function calls), it also holds an `llm`.

## Routes

Start with some routes:

```python
from semantic_router import Route

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

routes = [politics, chitchat]
```

## Encoder

There are plenty of encoders to choose from, local and API-based. We'll use `OpenAIEncoder` here:

```python
import os
from semantic_router.encoders import OpenAIEncoder

os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"

encoder = OpenAIEncoder()
```

## The router

Pass in the encoder and routes. When you call the router with a query, it returns the `Route` the query belongs to.

```python
from semantic_router import SemanticRouter

sr = SemanticRouter(encoder=encoder, routes=routes, auto_sync="local")
```

```python
sr("don't you love politics?")
```

```
[Out]: RouteChoice(name='politics', function_call=None, similarity_score=None)
```

You get back a `RouteChoice`. It holds the route name, any function call (for dynamic routes), and the similarity score that triggered the match.

```python
sr("how's the weather today?")
```

```
[Out]: RouteChoice(name='chitchat', function_call=None, similarity_score=None)
```

Both right. Now something that matches neither route:

```python
sr("I'm interested in learning about llama 3")
```

```
[Out]: RouteChoice(name=None, function_call=None, similarity_score=None)
```

No route cleared its threshold, so `name` is `None`.

## Getting more than one route

By default the router returns the single best match. Pass `limit` to get several, each with its score. `limit=None` returns every route that passes its threshold; `limit=3` returns the top three.

```python
sr("Hi! How are you doing in politics??", limit=None)
```

```
[Out]: [RouteChoice(name='politics', function_call=None, similarity_score=0.859),
        RouteChoice(name='chitchat', function_call=None, similarity_score=0.835)]
```

If nothing passes, you get an empty list:

```python
sr("I'm interested in learning about llama 3", limit=None)
```

```
[Out]: []
```

> One thing to watch: `top_k` (default 5) caps how many routes are considered, independently of `limit`. If you're using `limit > 1` or `limit=None`, raise `top_k` — for example `SemanticRouter(..., top_k=100)`.

The [introductory notebook](https://github.com/aurelio-labs/semantic-router/blob/main/docs/00-introduction.ipynb) walks through all of this end to end.
