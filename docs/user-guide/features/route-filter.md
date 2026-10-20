> **You're reading the docs for Semantic Router 0.x.** Version 1.0 removes `route_filter`. The [migration guide](https://docs.aurelio.ai/docs/semantic-router/v1/user-guide/guides/migration-to-v1#route_filter-is-gone) shows what to do instead.

Sometimes you only want the router to consider *some* of its routes. Maybe context tells you the user is mid-conversation about one topic, or a certain route shouldn't fire in this part of your app. `route_filter` lets you narrow the field per call.

Say your router has `politics`, `weather`, and `chitchat`. To consider only `chitchat` for one query:

```python
sr("don't you love politics?", route_filter=["chitchat"])
```

The router ignores every route not in the list.

## Full example

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-labs/semantic-router/blob/main/docs/09-route-filter.ipynb)
[![Open nbviewer](https://raw.githubusercontent.com/pinecone-io/examples/master/assets/nbviewer-shield.svg)](https://nbviewer.org/github/aurelio-labs/semantic-router/blob/main/docs/09-route-filter.ipynb)

```python
!pip install -qU semantic-router
```

Two routes to work with:

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

An encoder:

```python
import os
from getpass import getpass
from semantic_router.encoders import CohereEncoder, OpenAIEncoder

os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY") or getpass(
    "Enter Cohere API Key: "
)
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or getpass(
#     "Enter OpenAI API Key: "
# )

encoder = CohereEncoder()
# encoder = OpenAIEncoder()
```

And the router:

```python
from semantic_router import SemanticRouter

sr = SemanticRouter(encoder=encoder, routes=routes)
```

Without a filter it behaves as usual:

```python
sr("don't you love politics?")
```

```
RouteChoice(name='politics', function_call=None, similarity_score=None)
```

```python
sr("how's the weather today?")
```

```
RouteChoice(name='chitchat', function_call=None, similarity_score=None)
```

And an unrelated query returns `None`:

```python
sr("I'm interested in learning about llama 2")
```

```
RouteChoice(name=None, function_call=None, similarity_score=None)
```

## Filtering

Now restrict the router to `chitchat` and send it a political query:

```python
sr("don't you love politics?", route_filter=["chitchat"])
```

```
RouteChoice(name='chitchat', function_call=None, similarity_score=None)
```

It comes back as `chitchat` — the query would normally match `politics`, but that route wasn't allowed, and `chitchat` still cleared its threshold.

The other way round:

```python
sr("how's the weather today?", route_filter=["politics"])
```

```
RouteChoice(name=None, function_call=None, similarity_score=None)
```

`None` this time. The weather query is nowhere near `politics`, so nothing passed the threshold. A filter narrows the candidates; it doesn't force a match.
