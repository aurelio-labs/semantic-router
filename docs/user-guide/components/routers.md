> **You're reading the docs for Semantic Router 0.x.** Version 1.0 is out: read [this page for v1](https://docs.aurelio.ai/docs/semantic-router/v1/user-guide/components/routers), or the [migration guide](https://docs.aurelio.ai/docs/semantic-router/v1/user-guide/guides/migration-to-v1) to upgrade.

The router is the part you actually talk to. It takes a query, finds the best-matching route, and hands back a decision. Under the hood it wires together an encoder, an index, and your routes.

## What a router does

1. **Encodes** incoming queries.
2. **Matches** them to routes by similarity.
3. **Decides** which handler should run.
4. **Manages** routes — add, look up, list.
5. **Reports** a confidence score with each decision.

## Router types

### SemanticRouter

The standard router. It uses dense embeddings and is what most people reach for first.

```python
import os
from semantic_router import Route, SemanticRouter
from semantic_router.encoders import OpenAIEncoder
from semantic_router.index import LocalIndex

os.environ["OPENAI_API_KEY"] = "your-api-key"

routes = [
    Route(name="weather", utterances=["How's the weather?", "Is it raining?"]),
    Route(name="politics", utterances=["Tell me about politics", "Who's the president?"]),
]

router = SemanticRouter(
    encoder=OpenAIEncoder(),
    routes=routes,
    index=LocalIndex(),
)

result = router("What's the weather like today?")
print(result.name)   # "weather"
print(result.score)  # e.g. 0.92
```

### HybridRouter

The `HybridRouter` combines dense and sparse embeddings — semantic understanding plus keyword matching. It often wins when your queries carry specific terms that need to match exactly.

```python
import os
from semantic_router import Route
from semantic_router.routers import HybridRouter
from semantic_router.encoders import OpenAIEncoder, AurelioSparseEncoder
from semantic_router.index import HybridLocalIndex

os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
os.environ["AURELIO_API_KEY"] = "your-aurelio-api-key"

routes = [
    Route(name="weather", utterances=["How's the weather?", "Is it raining?"]),
    Route(name="politics", utterances=["Tell me about politics", "Who's the president?"]),
]

router = HybridRouter(
    encoder=OpenAIEncoder(),
    sparse_encoder=AurelioSparseEncoder(),
    routes=routes,
    index=HybridLocalIndex(),
    alpha=0.3,  # 0 = all dense, 1 = all sparse
)

result = router("What's the weather like today?")
print(result.name)  # "weather"
```

## Working with routes

Add, fetch, and list routes on a live router:

```python
router.add(Route(name="greetings", utterances=["Hello there", "Hi, how are you?"]))

greeting_route = router.get("greetings")
route_names = router.list_route_names()
```

## Thresholds

A route only matches when its similarity score clears a threshold. Set one for the whole router, or per route:

```python
# one threshold for every route
router = SemanticRouter(
    encoder=OpenAIEncoder(),
    routes=routes,
    score_threshold=0.75,
)

# a stricter threshold on one route
weather_route = Route(
    name="weather",
    utterances=["How's the weather?", "Is it raining?"],
    score_threshold=0.8,
)
```

Picking thresholds by hand is tedious. The [threshold optimization guide](../features/threshold-optimization) shows how to fit them from examples in seconds.

## Async

Both routers work in async code:

```python
result = await router.acall("What's the weather like today?")
await router.aadd(new_route)
```

## Loading from config

Load a router from a file or a config object — handy for shipping the same routes across environments:

```python
# from YAML
router = SemanticRouter.from_yaml("router_config.yaml")

# from a RouterConfig
from semantic_router.routers import RouterConfig

config = RouterConfig(routes=routes, encoder_type="openai")
router = SemanticRouter.from_config(config)
```

## Syncing with a remote index

With a remote index, `auto_sync` controls how local and stored routes are reconciled:

```python
router = SemanticRouter(
    encoder=encoder,
    routes=routes,
    index=remote_index,
    auto_sync="remote",  # "local", "remote", or None
)
```

The [sync guide](../features/sync) covers every strategy.

## Tuning hybrid alpha

`alpha` sets the dense/sparse balance in a `HybridRouter`:

```python
alpha=0.2  # lean semantic (80% dense, 20% sparse)
alpha=0.5  # even split
alpha=0.8  # lean keywords (20% dense, 80% sparse)
```

## What a router returns

Calling a router gives you a `RouteChoice` with:

- `name` — the matched route, or `None` if nothing matched.
- `score` — the confidence of the match.
- `function_call` — for dynamic routes, the function(s) to call and their arguments.
- `metadata` — any metadata attached to the route.

## SemanticRouter or HybridRouter?

- **HybridRouter** usually scores higher, because it matches on meaning *and* keywords.
- **SemanticRouter** is lighter and faster.
- If your queries contain specific terms that must match exactly — product codes, names, jargon — go hybrid.
- Hybrid needs a sparse encoder too, so check you have the API keys or local models it needs.

The API reference has the full detail on every option.
