v0.1 reworked the API and added new capabilities. Some of those changes are breaking. This guide covers what moved, what got renamed, and how to update your code.

## Key API changes

### `RouteLayer` is now `SemanticRouter`

```python
# before
from semantic_router import RouteLayer

# after
from semantic_router.routers import SemanticRouter
```

The class was renamed and moved into the `routers` module. Same job, clearer name, and it fits the modular layout alongside `HybridRouter`.

### `add` takes a list

`add` now accepts a list of routes. A single route still works.

```python
# before
route_layer = RouteLayer(encoder=encoder)
route_layer.add(route1)
route_layer.add(route2)

# after
semantic_router = SemanticRouter(encoder=encoder)
semantic_router.add([route1, route2])  # several at once
semantic_router.add(route3)            # one still works
```

### `retrieve_multiple_routes` is gone — use `limit`

To get more than one route back, call the router with a `limit`:

```python
# before (v0.0.x)
route_layer = RouteLayer(encoder=encoder, routes=routes)
multiple_routes = route_layer.retrieve_multiple_routes(query_text)

# transitional (v0.1.0–0.1.2) — deprecated, not recommended
semantic_router = SemanticRouter(encoder=encoder, routes=routes, auto_sync="local")
query_results = semantic_router._query(query_text)
multiple_routes = semantic_router._semantic_classify_multiple_routes(query_results)

# after (v0.1.3+, 0.1.5+ recommended)
semantic_router = SemanticRouter(encoder=encoder, routes=routes, auto_sync="local")
all_routes = semantic_router(query_text, limit=None)  # every route that passes its threshold
top_routes = semantic_router(query_text, limit=3)     # the top 3 that pass

# to score every route regardless of threshold
semantic_router.set_threshold(threshold=0.0)
all_route_scores = semantic_router(query_text, limit=None)
```

With `limit=1` (the default) you get a single `RouteChoice`. With `limit=None` or `limit > 1` you get a list.

> **Watch `top_k`.** It defaults to 5 and caps how many routes come back, independently of `limit`. If you use `limit > 1`, raise `top_k` — 100 or more is reasonable. If you use `limit=None` to get everything, set `top_k` to at least the total number of utterances across all your routes.
>
> ```python
> semantic_router = SemanticRouter(encoder=encoder, routes=routes, top_k=100)
> all_routes = semantic_router(query_text, limit=None)
> ```

### Sync is now explicit

If you expect local and remote routes to sync at startup, say so with `auto_sync`:

```python
semantic_router = SemanticRouter(
    encoder=encoder,
    routes=routes,
    index=PineconeIndex(...),
    auto_sync="local",  # push local routes to the remote index
)
```

The modes:

- `error` — raise if local and remote differ.
- `remote` — remote wins; update local.
- `local` — local wins; update remote.
- `merge-force-local` — merge, local takes priority.
- `merge-force-remote` — merge, remote takes priority.
- `merge` — merge, local wins on conflicts.

The [sync guide](../features/sync) explains each in depth.

## Other changes

### `RouterConfig` replaces `LayerConfig`

```python
from semantic_router.routers import RouterConfig

config = RouterConfig(
    routes=[route1, route2],
    encoder_type="openai",
    encoder_name="text-embedding-3-small",
)

semantic_router = SemanticRouter.from_config(config)
```

### More router types

The modular layout exposes:

- `SemanticRouter` — the standard router (the old `RouteLayer`).
- `HybridRouter` — dense plus sparse embeddings.
- `BaseRouter` — an abstract base for your own routers.

## Before and after

```python
# before (v0.0.x)
from semantic_router import RouteLayer, Route
from semantic_router.encoders import OpenAIEncoder

route = Route(name="example", utterances=["sample utterance"])
layer = RouteLayer(encoder=OpenAIEncoder())
layer.add(route)
result = layer("query text")

# after (v0.1.x)
from semantic_router import Route
from semantic_router.routers import SemanticRouter
from semantic_router.encoders import OpenAIEncoder

route = Route(name="example", utterances=["sample utterance"])
router = SemanticRouter(encoder=OpenAIEncoder())
router.add(route)
result = router("query text")
```
