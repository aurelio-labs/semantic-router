---
title: Migrating from 0.x to 1.0
description: Every breaking change in Semantic Router 1.0, with before and after code.
---

Semantic Router 1.0 is a breaking release. The routing layer was rebuilt around calibrated confidence and composable routers, and a lot of 0.x machinery is gone: index sync, dynamic routes and the LLM classes.

This guide walks through every change, each with the 0.x code and its 1.0 replacement. Most apps need only the first few sections: the router class, thresholds and `fit()`.

Not ready to move? Pin `semantic-router<1`. The 0.x line gets bug fixes on the [`v0` branch](https://github.com/aurelio-labs/semantic-router/tree/v0), and its docs stay online under [v0](../../get-started/versions).

## At a glance

| 0.x | 1.0 |
|---|---|
| `SemanticRouter` | `IndexedRouter` |
| `HybridRouter` | `IndexedRouter(sparse_encoder=..., alpha=...)`; `alpha` defaults to 0.7, not 0.3 |
| `encoder` defaults to `OpenAIEncoder()` | `encoder` is required |
| a default `score_threshold` on every encoder | removed: no threshold unless you set one |
| `score_threshold` (route and router) | `confidence_threshold` (route and router) |
| `fit()` searches for thresholds, returns `None` | `fit()` calibrates confidence, returns a `FitReport` |
| `RouteChoice.similarity_score` | `RouteChoice.confidence`, plus `RouteChoice.calibrated` |
| `RouteChoice.function_call` | removed, along with dynamic routes |
| `auto_sync`, `sync()`, `is_synced()`, `get_utterance_diff()` | removed: the index is the source of truth |
| `Route(function_schemas=..., llm=...)`, `semantic_router.llms` | removed |
| `router(..., route_filter=[...])` | removed: filter the ranked results |
| `router(vector=...)`, `simulate_static=` | removed |
| `router.delete(route_name=...)` | `router.remove(name=...)` |
| `router.get(...)`, `list_route_names()`, `update()` | read `router.routes`; `remove()` then `add()` |
| `to_yaml(file_path)`, `from_yaml()`, `RouterConfig` | `to_yaml(path=)`, `load_router(path=)` |
| Python 3.9+ | Python 3.10+ |

New in 1.0, with no 0.x equivalent: [`DecisionRouter`](../features/decision-models), [`Chain` and `Merge`](../features/composition), and [calibration](../features/calibration).

## Install

```bash
pip install -qU "semantic-router>=1,<2"
```

The extras are unchanged, except that `[local]` no longer installs `llama-cpp-python`, because the LLM classes are gone. FastEmbed has its own extra, `[fastembed]`.

## SemanticRouter is now IndexedRouter

The router that embeds your utterances and searches an index is now called `IndexedRouter`, because it's one kind of router among several. It takes the same core arguments.

**0.x**

```python
# 0.x
from semantic_router import Route, SemanticRouter
from semantic_router.encoders import OpenAIEncoder

router = SemanticRouter(
    encoder=OpenAIEncoder(),
    routes=routes,
    auto_sync="local",
)
choice = router("why was I charged twice?")
choice.name, choice.similarity_score
```

**1.0**

```python
from semantic_router import IndexedRouter, Route
from semantic_router.encoders import OpenAIEncoder

router = IndexedRouter(
    encoder=OpenAIEncoder(),
    routes=routes,
)
choice = router("why was I charged twice?")
choice.name, choice.confidence, choice.calibrated
```

The constructor embeds every utterance and writes it to the index straight away, so the router is ready when it returns. `index`, `top_k` and `aggregation` work as before.

## HybridRouter is a flag, not a class

Pass a `sparse_encoder` to `IndexedRouter` and it becomes a hybrid router. Without an index, it uses a `HybridLocalIndex`.

**0.x**

```python
# 0.x
from semantic_router.routers import HybridRouter
from semantic_router.encoders import BM25Encoder, OpenAIEncoder

router = HybridRouter(
    encoder=OpenAIEncoder(),
    sparse_encoder=BM25Encoder(),
    routes=routes,
    alpha=0.3,
)
```

**1.0**

```python
from semantic_router import IndexedRouter
from semantic_router.encoders import BM25Encoder, OpenAIEncoder

router = IndexedRouter(
    encoder=OpenAIEncoder(),
    sparse_encoder=BM25Encoder(),
    routes=routes,
    alpha=0.7,
)
```

`alpha` is still the weight on the dense score, with the rest on the sparse score. Three things changed:

- **The default is 0.7, not 0.3**, so meaning leads and keywords boost. On our test sets, 0.7 beat 0.3 by up to 17 points of top-1 accuracy (BM25: .93 against .76). The full table is on the [routers page](../components/routers#hybrid-search).
- **On `HybridLocalIndex`, `alpha` now weights the dense side too.** In 0.x that index normalised the query vector, which cancelled `alpha` on the dense score, so only the sparse score was scaled. Expect hybrid scores to move.
- **On Pinecone, `alpha` already weighted both sides.** To keep 0.x's weighting there, pass `alpha=0.3`.

What's gone is the threshold rescaling: `HybridRouter` multiplied its threshold by `alpha`, and 1.0 doesn't. Calibration absorbs the difference, so refit after you upgrade.

A *fitted* sparse encoder such as `BM25Encoder` changes its vocabulary whenever the routes change. So `add()` on a hybrid router refits the sparse encoder and rebuilds the whole index. On Pinecone, rebuilding deletes and recreates the remote index.

## The encoder is required

0.x quietly built an `OpenAIEncoder()` when you didn't pass one. 1.0 makes you choose.

**0.x**

```python
# 0.x
router = SemanticRouter(routes=routes)  # OpenAI, from OPENAI_API_KEY
```

**1.0**

```python
from semantic_router.encoders import OpenAIEncoder

router = IndexedRouter(encoder=OpenAIEncoder(), routes=routes)
```

## There's no default threshold

This is the change most likely to surprise you. In 0.x, every encoder came with a default `score_threshold` (0.3 for OpenAI, 0.5 for FastEmbed, and so on), so an off-topic query usually came back with `name=None`.

In 1.0, a router has **no threshold unless you set one**, and the best route always wins, however weak the match. The encoders no longer take a `score_threshold` at all.

**0.x**

```python
# 0.x
router = SemanticRouter(encoder=encoder, routes=routes)
router("what's the capital of australia?").name  # None: under the encoder's default
```

**1.0**

```python
router = IndexedRouter(encoder=encoder, routes=routes)
router("what's the capital of australia?").name  # the closest route, however weak

# calibrate, then set a bar to get None back for off-topic queries
router.fit(X=held_out_queries, y=labels)
router.confidence_threshold = 0.5
router("what's the capital of australia?").name  # None
```

If you depended on `name=None` for a fallback path, add a `confidence_threshold` before you ship.

## score_threshold is now confidence_threshold

Thresholds sit in the same two places as before, the router and the route, under a new name. A route's own threshold wins over the router's.

**0.x**

```python
# 0.x
billing = Route(name="billing", utterances=billing_utterances, score_threshold=0.82)
router = SemanticRouter(encoder=encoder, routes=[billing, chitchat], score_threshold=0.75)

router.set_threshold(threshold=0.8, route_name="billing")
router.get_thresholds()
```

**1.0**

```python
billing = Route(name="billing", utterances=billing_utterances, confidence_threshold=0.9)
router = IndexedRouter(encoder=encoder, routes=[billing, chitchat], confidence_threshold=0.75)

router.confidence_threshold = 0.8  # the router's default bar
{r.name: r.confidence_threshold for r in router.routes}
```

The name changed because the number changed. Until you call `fit()`, the bar applies to raw similarity, as in 0.x. After `fit()`, it applies to a calibrated probability: 0.9 means "the route is right at least 90% of the time". So **don't copy your old `score_threshold` values across.** Fit first, then pick a bar from how often you can afford to be wrong.

You can also pass a one-off bar per call with `router(query, threshold=0.6)`. It wins over both the route's and the router's.

## fit() calibrates instead of searching

In 0.x, `fit()` ran a random search over per-route thresholds and returned nothing. In 1.0 it learns a curve from raw score to the probability that the top route is right, and returns a report.

**0.x**

```python
# 0.x
router.fit(X=queries, y=labels, max_iter=500)  # -> None
accuracy = router.evaluate(X=test_queries, y=test_labels)
```

**1.0**

```python
report = router.fit(X=held_out_queries, y=labels, method="platt")
# FitReport(method='platt', n=150, accuracy=0.93, ece=0.03)
accuracy = router.evaluate(X=test_queries, y=test_labels)
```

- **Fit on held-out queries, never on the utterances you gave the routes.** A stored utterance matches itself at a similarity near 1.0, so the calibrator learns that only near-perfect scores are trustworthy, and real queries look unsure. `IndexedRouter` warns if it sees stored utterances in `X`.
- A label of `None` still means "no route should match".
- `batch_size`, `max_iter` and `local_execution` are gone. `method` is `"platt"` (the default) or `"isotonic"`.
- `evaluate()` is unchanged: end-to-end accuracy with your thresholds applied.

The [calibration guide](../features/calibration) explains both methods and how much data each needs.

## RouteChoice has three fields

**0.x**

```python
# 0.x
class RouteChoice(BaseModel):
    name: Optional[str] = None
    function_call: Optional[List[Dict]] = None
    similarity_score: Optional[float] = None
```

**1.0**

```python
class RouteChoice(BaseModel):
    name: Optional[str] = None
    confidence: Optional[float] = None
    calibrated: bool = False
```

- `similarity_score` becomes `confidence`. Before `fit()` it's the raw score, as before; after, it's a probability.
- `calibrated` tells you which of the two you're looking at.
- `function_call` is gone with dynamic routes.

`limit` works as before: `limit=1` returns one `RouteChoice` (with `name=None` when nothing clears the bar), and `limit=None` returns every route that clears it, best first.

## Sync is gone

0.x kept a copy of your routes in the router and another in the index, and the `auto_sync` modes reconciled the two. 1.0 drops all of it. **The index is the single source of truth**: the router writes routes to it, and `router.routes` reads them back. Each record's id comes from its route and utterance, so writing the same routes twice changes nothing.

Removed: `auto_sync`, `sync()`, `async_sync()`, `is_synced()`, `get_utterance_diff()`, the index hash, lock and config records, and `SyncMode`, `UtteranceDiff` and `ConfigParameter` from `semantic_router.schema`.

**0.x**

```python
# 0.x
router = SemanticRouter(
    encoder=encoder,
    routes=routes,
    index=PineconeIndex(index_name="routes", dimensions=1536),
    auto_sync="local",
)
if not router.is_synced():
    router.sync("merge-force-local")
```

**1.0**

```python
router = IndexedRouter(
    encoder=encoder,
    routes=routes,
    index=PineconeIndex(index_name="routes", dimensions=1536),
)
```

Passing `routes=` adds or updates; it never deletes. Routes that are in the index but not in your list stay there, unlike `auto_sync="local"`. Remove a route with `router.remove(name=...)`.

To connect to an index that's already filled, leave `routes` out: `IndexedRouter(encoder=encoder, index=index)` reads the routes from the index, which replaces `auto_sync="remote"`. To see routes that other processes add later, call `router.refresh()` or set `refresh_interval`; see [sharing an index](../components/indexes#sharing-an-index).

Upgrading a remote index that 0.x filled? The cleanest path is to point 1.0 at a new index (or a new namespace) and let the router fill it from your routes. Reusing the old one also works:

- Record ids use the same scheme as 0.x, so adding your routes again updates the old records rather than duplicating them.
- 1.0 ignores the old sync records (the `sr_config` namespace on Pinecone, the `sr_config` collection on Qdrant). They're harmless, and you can delete them by hand.
- Routes read back without a description or threshold until you add them again.
- **Postgres:** rows written by 0.x's async `aadd()` have random ids, so they don't dedupe. If you used it, start a new table.

The [indexes guide](../components/indexes) covers sharing one index between routers.

## Managing routes

**0.x**

```python
# 0.x
router.add(Route(name="greetings", utterances=["hello there", "hi, how are you?"]))
router.delete(route_name="greetings")
router.get("billing")
router.list_route_names()
router.update(name="billing", utterances=new_utterances)
```

**1.0**

```python
router.add(routes=Route(name="greetings", utterances=["hello there", "hi, how are you?"]))
router.remove(name="greetings")
next(r for r in router.routes if r.name == "billing")
[r.name for r in router.routes]

# to change a route, replace it
router.remove(name="billing")
router.add(routes=Route(name="billing", utterances=new_utterances))
```

`add()` takes one route or a list. If the router is calibrated, `add()` and `remove()` refit it on the labelled queries you last passed to `fit()`, and warn you when a new route has no labelled queries yet.

## Dynamic routes and the LLM classes are gone

1.0 is a decision layer: it tells you *which* route applies, and your code decides what happens next. Dynamic routes, which generated function arguments with an LLM, are removed along with everything that supported them:

- `Route(function_schemas=..., llm=...)`, `Route.from_dynamic_route()`, and calling a `Route` directly
- the `llm` argument on routers, and `RouteChoice.function_call`
- the whole `semantic_router.llms` package: `BaseLLM`, `OpenAILLM`, `AzureOpenAILLM`, `CohereLLM`, `MistralAILLM`, `OllamaLLM`, `OpenRouterLLM` and `LlamaCppLLM`
- `semantic_router.utils.function_call` (`get_schema`, `get_schemas_openai`) and the `Message` schema

Route first, then call the tool with whatever LLM client you already use.

**0.x**

```python
# 0.x
from semantic_router.llms import OpenAILLM
from semantic_router.utils.function_call import get_schemas_openai

time_route = Route(
    name="get_time",
    utterances=["what time is it in rome?", "what's the time in new york?"],
    function_schemas=get_schemas_openai([get_time]),
)
router = SemanticRouter(encoder=encoder, routes=[time_route], llm=OpenAILLM())
choice = router("what's the time in tokyo?")
get_time(**choice.function_call[0]["arguments"])
```

**1.0**

```python
from openai import OpenAI

time_route = Route(
    name="get_time",
    utterances=["what time is it in rome?", "what's the time in new york?"],
)
router = IndexedRouter(encoder=encoder, routes=[time_route], confidence_threshold=0.5)

query = "what's the time in tokyo?"
if router(query).name == "get_time":
    # the router picked the tool; your LLM fills in its arguments
    response = OpenAI().chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": query}],
        tools=[{"type": "function", "function": get_time_schema}],
        tool_choice={"type": "function", "function": {"name": "get_time"}},
    )
```

Routing still saves you the expensive part: the LLM only runs when a tool is actually needed, and it never has to choose between tools.

## route_filter is gone

`limit=None` returns every route that clears its bar, best first, so you can filter the ranked list yourself.

**0.x**

```python
# 0.x
router("how's the weather?", route_filter=["chitchat", "weather"])
```

**1.0**

```python
allowed = {"chitchat", "weather"}
picks = [c for c in router("how's the weather?", limit=None) if c.name in allowed]
choice = picks[0] if picks else None
```

If you always route over the same subset, build a second router with only those routes. Routers can share one index.

## Other call arguments

A router call is now `router(query, limit=1, threshold=None)`.

- `vector=` is gone: routers take the text and embed it themselves.
- `simulate_static=` is gone with dynamic routes.
- `text=` is now the first positional argument, `query`.

**0.x**

```python
# 0.x
router(text="hello", simulate_static=True)
```

**1.0**

```python
router("hello")
```

## Saving and loading

A config file now says which kind of router it describes, in a `router_type` field, so one loader can rebuild any router: an `IndexedRouter`, a `DecisionRouter`, a `Chain` or a `Merge`. A fitted calibrator is saved too, so a loaded router is calibrated straight away.

**0.x**

```python
# 0.x
router.to_yaml("router.yaml")
router = SemanticRouter.from_yaml("router.yaml")

from semantic_router import RouterConfig
config = RouterConfig.from_file("router.yaml")
```

**1.0**

```python
from semantic_router import load_router

router.to_yaml(path="router.yaml")
router = load_router(path="router.yaml")
```

0.x files have no `router_type`, so `load_router` won't read them. Rebuild the routes from an old file in a few lines, then save in the new format:

```python
import yaml

from semantic_router import IndexedRouter, Route

with open("old_router.yaml") as f:
    old = yaml.safe_load(f)

routes = [
    Route(
        name=r["name"],
        utterances=r["utterances"],
        description=r.get("description"),
        metadata=r.get("metadata") or {},
    )
    for r in old["routes"]
]
router = IndexedRouter(encoder=encoder, routes=routes)
router.to_yaml(path="router.yaml")
```

Leave `score_threshold` behind and calibrate instead (see above). The [save and load guide](save-load-from-file) shows the new format.

## Async

The async methods keep their shape and gain the new ones: `acall`, `aadd`, `aremove`, `afit` and `aevaluate`.

**0.x**

```python
# 0.x
choice = await router.acall(text="hello")
await router.aadd(new_route)
await router.adelete(route_name="greetings")
```

**1.0**

```python
choice = await router.acall("hello")
await router.aadd(routes=new_route)
await router.aremove(name="greetings")
report = await router.afit(X=held_out_queries, y=labels)
```

Every sync method runs its async twin on one shared background event loop. That keeps HTTP connections pooled between calls and works inside notebooks. Inside your own async code, use the `a`-methods with `await`.

## Keyword arguments

Every example in these docs passes arguments by keyword, `IndexedRouter(encoder=encoder, routes=routes)` and `Chain(routers=[fast, smart])`, and we recommend you do the same. The query is the one exception: `router("...")`.

## Building on 1.0

A router is now anything that scores every route. Subclass `BaseRouter`, implement one async method, and you get calling, thresholds, calibration, evaluation and composition for free. See [routers](../components/routers#build-your-own-router).
