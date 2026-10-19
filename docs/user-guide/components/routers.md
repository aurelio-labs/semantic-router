---
title: Routers
description: IndexedRouter, DecisionRouter, Chain and Merge, and how to build your own router.
---

A router takes a query, scores every route, and hands back a decision. Version 1 has several kinds of router, and they all share one contract: each scores the routes its own way, and the base class does the rest. Thresholds, calibration, evaluation, sync and async calling, and adding and removing routes all work the same on every router.

For how they compare on accuracy, calibration and latency, see the [benchmarks](../concepts/benchmarks).

```python
from semantic_router import Chain, DecisionRouter, IndexedRouter, Merge
```

| Router | Scores routes by | Needs |
|---|---|---|
| [`IndexedRouter`](#indexedrouter) | similarity between the query and each route's utterances | an encoder; an index (in memory by default) |
| [`DecisionRouter`](#decisionrouter) | asking a decision model which route applies | a Jev API key (or another model with the same API) |
| [`Chain`](#chain-and-merge) | trying routers in order until one is confident | other routers |
| [`Merge`](#chain-and-merge) | averaging several routers' confidences | other routers |

## IndexedRouter

The `IndexedRouter` embeds every route's utterances once, stores them in an index, and at query time finds the stored utterances closest to the query. It's fast, it runs locally if your encoder does, and it's what most apps start with.

```python
from semantic_router import IndexedRouter, Route
from semantic_router.encoders import FastEmbedEncoder

routes = [
    Route(name="weather", utterances=["how's the weather?", "is it raining?", "will it be sunny tomorrow?"]),
    Route(name="politics", utterances=["tell me about politics", "who's the president?", "what do you think of the election?"]),
]

router = IndexedRouter(
    encoder=FastEmbedEncoder(name="BAAI/bge-small-en-v1.5"),
    routes=routes,
)

router("what's the weather like today?").name
```

```
[Out]: 'weather'
```

The constructor embeds the utterances and writes them to the index before it returns, so the router is ready to call.

| Argument | Default | What it does |
|---|---|---|
| `encoder` | required | The dense encoder that embeds utterances and queries. |
| `routes` | none | Routes to write to the index. Leave it out to use the routes already in the index. |
| `refresh_interval` | `None` | Seconds before the router re-reads its routes from the index on the next call, to see other writers' changes. `None` never re-reads; call `refresh()` instead. See [sharing an index](indexes#sharing-an-index). |
| `index` | in memory | Where the vectors live: `LocalIndex`, `PineconeIndex`, `QdrantIndex`, `PostgresIndex`, ... See [indexes](indexes). |
| `sparse_encoder` | `None` | Set it to make the router hybrid (below). |
| `alpha` | `0.7` | Hybrid only: the weight on the dense score; the sparse score gets `1 - alpha`. |
| `top_k` | `5` | How many stored utterances to fetch per query. |
| `aggregation` | `"mean"` | How a route's hits combine into its score: `"mean"`, `"max"` or `"sum"`. |
| `confidence_threshold` | `None` | The router's default bar. See [thresholds](#thresholds). |
| `calibrator` | `None` | A fitted `Calibrator`. Usually you call `fit()` instead. |

A route's score is the aggregate of its utterances among the `top_k` hits, and a route with no hits scores 0. If you ask for several routes with `limit`, raise `top_k` so that more routes get a chance to appear.

### Hybrid search

Pass a sparse encoder as well, and the router weighs exact keywords alongside meaning. That helps when queries carry terms that must match: product codes, names, jargon.

```python
from semantic_router.encoders import BM25Encoder, FastEmbedEncoder

hybrid = IndexedRouter(
    encoder=FastEmbedEncoder(name="BAAI/bge-small-en-v1.5"),
    sparse_encoder=BM25Encoder(),
    routes=routes,
    alpha=0.7,
)
```

Without an `index`, a hybrid router uses a `HybridLocalIndex`. The only remote index that stores sparse vectors is `PineconeIndex`; the dense-only indexes raise an error if you pair them with a `sparse_encoder`.

`alpha` sets the balance: 1.0 is dense only, 0.0 is sparse only. The default, 0.7, keeps meaning in charge and uses keywords as a boost. Top-1 accuracy with a local FastEmbed model (`bge-small-en-v1.5`) on `HybridLocalIndex`:

| Queries | Sparse encoder | 0.0 | 0.3 | 0.5 | 0.7 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|
| notebook 06, 59 queries | BM25 | .69 | .76 | .88 | **.93** | .95 | .95 |
| notebook 06, 59 queries | TF-IDF | .51 | .51 | .56 | **.64** | .83 | .92 |
| support benchmark, 15 queries | BM25 | .93 | 1.0 | 1.0 | **1.0** | 1.0 | 1.0 |
| support benchmark, 15 queries | TF-IDF | .87 | .80 | .80 | **.87** | .93 | .93 |

The dense encoder alone scores .95 and 1.0 on these sets, which have no keyword-heavy routes. Sparse earns its place when queries hinge on exact terms like product codes and names. Measure on your own queries with [`evaluate()`](../features/calibration#measure-it) before you lower `alpha`.

A *fitted* sparse encoder such as `BM25Encoder` learns its vocabulary from your utterances, so `add()` refits it and rebuilds the index.

## DecisionRouter

The `DecisionRouter` asks a decision model which route a query belongs to. You pass it the model, as you pass an encoder to an `IndexedRouter`. It needs no encoder and no index: the model reads each route's `description` and returns a probability for every route, so its confidences are calibrated from the start.

```python
from semantic_router import DecisionRouter, Route
from semantic_router.decision_models import TypeSafeDecisionModel

routes = [
    Route(name="billing", description="Money already paid or owed: refunds, charges, invoices."),
    Route(name="technical", description="Something in the product is broken or not working."),
]

router = DecisionRouter(model=TypeSafeDecisionModel(), routes=routes)  # reads JEV_API_KEY
router("why was I charged twice this month?")
```

```
[Out]: RouteChoice(name='billing', confidence=1.0, calibrated=True)
```

It's slower than an `IndexedRouter` (a network call of a few hundred milliseconds) but it understands instructions and edge cases that example utterances can't capture. It can also answer several questions about one message in a single call. See [decision models](../features/decision-models), and for background on the models themselves, [Understanding Jev](https://jamesbriggs.io/understanding-jev) and [the video](https://youtu.be/P-t4F_UlxkI).

## Chain and Merge

`Chain` and `Merge` are routers made of routers.

- **`Chain(routers=[fast, smart])`** tries `fast` first. If its best route clears its bar, that's the answer and `smart` never runs. Otherwise it moves on to `smart`. Easy queries cost only the fast router.
- **`Merge(routers=[a, b], weights=[1, 2])`** asks every router at once and averages their confidences per route. Every query gets a second opinion.

Both are routers themselves, so they nest, and you can `fit()` them. See [Chain and Merge](../features/composition).

## Calling a router

```python
router("what's the weather like today?")                 # the best route, as a RouteChoice
router("what's the weather like today?", limit=2)        # up to 2 routes, best first
router("what's the weather like today?", limit=None)     # every route that clears its bar
router("what's the weather like today?", threshold=0.8)  # a bar for this call only
```

With `limit=1` (the default) you get one `RouteChoice`, with `name=None` when nothing clears the bar. With any other `limit` you get a list, which is empty when nothing clears it.

Every method has an async twin:

```python
choice = await router.acall("what's the weather like today?")
```

The sync methods run their async twin on one shared background event loop. That keeps HTTP connections pooled between calls, and it works inside notebooks. Inside your own async code, use the `a`-methods with `await`: a sync call from code running on that shared loop raises an error instead of deadlocking.

## Thresholds

A route is returned only when its confidence clears its bar. The bar is, in order:

1. the `threshold=` you pass to the call, if any
2. the route's own `confidence_threshold`, if set
3. the router's `confidence_threshold`, if set
4. otherwise no bar: the best route always wins

Until you calibrate, a router's confidence is its raw score, so a bar is a raw similarity. After `fit()` it's a probability. The [calibration guide](../features/calibration) shows how.

## Managing routes

```python
router.add(routes=Route(name="greetings", utterances=["hello there", "hi, how are you?"]))
router.remove(name="greetings")
[r.name for r in router.routes]
```

`add()` takes one route or a list. On an `IndexedRouter`, `add()` embeds and writes the new utterances, and `remove()` deletes the route's records from the index. To change a route, remove it and add the new version.

If you've calibrated the router, `add()` and `remove()` refit it on the labelled queries you last gave `fit()`. A new route with no labelled queries gets a warning: call `fit()` again with some.

## Build your own router

A router is anything that scores every route. Subclass `BaseRouter` and implement one async method, `_scores`, which returns a raw score per route (higher means more likely). You get calling, `limit`, thresholds, calibration, `evaluate()`, `add()`/`remove()` and composition for free.

```python
from semantic_router import Route
from semantic_router.routers import BaseRouter


class KeywordRouter(BaseRouter):
    async def _scores(self, query: str) -> dict[str, float]:
        q = query.lower()
        return {
            r.name: float(any(str(u) in q for u in r.utterances))
            for r in self.routes
        }


keyword_router = KeywordRouter(
    routes=[
        Route(name="refund", utterances=["refund", "money back"]),
        Route(name="cancel", utterances=["cancel", "unsubscribe"]),
    ],
    confidence_threshold=0.5,
)
keyword_router("can I get my money back?")
```

```
[Out]: RouteChoice(name='refund', confidence=1.0, calibrated=False)
```

| Hook | Override when |
|---|---|
| `_scores(query)` | Always: the raw score for every route. |
| `_scores_many(queries)` | You can score a batch more cheaply than one at a time (`IndexedRouter` embeds a whole batch in one call). The default runs `_scores` concurrently, eight at a time. |
| `_on_add(routes)` / `_on_remove(name)` | You keep per-route state, such as an index. |
| `probabilistic = True` | Your raw scores are already probabilities, so results report `calibrated=True` without a fit. |
