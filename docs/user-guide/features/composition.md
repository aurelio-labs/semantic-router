---
title: Chain and Merge
description: Build routers out of routers. Fall through from a fast router to a smart one, or average several opinions.
---

`Chain` and `Merge` are routers made of other routers. They take the same routes, return the same `RouteChoice`, and can be fitted, evaluated and nested like any other router.

- **`Chain`** is sequential. It asks each router in turn and stops at the first one that's confident. Use it to answer easy queries cheaply and spend more only on hard ones.
- **`Merge`** is parallel. It asks every router at once and averages their confidences. Use it to get a second opinion on every query.

## Chain: fast first, smart when unsure

The classic pairing is an `IndexedRouter`, which answers in milliseconds, in front of a `DecisionRouter`, which is slower but follows instructions:

```python
from semantic_router import Chain, DecisionRouter, IndexedRouter, Route
from semantic_router.decision_models import TypeSafeDecisionModel
from semantic_router.encoders import FastEmbedEncoder

routes = [
    Route(
        name="billing",
        utterances=["i want a refund", "why was i charged twice?", "where can I find my invoice?"],
        description="Money already paid or owed: refunds, charges, invoices.",
    ),
    Route(
        name="technical",
        utterances=["the app keeps crashing", "I can't log in", "the page won't load"],
        description="Something in the product is broken or not working.",
    ),
]

fast = IndexedRouter(
    encoder=FastEmbedEncoder(name="BAAI/bge-small-en-v1.5"),
    routes=routes,
    confidence_threshold=0.85,
)
fast.fit(X=held_out_queries, y=held_out_labels)  # so 0.85 means 85%
smart = DecisionRouter(model=TypeSafeDecisionModel(), routes=routes)

router = Chain(routers=[fast, smart])
router("I was billed for a plan I cancelled")
```

For each query, the Chain:

1. Asks `fast` to score the routes.
2. Takes `fast`'s top route and compares its confidence to that route's bar: the route's `confidence_threshold`, else `fast`'s, else the Chain's own.
3. If it clears the bar, answers from `fast`. `smart` never runs.
4. Otherwise moves on to `smart`. The last router always answers.

Measured with an OpenAI encoder and Jev, a confident query that stopped at the index took 265 ms at p50; an ambiguous one that fell through to Jev took 610 ms. Only the traffic that needs the decision model pays for it.

The same fast-then-slow split, with a decision model making the quick calls and an LLM setting the strategy, is the [thinking fast and slow](https://youtu.be/P-t4F_UlxkI?t=1542) chapter of the Jev video.

Two things to get right:

- **Give the first router a bar.** A router with no threshold (and no threshold on the Chain) always answers, so the routers after it never run.
- **Calibrate the first router.** Its bar is only meaningful if its confidence is a probability. If you fit it on its own stored utterances, real queries look unsure and nearly everything falls through. See [calibration](calibration#never-fit-on-the-stored-utterances).

## Merge: a second opinion on every query

```python
from semantic_router import Merge

router = Merge(routers=[fast, smart], weights=[1, 2])
router("I was billed for a plan I cancelled")
```

Merge runs every router at the same time and returns, for each route, the weighted mean of their confidences: `(1 × fast + 2 × smart) / 3` here. Without `weights`, every router counts equally. A call always takes as long as the slowest router.

Average only like with like. A raw cosine similarity and a probability are different units, so calibrate every router before you merge them.

## Chain or Merge?

| | Chain | Merge |
|---|---|---|
| Runs | one router after another, stopping early | every router, at once |
| Costs | the fast router for confident queries; all of them only when unsure | the slowest router, every time |
| Answer | one router's answer | a blend of all of them |
| Use it to | save time and money on easy queries | improve accuracy with a second opinion |

A Chain can't do what a Merge does: it never combines answers. And a Merge never saves you a call.

## Thresholds, calibration and nesting

`Chain` and `Merge` take a `confidence_threshold` like any router. It applies to the final answer, and in a Chain it's also the fallback bar for routers without one of their own.

A Chain or Merge reports `calibrated=True` only when every router in it is calibrated. You can also `fit()` the whole composition on labelled queries, like any router.

Both are routers, so they nest. Here, queries the fast router is unsure about get a blended answer from a hybrid router and the decision model:

```python
from semantic_router.encoders import BM25Encoder

hybrid = IndexedRouter(
    encoder=FastEmbedEncoder(name="BAAI/bge-small-en-v1.5"),
    sparse_encoder=BM25Encoder(),
    routes=routes,
)
hybrid.fit(X=held_out_queries, y=held_out_labels)

router = Chain(routers=[fast, Merge(routers=[hybrid, smart])])
```

Their routes are the routes of the first router you pass in. Give every router in a composition the same routes.
