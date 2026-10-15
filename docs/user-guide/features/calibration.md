---
title: Calibration and thresholds
description: Turn raw router scores into probabilities with fit(), then set thresholds that mean what they say.
---

A raw similarity of 0.8 means something different for every encoder, every set of routes and every kind of query. So a threshold set on raw scores is a guess, and it breaks when you change the encoder.

Calibration fixes that. `fit()` takes labelled example queries and learns a curve from a router's raw score to the probability that its top route is right. After that, a confidence of 0.85 means the route is right about 85% of the time, and a threshold is a policy you can reason about: "act when you're 90% sure, otherwise ask a human".

## Fit a router

You need some labelled queries the router hasn't seen. Label the ones that shouldn't match any route with `None`.

```python
from semantic_router import IndexedRouter, Route
from semantic_router.encoders import FastEmbedEncoder

routes = [
    Route(name="billing", utterances=["i want a refund", "why was i charged twice?", "where can I find my invoice?", "cancel my subscription and refund me", "you overcharged me"]),
    Route(name="technical", utterances=["the app keeps crashing", "I can't log in", "the page won't load", "I get an error when I upload a file", "sync isn't working"]),
    Route(name="chitchat", utterances=["how's the weather today?", "how are things going?", "lovely weather today", "the weather is horrendous", "let's go to the chippy"]),
]
router = IndexedRouter(encoder=FastEmbedEncoder(name="BAAI/bge-small-en-v1.5"), routes=routes)

X_train = [
    "can I get my money back?", "there's a double charge on my card", "send me last month's receipt", "I was billed the wrong amount",
    "the screen goes blank when I open settings", "password reset email never arrives", "the export button does nothing", "it says error 500",
    "nice day out, isn't it?", "how was your weekend?", "it's freezing outside", "how have you been?",
    "what's the capital of australia?", "write me a poem about cats", "recommend a good sci-fi book", "how many legs does a spider have?",
]
y_train = ["billing"] * 4 + ["technical"] * 4 + ["chitchat"] * 4 + [None] * 4

router.fit(X=X_train, y=y_train)
```

```
[Out]: FitReport(method='platt', n=16, accuracy=1.0, ece=0.13265465825903883)
```

`fit()` returns a `FitReport`:

| Field | Meaning |
|---|---|
| `method` | `"platt"` or `"isotonic"` |
| `n` | how many labelled queries it fitted on |
| `accuracy` | how often the top route was right, over the queries that have a route (before any threshold) |
| `ece` | expected calibration error: the average gap between stated confidence and actual accuracy. Lower is better; 0 is perfect. |

From now on, every result has `calibrated=True` and its `confidence` is a probability.

## Never fit on the stored utterances

Use queries the router has never seen: paraphrases, real traffic, anything that isn't in the routes' `utterances`. A stored utterance always finds itself in the index, at a similarity near 1.0, which real queries never reach. Fit on those, and the calibrator learns that only near-perfect scores are trustworthy, so real queries come out unsure. If you chain routers, too many of them then fall through to the slow one.

`IndexedRouter.fit()` warns when it sees stored utterances among your queries:

```
WARNING semantic_router 15/19 training queries are stored utterances; they match themselves at ~1.0, so the calibrator will under-rate real queries, which score lower. fit on held-out queries.
```

## Set thresholds

A threshold is the confidence a route must reach before the router returns it. With none set, the best route always wins, however weak, so set one before you rely on `name=None` as a fallback.

```python
router.confidence_threshold = 0.5

router("who won the football?")
```

```
[Out]: RouteChoice(name=None, confidence=None, calibrated=True)
```

Thresholds live in up to three places. The first one that's set wins:

1. **The call:** `router(query, threshold=0.8)` for a one-off bar.
2. **The route:** `Route(confidence_threshold=0.95)` for a decision that's costlier to get wrong.
3. **The router:** `IndexedRouter(confidence_threshold=0.5)`, or `router.confidence_threshold = 0.5`, as the default.

```python
review_routes = [
    # cheap to be wrong
    Route(name="flag_possible_bug", utterances=["this looks like a bug", "is this an off-by-one?"], confidence_threshold=0.6),
    # costly to be wrong
    Route(name="refund_customer", utterances=["refund this customer", "give them their money back"], confidence_threshold=0.95),
]
review_router = IndexedRouter(
    encoder=FastEmbedEncoder(name="BAAI/bge-small-en-v1.5"),
    routes=review_routes,
    confidence_threshold=0.8,  # for routes without their own bar
)
```

A router has one calibrator, fitted across all its routes, so a 0.95 bar on one route is only as honest as the calibration is on that route. Give every route some labelled queries.

## Measure it

`evaluate()` runs labelled queries through the router, thresholds and all, and returns the accuracy. A `None` label counts as right when no route clears its bar. Use a test set you didn't fit on.

```python
X_test = ["refund my order please", "why is my bill so high?", "the app froze again", "I can't upload my photo", "lovely sunny afternoon", "fancy a chat?", "who won the football?", "translate hello into french"]
y_test = ["billing", "billing", "technical", "technical", "chitchat", "chitchat", None, None]

router.confidence_threshold = None
router.evaluate(X=X_test, y=y_test)
```

```
[Out]: 0.75
```

With no threshold, the two off-topic queries still get a route, so they count as wrong. With a bar:

```python
router.confidence_threshold = 0.5
router.evaluate(X=X_test, y=y_test)
```

```
[Out]: 0.875
```

Raising the threshold trades coverage for precision: more queries fall back to "no route", and the ones that do route are more often right. Try a few values against your test set, and pick the one whose mistakes you can live with.

## Platt or isotonic

`fit()` takes `method="platt"` (the default) or `method="isotonic"`.

```python
router.fit(X=X_train, y=y_train, method="isotonic")
```

```
[Out]: FitReport(method='isotonic', n=16, accuracy=1.0, ece=0.05555555555555549)
```

**Platt scaling** fits an S-curve with two numbers, *a* and *b*:

P(right | s) = 1 / (1 + e^-(a·s + b))

*a* sets how steep the curve is and *b* where it crosses 50%. They're chosen to minimise log-loss on your (score, right?) pairs, with the targets softened slightly so a small class can't pin the curve at exactly 0 or 1.

**Isotonic regression** assumes only that a higher score is never worse. It sorts the scores and merges neighbouring blocks wherever accuracy drops (pool adjacent violators), so the curve is a staircase of observed accuracies.

| | Platt | Isotonic |
|---|---|---|
| Assumes | an S-shape | only "a higher score is never worse" |
| Learns | 2 numbers | one step per block, up to one per example |
| Needs | tens of labelled queries | hundreds, or it overfits into flat steps |
| Use when | by default | you have lots of data and the curve isn't S-shaped |

Don't read too much into a low `ece` on the data you fitted on: isotonic can follow small samples almost exactly. Check it with `evaluate()` on a held-out set.

Both methods are monotone, so calibration never changes which route ranks first. It only changes the number attached to it.

## When routes change

`add()` and `remove()` refit the calibrator on the labelled queries you last passed to `fit()`. Labels for a removed route become `None`. A new route has no labelled queries yet, so the router warns you to call `fit()` again with some.

A calibrator you passed in yourself, or loaded from a file, has no labelled queries to refit on. When the routes change, it's dropped with a warning, and confidences are raw again until you call `fit()`.

## Routers that are already calibrated

A `DecisionRouter` returns probabilities straight from its model, so its results say `calibrated=True` without a fit. You can still `fit()` one to correct a model that's systematically over- or under-confident on your traffic.

A `Chain` or `Merge` is calibrated only when every router in it is. Calibrate the parts first, so their confidences are on the same scale; otherwise a threshold or an average mixes raw similarities with probabilities.
