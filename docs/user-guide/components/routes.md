---
title: Routes
description: A route is one decision a router can make, and what tells it when to make it.
---

A `Route` is one decision a router can make: "this is a billing question", "this is small talk", "this needs a human". Every router type takes the same `Route` objects, so you can define your routes once and use them with an `IndexedRouter`, a `DecisionRouter`, or both.

## Fields

```python
from semantic_router import Route

billing = Route(
    name="billing",
    utterances=[
        "i want a refund",
        "why was i charged twice?",
        "where can I find my invoice?",
    ],
    description="Money already paid or owed: refunds, charges, invoices.",
    confidence_threshold=0.9,
    metadata={"team": "payments"},
)
```

| Field | What it's for | Read by |
|---|---|---|
| `name` | The route's identifier. It's what the router returns. | every router |
| `utterances` | Example queries that belong to the route. | `IndexedRouter` embeds them; `DecisionRouter` reads the first five when there's no description |
| `description` | What the route covers, stated literally. | `DecisionRouter` |
| `confidence_threshold` | The confidence this route needs before a router returns it. Overrides the router's own threshold. | every router |
| `metadata` | Anything else you want to keep with the route, such as an owner or a handler name. Stored with the route in the index. | you |

Only `name` is required. An `IndexedRouter` needs utterances to match against; a `DecisionRouter` works from a description alone.

## Utterances

Utterances are examples, not keywords. Write them the way your users actually talk, and cover the different ways people ask for the same thing. Five to twenty per route is a good start. More helps when a route is broad or its neighbours are close.

Keep the queries you use to *test* or *calibrate* a router separate from the utterances. A router always matches its own utterances well, so testing on them tells you nothing. See [calibration](../features/calibration).

## Descriptions

A decision model reads the description and nothing else, so say exactly what belongs in the route. These models take you literally and don't fill in gaps:

```python
# vague: the model has to guess where the edges are
Route(name="billing", description="Billing stuff.")

# literal: says what's in, and what's next door
Route(
    name="billing",
    description="Money already paid or owed: refunds, charges, invoices. Not pricing questions before a purchase.",
)
```

`IndexedRouter` doesn't embed the description. Descriptions aren't phrased like user queries, so mixing them in with the utterances would skew the scores. It does store the description with the route, so a router that reads the routes back from the index gets it too.

## Thresholds

A route's `confidence_threshold` is the bar its confidence must clear before a router returns it. When a route doesn't set one, the router's `confidence_threshold` applies. When neither is set, there's no bar, and the best route always wins.

Use a route-level bar when some decisions are costlier to get wrong than others:

```python
routes = [
    # cheap to be wrong
    Route(name="flag_possible_bug", utterances=["this looks like a bug", "is this an off-by-one?"], confidence_threshold=0.6),
    # costly to be wrong
    Route(name="refund_customer", utterances=["refund this customer", "give them their money back"], confidence_threshold=0.95),
]
```

These numbers mean most once the router is calibrated: then 0.95 means "right at least 95% of the time". See [calibration](../features/calibration).

## What a router returns

A router call returns a `RouteChoice`:

```python
class RouteChoice(BaseModel):
    name: Optional[str] = None    # the route, or None if nothing cleared its bar
    confidence: Optional[float] = None
    calibrated: bool = False      # True when confidence is a probability
```

`confidence` is a probability when `calibrated` is `True`: after `fit()`, or from a router like `DecisionRouter` whose scores are already probabilities. Otherwise it's the router's raw score, such as a cosine similarity.

## Saving routes

`Route.to_dict()` and `Route.from_dict()` round-trip a route through a plain dict. To save a whole router, routes included, see [save and load](../guides/save-load-from-file).
