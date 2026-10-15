---
title: Save and load routers
description: Save any router, calibration included, to a YAML or JSON file, and load it back with one function.
---

You can save a router to a file and load it back later. That's useful for shipping the same routes and settings between environments, keeping them in version control, or skipping a refit every time you start up.

One loader handles every kind of router. Each file says what it describes in a `router_type` field: `indexed`, `decision`, `chain` or `merge`.

## Save a router

```python
from semantic_router import IndexedRouter, Route
from semantic_router.encoders import FastEmbedEncoder

routes = [
    Route(
        name="politics",
        utterances=["isn't politics the best thing ever", "why don't you tell me about your political opinions"],
    ),
    Route(
        name="chitchat",
        utterances=["how's the weather today?", "how are things going?"],
        description="Small talk.",
        confidence_threshold=0.9,
    ),
]
router = IndexedRouter(
    encoder=FastEmbedEncoder(name="BAAI/bge-small-en-v1.5"),
    routes=routes,
    confidence_threshold=0.5,
)
router.fit(
    X=["who should I vote for?", "nice day isn't it", "what's 2+2?", "the election is rigged", "how have you been?", "translate cat to french"],
    y=["politics", "chitchat", None, "politics", "chitchat", None],
)

router.to_yaml(path="router.yaml")
```

`router.to_json(path="router.json")` writes the same thing as JSON. Here's the YAML:

```yaml
router_type: indexed
encoder:
  type: fastembed
  name: BAAI/bge-small-en-v1.5
sparse_encoder: null
alpha: 0.7
top_k: 5
aggregation: mean
index:
  type: local
confidence_threshold: 0.5
calibrator:
  method: platt
  params:
    a:
    - 9.789179445243244
    b:
    - -4.871250211949403
refresh_interval: null
routes:
- name: politics
  utterances:
  - isn't politics the best thing ever
  - why don't you tell me about your political opinions
  description: null
  confidence_threshold: null
  metadata: {}
- name: chitchat
  utterances:
  - how's the weather today?
  - how are things going?
  description: Small talk.
  confidence_threshold: 0.9
  metadata: {}
```

The encoder is saved by type and model name, and the index by type and its settings. The fitted calibrator is saved too. The routes are in the file because this router's index is in memory, so the file is the only place they can come back from.

## Load it back

```python
from semantic_router import load_router

router = load_router(path="router.yaml")
router("don't you love politics?")
```

```
[Out]: RouteChoice(name='politics', confidence=0.8947431378911611, calibrated=True)
```

`load_router` returns the class the file's `router_type` names: an `IndexedRouter`, a `DecisionRouter`, a `Chain` or a `Merge`. It rebuilds the encoder and the index, writes the routes to the index, and restores the calibrator, so the loaded router routes exactly like the one you saved, with no refit.

## Routers on a remote index

A remote index (Pinecone, Qdrant, Postgres) already holds the routes, and it's their single source of truth. So the file names the index and leaves the routes out:

```yaml
router_type: indexed
encoder:
  type: fastembed
  name: BAAI/bge-small-en-v1.5
sparse_encoder: null
alpha: 0.7
top_k: 5
aggregation: mean
index:
  type: qdrant
  dimensions: 384
  index_name: support-routes
  location: null
  url: http://localhost:6333
confidence_threshold: null
calibrator: null
refresh_interval: null
```

`load_router` connects to that index and reads the routes from it, so a loaded router always routes over what's stored now, not what was there when you saved.

A loaded calibrator doesn't come with the labelled queries it was fitted on. If you `add()` or `remove()` a route on a loaded router, the calibrator is dropped with a warning: call `fit()` again.

## No secrets in files

Files never hold API keys or connection strings. Encoders are saved as their type and model name, and indexes keep only settings that aren't secret, such as `index_name`, `namespace` and `metric`. When you load a file, the encoder, index and decision model read their keys from the environment, as they do when you build them in code: `OPENAI_API_KEY`, `PINECONE_API_KEY`, `JEV_API_KEY` and so on.

`load_router` refuses a file with an `api_key` written into it.

## Other router types

A `DecisionRouter` saves its model settings and routes:

```yaml
router_type: decision
model: jev-latest
base_url: https://api.typesafe.ai
instructions: Which route does the user's message belong to?
timeout: 10.0
retries: 3
concurrency: 8
confidence_threshold: null
calibrator: null
routes:
- name: billing
  utterances:
  - i want a refund
  description: 'Money already paid or owed: refunds, charges, invoices.'
  confidence_threshold: null
  metadata: {}
```

A `Chain` or `Merge` nests the routers it's made of, under `routers`, each with its own `router_type`. A `Merge` adds its `weights`. Trimmed, a Chain looks like this:

```yaml
router_type: chain
confidence_threshold: null
calibrator: null
routers:
- router_type: indexed
  encoder:
    type: fastembed
    name: BAAI/bge-small-en-v1.5
  index:
    type: local
  confidence_threshold: 0.85
  routes: [...]
- router_type: decision
  model: jev-latest
  routes: [...]
```

## What can't be saved

Saving fails straight away, rather than on load, for anything the loader couldn't rebuild: a custom router or index, and encoders that can't be rebuilt from a type and a name (your own encoder classes, `HFEndpointEncoder`, `LocalSparseEncoder`). For those, build the router in code.

Unknown fields are errors, so a typo in a hand-edited file fails loudly instead of being ignored.

## Files from 0.x

0.x files have no `router_type`, so `load_router` rejects them with an error that links the [migration guide](migration-to-v1#saving-and-loading). The guide shows how to rebuild the routes from an old file in a few lines and save them in the new format.
