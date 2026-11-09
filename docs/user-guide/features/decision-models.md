---
title: Decision models
description: Route with a decision model, and ask several typed questions about one message in a single call.
---

Some routing decisions depend on instructions rather than resemblance: "refunds and charges, but not pricing questions before a purchase". Example utterances struggle to draw lines like that. A decision model can: it's a small, fast model trained to answer typed questions about a piece of text, with a probability for every answer.

`DecisionRouter` routes with one. The model is a component you pass in, the way you pass an encoder to an `IndexedRouter`. The models that ship speak the `/v1/systemone` API that TypeSafe's **Jev** introduced, which nace.ai's **Drex**, **Laya** and **Von** also implement.

## Learn about decision models

New to decision models? Two walkthroughs by James Briggs explain what they are and where they fit:

- **Article:** [Understanding Jev and Where It Fits in the AI Stack](https://jamesbriggs.io/understanding-jev) covers what a decision model is and how it differs from an LLM, what a calibrated probability gives you, where to use one and where not to, and how open models such as GLiNER2.5 compare, with benchmarks.
- **Video:** [I Tested Jev AI Against LLMs and Open Source](https://youtu.be/P-t4F_UlxkI) (34 min) covers the same ground with live tests. Jump to [the three types of decision](https://youtu.be/P-t4F_UlxkI?t=266), [calibrated probability](https://youtu.be/P-t4F_UlxkI?t=395), [open-source alternatives](https://youtu.be/P-t4F_UlxkI?t=999) or [GLiNER](https://youtu.be/P-t4F_UlxkI?t=1228).

## Route with a decision model

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

There's no encoder and no index. The router turns your routes into one multiple-choice question, sends it with the query, and gets back a probability for every route. Those are already probabilities, so results say `calibrated=True` without a `fit()`.

The router takes:

| Argument | Default | What it does |
|---|---|---|
| `model` | required | The decision model that answers. |
| `routes` | `()` | The routes to choose between. |
| `instructions` | `"Which route does the user's message belong to?"` | The question asked about each query. |
| `confidence_threshold` | `None` | The router's default bar, as on every router. |

The model holds the connection settings:

| Argument | Default | What it does |
|---|---|---|
| `name` | `"jev-latest"` | The model to ask. |
| `api_key` | `JEV_API_KEY` | Your API key. |
| `base_url` | `"https://api.typesafe.ai"` | The server; requests go to `{base_url}/v1/systemone`. |
| `timeout` | `10.0` | Seconds per request. |
| `retries` | `3` | Retries on rate limits (429, 529), server errors and timeouts, with exponential backoff and jitter. |
| `concurrency` | `8` | Requests in flight when scoring a batch (`fit`, `evaluate`, `Chain`, `Merge`). Each query is one request, so this caps a batch's throughput: raise it as far as your rate limit allows. |

## Use Nace Drex

`NaceDecisionModel` is nace.ai's [Drex](https://console.nace.ai/docs/drex/guides/drex-max) with its defaults filled in: `drex-max` at `https://console.nace.ai`, with the key read from `NACE_API_KEY`. It takes the same `timeout`, `retries` and `concurrency`.

```python
from semantic_router.decision_models import NaceDecisionModel

router = DecisionRouter(model=NaceDecisionModel(), routes=routes)  # reads NACE_API_KEY
```

It answers the same `choice`, `noul` and `score` questions, so `aask()` works the same way, with up to 512 questions in one call. `drex-max` can also read images, but this model sends the text `state` only. To call the separately served `drex-max-v1.1` instead, pass `name="drex-max-v1.1"`. Files save it as `model: {type: nace, name: drex-max}`.

## Use another server

`TypeSafeDecisionModel` and `NaceDecisionModel` are Jev and Drex with their defaults filled in. For a Laya or Von deployment, or a server of your own that speaks `/v1/systemone`, use `SystemOneDecisionModel` and say where it is:

```python
from semantic_router.decision_models import SystemOneDecisionModel

model = SystemOneDecisionModel(name="my-model", base_url="http://localhost:8080")
router = DecisionRouter(model=model, routes=routes)
```

It sends a key only if you pass `api_key`, so a local server needs none.

To plug in a different kind of model, subclass `DecisionModel` and implement `aask(state, questions)`, which returns an answer for each question.

## Run a decision model locally

`GLiNER2DecisionModel` runs [GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide), Fastino's open decision model, in your own process. Your queries never leave your infrastructure, and there's no API key.

```bash
pip install "semantic-router[local]"  # Python 3.10 to 3.12
```

```python
from semantic_router import DecisionRouter
from semantic_router.decision_models import GLiNER2DecisionModel

router = DecisionRouter(model=GLiNER2DecisionModel(), routes=routes)
router.fit(X=held_out_queries, y=held_out_labels)

router("why was I charged twice this month?")
```

The first time it loads, it downloads the weights from Hugging Face (about 1.8 GB). It answers the same `choice`, `noul` and `score` questions as Jev, so `aask()` works the same way.

| Argument | Default | What it does |
|---|---|---|
| `name` | `"fastino/GLiNER2.5-Decide"` | The model on the Hugging Face Hub, or a local directory. |
| `device` | the first available | `"cuda"`, `"mps"` or `"cpu"`. |
| `batch_size` | `8` | Queries per forward pass when scoring a batch (`fit`, `evaluate`, `Chain`, `Merge`). |
| `classifier` | `None` | A `gliner2` classifier you loaded yourself, such as one with a fine-tuned adapter. |

On the [benchmark](../concepts/benchmarks) (CLINC150, 15 routes), it comes close to Jev:

| | Jev | GLiNER2.5-Decide |
|---|---|---|
| Top-1 accuracy | 99.3% | 97.6% |
| End-to-end, at a 0.5 bar | 97.8% | 93.8% |
| Out-of-scope queries turned away | 95.3% | 97.3% |
| Calibration error, before / after `fit()` | 0.150 / 0.017 | 0.221 / 0.029 |

Three things to know:

- **Fit it before you set a threshold.** Its raw scores rank the routes well, but they aren't calibrated: a 0.8 isn't right 80% of the time. [`fit()`](calibration) on held-out queries fixes that.
- **Its speed depends on your routes.** Every route's name and description go through the model in one pass with the query, so more routes and longer descriptions take longer. On an Apple M4 Max (MPS), a query takes about 50 ms with two short routes and about 210 ms with the benchmark's 15 described routes; on CPU, about 330 ms. Keep descriptions short, or put an `IndexedRouter` in front of it with a [Chain](composition).
- **It needs Python 3.12 or earlier**, like the other models in the `local` extra.

The [GLiNER chapter](https://youtu.be/P-t4F_UlxkI?t=1228) of the Jev video covers how it works and how it compares.

## Write literal descriptions

The model reads each route's `description`, and it takes you at your word. Say what belongs in the route and, where it helps, what doesn't:

```python
Route(
    name="billing",
    description="Money already paid or owed: refunds, charges, invoices. Not pricing questions before a purchase.",
)
```

A route with no description falls back to its first five utterances, joined together. That works, but a written description is usually better.

## Ask several questions in one call

A decision model can answer several questions about the same message in one pass, for about the cost of one. `aask()` sends the message (the `state`) once, with a dict of named questions, and returns an answer for each.

```python
from semantic_router.decision_models import choice, noul, score

answers = await router.aask(
    state="my card was charged twice and I need it fixed today or I'm cancelling",
    questions={
        "route": router.route_question(),
        "urgent": noul(instructions="Is this time-sensitive?", yes="needs action now", no="can wait"),
        "mood": score(instructions="How upset is the user?", levels=["calm", "annoyed", "angry"]),
    },
)
answers["route"]["choice"], answers["urgent"]["noul"], answers["mood"]["score"]
```

There are three kinds of question:

| Helper | Asks | Answer |
|---|---|---|
| `choice(instructions=, options=)` | which one of several options applies. `options` maps each option's name to a description. | `choice`, `confidence`, and `probabilities` per option |
| `noul(instructions=, yes=, no=)` | a yes/no question. `yes` and `no` describe what each answer means. | `noul`: the probability of yes |
| `score(instructions=, levels=)` | where the message sits on an ordered scale, lowest first. | `score` (the expected level, 0-based), `confidence`, `probabilities` per level, and a `legend` |

`router.route_question()` is the `choice` question the router asks for routing, built from your routes. The answers above look like this:

```json
{
  "route": {"type": "choice", "choice": "billing", "confidence": 1.0,
            "probabilities": {"billing": 1.0, "technical": 0.0}},
  "urgent": {"type": "noul", "noul": 0.96},
  "mood": {"type": "score", "score": 1.92, "confidence": 0.88,
           "legend": {"0": "calm", "1": "annoyed", "2": "angry"},
           "probabilities": {"0": 0.0, "1": 0.08, "2": 0.92}}
}
```

*Noul* is TypeSafe's name for a yes/no question, short for *Bernoulli*: one trial, two outcomes. The answer is the probability of yes, so you choose the cut-off.

So one call can route a message, check it against a guardrail and score its urgency, where 0.x needed a separate router for each.

`aask()` is async only. In a notebook you can `await` it directly; in a script, run it with `asyncio.run()`.

## Latency

A decision model is a network call. Measured against Jev (p50, after warm-up):

| Call | p50 |
|---|---|
| one question | 351 ms |
| three questions in one `aask()` | 327 ms |
| three separate calls, one after another | 1,012 ms |

- **Batch questions** into one `aask()` rather than fanning out calls.
- **Warm up at startup.** The first call to a model can take many seconds; later ones take a few hundred milliseconds. Make one call when your app starts, never on a user's first request.
- **Put a fast router in front.** A [Chain](composition) answers confident queries from an `IndexedRouter` in milliseconds and only sends the ambiguous ones to the decision model.

## Close the session

A `DecisionRouter`'s model keeps a pooled HTTP session, so repeated calls reuse the connection. Close it when you're done. After async calls (`acall`, `aask`), prefer:

```python
await router.aclose()
```

From sync code, `router.close()` does the same. A sync `close()` from inside a running event loop raises an error that tells you to `await router.aclose()` instead.
