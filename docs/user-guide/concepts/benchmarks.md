---
title: Benchmarks
description: Accuracy, calibration and latency of every router type on CLINC150, measured the same way.
---

How the router types compare on one task, all measured the same way by [`benchmarks/accuracy.py`](https://github.com/aurelio-labs/semantic-router/blob/main/benchmarks/accuracy.py). The raw results are in [`benchmarks/results/accuracy.json`](https://github.com/aurelio-labs/semantic-router/blob/main/benchmarks/results/accuracy.json).

## The task

[CLINC150](https://github.com/clinc/oos-eval) is an intent dataset with an out-of-scope class: queries that match none of the intents, which a router should turn away.

- **Routes:** 15 intents picked at random (seed 0). Each route gets 10 of the intent's training utterances and a one-line description written from the intent's name alone.
- **Fitting:** every router is [calibrated](../features/calibration) on 400 validation queries, 300 in scope and 100 out of scope.
- **Testing:** 600 held-out test queries, 450 in scope and 150 out of scope. A router answers when its calibrated confidence is at least 0.5, and returns no route otherwise.

The intents are book_hotel, direct_deposit, ingredient_substitution, maybe, mpg, nutrition_info, pto_used, report_fraud, restaurant_suggestion, routing, timer, tire_change, transactions, transfer and yes.

## Accuracy and calibration

| Router | Top-1 | End-to-end | In scope | Out of scope turned away | Calibration error |
|---|---|---|---|---|---|
| `IndexedRouter`, OpenAI `text-embedding-3-small` | 97.3% | 94.5% | 96.4% | 88.7% | 0.033 |
| `IndexedRouter`, FastEmbed `bge-small-en-v1.5` | 97.3% | 92.0% | 95.3% | 82.0% | 0.042 |
| `IndexedRouter`, hybrid (OpenAI and TF-IDF) | 94.9% | 89.3% | 91.1% | 84.0% | 0.055 |
| `DecisionRouter`, Jev | 99.3% | 97.8% | 98.7% | 95.3% | 0.017 |
| `DecisionRouter`, GLiNER2.5-Decide (local) | 97.6% | 93.8% | 92.7% | 97.3% | 0.029 |
| `Chain`: OpenAI index, then Jev | 98.0% | 96.2% | 97.6% | 92.0% | 0.005 |
| `Merge`: OpenAI index and Jev | 99.3% | 98.3% | 98.9% | 96.7% | 0.011 |
| `Chain`: FastEmbed index, then Jev | 98.2% | 95.8% | 98.0% | 89.3% | 0.013 |
| `Merge`: FastEmbed index and Jev | 99.3% | 98.3% | 98.9% | 96.7% | 0.018 |

- **Top-1:** of the in-scope queries, how often the highest-scoring route is the right one, before any threshold.
- **End-to-end:** of all 600 test queries, how often the router does the right thing at the 0.5 bar: the right route for an in-scope query, and no route for an out-of-scope one.
- **In scope:** end-to-end on the 450 in-scope queries alone.
- **Out of scope turned away:** of the 150 out-of-scope queries, how many get no route.
- **Calibration error:** the expected calibration error (ECE) after `fit()`: the average gap between the confidence a router states and how often it's right. 0 is perfect.

Before `fit()`, the single routers' raw scores are much further off:

| Router | Calibration error before `fit()` |
|---|---|
| `IndexedRouter`, OpenAI | 0.314 |
| `IndexedRouter`, FastEmbed | 0.222 |
| `IndexedRouter`, hybrid | 0.327 |
| `DecisionRouter`, Jev | 0.150 |
| `DecisionRouter`, GLiNER2.5-Decide | 0.221 |

A `Chain` or `Merge` is built from routers that are already fitted.

## Latency

| Router | p50 | p95 | Throughput |
|---|---|---|---|
| `IndexedRouter`, OpenAI | 394 ms | 608 ms | 35 queries/s |
| `IndexedRouter`, FastEmbed | 3 ms | 3 ms | 978 queries/s |
| `IndexedRouter`, hybrid | 340 ms | 451 ms | 43 queries/s |
| `DecisionRouter`, Jev | 257 ms | 434 ms | 26 queries/s |
| `Chain`: OpenAI index, then Jev | 454 ms | 793 ms | 25 queries/s |
| `Merge`: OpenAI index and Jev | 345 ms | 577 ms | 25 queries/s |
| `Chain`: FastEmbed index, then Jev | 3 ms | 336 ms | 52 queries/s |
| `Merge`: FastEmbed index and Jev | 263 ms | 538 ms | 29 queries/s |

p50 and p95 are over 30 single queries, after two warm-up calls. Throughput is a batch of 50 queries scored at once. These were timed on 15 October 2026 on a quiet 32-core Linux machine, against the live OpenAI and Jev APIs, so the API-backed numbers include that machine's network round trip. Yours will differ.

The OpenAI chain answered 66.8% of the test queries from the index alone (87.6% of the in-scope ones) and sent the rest to Jev.

GLiNER2.5-Decide was timed separately, on an Apple M4 Max on 17 October 2026:

| Setup | p50 | Throughput |
|---|---|---|
| MPS, the benchmark's 15 described routes | 211 ms | 5 queries/s |
| MPS, 2 short routes | 50 ms | 47 queries/s |
| CPU, the benchmark's 15 described routes | 328 ms | 2 queries/s |

Its latency grows with the number of routes and the length of their descriptions, because every route goes through the model in one pass with the query.

## Reading the results

- **Most accurate:** Jev, alone or merged with an index. Jev gets 99.3% top-1, and the merges reach 98.3% end-to-end.
- **Fastest:** a FastEmbed index answers in 3 ms, with the same top-1 as OpenAI embeddings (97.3%). It turns away fewer out-of-scope queries, though (82.0% against 88.7%).
- **Fast and accurate:** a `Chain` answers most queries from the index and pays for Jev only on the rest. The FastEmbed chain has a 3 ms p50 and 95.8% end-to-end.
- **Local decision model:** GLiNER2.5-Decide gets 97.6% top-1 with no API key, and turns away the most out-of-scope queries (97.3%). It also turns away more in-scope ones (92.7%) at the 0.5 bar. A bar of 0.3, picked on the validation set, lifts its end-to-end to 95.0%.
- **Hybrid:** adding TF-IDF lowers accuracy here, because none of these routes hinge on exact keywords. See [routers](../components/routers) for when sparse helps.
- **Calibration:** every router's raw scores are off by 0.15 to 0.33. After `fit()`, all of them are within 0.06. Fit before you set a threshold.

## Run it yourself

```bash
uv run python benchmarks/accuracy.py --dry-run  # prints the API budget, then stops
uv run python benchmarks/accuracy.py
```

It needs `OPENAI_API_KEY` and `JEV_API_KEY`. Jev's answers are cached in `benchmarks/.cache/`, keyed on the whole request, so a rerun only spends Jev calls on latency. `--latency none` measures accuracy alone, and `--latency decision_jev,chain` re-times only those routers.

GLiNER2.5-Decide needs the `local` extra, so run it on Python 3.12:

```bash
uv run --python 3.12 --extra local --extra fastembed python benchmarks/accuracy.py
```
