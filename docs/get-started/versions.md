---
title: Versions
description: Which Semantic Router release line to use, and where its docs live.
---

Semantic Router has two release lines. Pick the one that matches the version you have installed, and use the version switcher to read the matching docs.

## v1: the 1.x line

You're reading the v1 docs. Version 1 rebuilds the routing layer around three ideas:

- **Calibrated confidence.** `fit()` turns raw scores into probabilities, so a threshold means what it says.
- **More than one kind of router.** `IndexedRouter` matches by embedding, `DecisionRouter` asks a decision model, and every router follows the same contract.
- **Composition.** `Chain` and `Merge` build routers out of routers.

It's a breaking release: the router classes, thresholds and sync all change, and dynamic routes are gone. The [migration guide](../user-guide/guides/migration-to-v1) covers every change, with before and after code.

```bash
pip install -qU "semantic-router>=1,<2"
```

The 1.x source lives on the [`main` branch](https://github.com/aurelio-labs/semantic-router).

## v0: the 0.x line

The 0.x releases are the library as it shipped from 2023 to 2026: `SemanticRouter`, `HybridRouter`, `score_threshold`, dynamic routes and index sync. The final release of this line is **0.2.0**. From then on it gets bug fixes only, on the [`v0` branch](https://github.com/aurelio-labs/semantic-router/tree/v0).

Install it with an upper bound, so an upgrade never moves you to 1.x by accident:

```bash
pip install -qU "semantic-router<1"
```

The [v0 docs](../../v0/get-started/introduction) stay online.

## Which one should I use?

- **Starting something new?** Use 1.x.
- **Running 0.x in production?** Nothing changes until you choose to upgrade. Pin `semantic-router<1`, keep using the v0 docs, and move over with the [migration guide](../user-guide/guides/migration-to-v1) when you're ready.
- **Contributing?** Fixes for 0.x go to `v0`; everything else goes to `main`. See [Contributing](https://github.com/aurelio-labs/semantic-router/blob/main/CONTRIBUTING.md).
