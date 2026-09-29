---
title: Versions
description: Which Semantic Router release line to use, and where its docs live.
---

Semantic Router has two release lines. Pick the one that matches the version you have installed, and use the version switcher at the top of the sidebar to read the matching docs.

## v0: the 0.x line

The 0.x releases are the library as it has shipped since 2023: `SemanticRouter`, `HybridRouter`, utterance-based routes, `score_threshold`, and index synchronisation. The final release of this line is **0.2.0**, which is being finished on the `v0` branch. After that the line receives bug fixes only.

Install it with an upper bound so an upgrade never moves you to 1.x by accident:

```bash
pip install "semantic-router<1"
```

The 0.x source lives on the [`v0` branch](https://github.com/aurelio-labs/semantic-router/tree/v0). Bug reports and fixes for 0.x should target that branch.

## v1: the 1.x line

The 1.x line is a rewrite of the routing layer around calibrated confidence, decision-model backends, and composable routers. It is being developed on the [`main` branch](https://github.com/aurelio-labs/semantic-router) and will ship as **1.0.0**. It is a breaking change: route definitions, thresholds, and the router classes all move. A migration guide will be published with the release.

Pre-releases, once available, install with:

```bash
pip install --pre "semantic-router>=1.0.0.dev0"
```

## Which one should I use?

- **Running 0.x in production?** Stay on the v0 docs. Nothing changes for you until you choose to migrate.
- **Starting something new?** Use 0.x today. Move to 1.x when 1.0.0 ships, unless you want to follow development on `main`.
- **Contributing?** Fixes for 0.x go to `v0`; everything else goes to `main`. See [Contributing](https://github.com/aurelio-labs/semantic-router/blob/main/CONTRIBUTING.md).
