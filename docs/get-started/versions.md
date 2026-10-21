---
title: Versions
description: Which Semantic Router release line to use, and where its docs live.
---

> **You're reading the docs for Semantic Router 0.x.** Version 1.0 is out: read [this page for v1](https://docs.aurelio.ai/docs/semantic-router/v1/get-started/versions), or the [migration guide](https://docs.aurelio.ai/docs/semantic-router/v1/user-guide/guides/migration-to-v1) to upgrade.

Semantic Router has two release lines. Pick the one that matches the version you have installed, and use the version switcher at the top of the sidebar to read the matching docs.

## v0: the 0.x line

The 0.x releases are the library as it has shipped since 2023: `SemanticRouter`, `HybridRouter`, utterance-based routes, `score_threshold`, and index synchronisation. The final feature release of this line is **0.3.0**. After that it receives bug fixes only.

Install it with an upper bound so an upgrade never moves you to 1.x by accident:

```bash
pip install "semantic-router<1"
```

The 0.x source lives on the [`v0` branch](https://github.com/aurelio-labs/semantic-router/tree/v0). Bug reports and fixes for 0.x should target that branch.

## v1: the 1.x line

The 1.x line rebuilds the routing layer around calibrated confidence, decision-model backends, and composable routers. It shipped as **1.0.0** from the [`main` branch](https://github.com/aurelio-labs/semantic-router). It is a breaking change: route definitions, thresholds, and the router classes all move. The [migration guide](https://docs.aurelio.ai/docs/semantic-router/v1/user-guide/guides/migration-to-v1) covers every change, with before and after code.

```bash
pip install -qU "semantic-router>=1,<2"
```

## Which one should I use?

- **Starting something new?** Use 1.x, and read the [v1 docs](https://docs.aurelio.ai/docs/semantic-router/v1/get-started/introduction).
- **Running 0.x in production?** Nothing changes until you choose to migrate. Pin `semantic-router<1` and stay on these docs until then.
- **Contributing?** Fixes for 0.x go to `v0`; everything else goes to `main`. See [Contributing](https://github.com/aurelio-labs/semantic-router/blob/main/CONTRIBUTING.md).
