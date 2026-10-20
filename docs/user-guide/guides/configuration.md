---
title: Configuration
description: Control Semantic Router's logging with environment variables.
---

## Logging

Semantic Router uses Python's standard `logging` module. You control how much it says with an environment variable.

### Setting the log level

Two variables work. Set the library-specific one if you can:

```bash
export SEMANTIC_ROUTER_LOG_LEVEL=DEBUG
```

Or use the general one, which other libraries may share:

```bash
export LOG_LEVEL=WARNING
```

`SEMANTIC_ROUTER_LOG_LEVEL` wins if both are set. If neither is, the level is `INFO`.

### Levels

- `DEBUG` — detailed diagnostics.
- `INFO` — general progress (the default).
- `WARNING` — something looks off but still works.
- `ERROR` — something failed.
- `CRITICAL` — something failed badly.

### From Python

Set the variable *before* importing the library:

```python
import os
os.environ["SEMANTIC_ROUTER_LOG_LEVEL"] = "DEBUG"

from semantic_router import IndexedRouter, Route
# debug logs now show
```

Turn on `DEBUG` when you're chasing an encoder or index problem, want to see what a router is doing on each call, or need to understand what the library is doing under the hood.
