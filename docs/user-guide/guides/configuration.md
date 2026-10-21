> **You're reading the docs for Semantic Router 0.x.** Version 1.0 is out: read [this page for v1](https://docs.aurelio.ai/docs/semantic-router/v1/user-guide/guides/configuration), or the [migration guide](https://docs.aurelio.ai/docs/semantic-router/v1/user-guide/guides/migration-to-v1) to upgrade.

# Configuration

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

from semantic_router import Route, SemanticRouter
# debug logs now show
```

Turn on `DEBUG` when you're chasing an encoder or index problem, want to see why a query matched the route it did, or need to understand what the library is doing under the hood.
