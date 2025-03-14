The v0.1 release of semantic router introduces several breaking changes to improve the API design and add new functionality. This guide will help you migrate your code to the new version.

## Key API Changes

### Module Imports and Class Renaming

- `from semantic_router import RouteLayer` → `from semantic_router.routers import SemanticRouter`
  
  The `RouteLayer` class has been renamed to `SemanticRouter` and moved to the `routers` module to better reflect its purpose and fit into the modular architecture.

### Method Signatures

- `SemanticRouter.add(route: Route)` → `SemanticRouter.add(routes: List[Route])`
  
  The `add` method now accepts a list of routes, making it easier to add multiple routes at once. However, it still supports adding a single route for backward compatibility.

  ```python
  # Before
  route_layer = RouteLayer(encoder=encoder)
  route_layer.add(route1)
  route_layer.add(route2)
  
  # After
  semantic_router = SemanticRouter(encoder=encoder)
  semantic_router.add([route1, route2])  # Add multiple routes at once
  semantic_router.add(route3)  # Still works for a single route
  ```

### Synchronization Strategy

- If expecting routes to sync between local and remote on initialization, use `SemanticRouter(..., auto_sync="local")`. 

  The `auto_sync` parameter provides control over how routes are synchronized between local and remote indexes. Read more about `auto_sync` and [synchronization strategies](../features/sync).

  Available synchronization modes:
  
  - `error`: Raise an error if local and remote are not synchronized.
  - `remote`: Take remote as the source of truth and update local to align.
  - `local`: Take local as the source of truth and update remote to align.
  - `merge-force-local`: Merge both local and remote keeping local as the priority.
  - `merge-force-remote`: Merge both local and remote keeping remote as the priority.
  - `merge`: Merge both local and remote, with local taking priority for conflicts.

  ```python
  # Example: Initialize with synchronization strategy
  semantic_router = SemanticRouter(
      encoder=encoder,
      routes=routes,
      index=PineconeIndex(...),
      auto_sync="local"  # Local routes will be used to update the remote index
  )
  ```

## Other Important Changes

### Router Configuration

The `RouterConfig` class has been introduced as a replacement for the `LayerConfig` class, providing a more flexible way to configure routers:

```python
from semantic_router.routers import RouterConfig

# Create configuration for your router
config = RouterConfig(
    routes=[route1, route2],
    encoder_type="openai",
    encoder_name="text-embedding-3-small"
)

# Initialize a router from config
semantic_router = SemanticRouter.from_config(config)
```

### Advanced Router Options

The modular architecture now provides access to different router types:

- `SemanticRouter`: The standard router that replaces the old `RouteLayer`
- `HybridRouter`: A router that can combine dense and sparse embedding methods
- `BaseRouter`: An abstract base class for creating custom routers

## Migration Example

```python
# Before (v0.0.x)
from semantic_router import RouteLayer, Route
from semantic_router.encoders import OpenAIEncoder

route = Route(name="example", utterances=["sample utterance"])
layer = RouteLayer(encoder=OpenAIEncoder())
layer.add(route)
result = layer("query text")

# After (v0.1.x)
from semantic_router import Route
from semantic_router.routers import SemanticRouter
from semantic_router.encoders import OpenAIEncoder

route = Route(name="example", utterances=["sample utterance"])
router = SemanticRouter(encoder=OpenAIEncoder())
router.add(route)  # Still works for a single route
result = router("query text")
``` 