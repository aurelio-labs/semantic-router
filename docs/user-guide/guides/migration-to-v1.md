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

- `RouteLayer.retrieve_multiple_routes()` → `SemanticRouter.__call__(limit=None)` or `SemanticRouter.acall(limit=None)`

  The `retrieve_multiple_routes` method has been removed. If you need similar functionality:
  
  - In versions 0.1.0-0.1.2: You can use the deprecated `_semantic_classify_multiple_routes` method
  - In version 0.1.3+ (0.1.5+ is recommended): Use the `__call__` or `acall` methods with appropriate `limit` parameter.
  
  ```python
  # Before (v0.0.x)
  route_layer = RouteLayer(encoder=encoder, routes=routes)
  multiple_routes = route_layer.retrieve_multiple_routes(query_text)
  
  # Transitional (v0.1.0-0.1.2)
  # Using deprecated method (not recommended)
  semantic_router = SemanticRouter(encoder=encoder, routes=routes, auto_sync="local")
  query_results = semantic_router._query(query_text)
  multiple_routes = semantic_router._semantic_classify_multiple_routes(query_results)
  
  # After (v0.1.3+)
  semantic_router = SemanticRouter(encoder=encoder, routes=routes, auto_sync="local")
  # Return all routes that pass their score thresholds
  all_routes = semantic_router(query_text, limit=None)
  # Or return top N routes that pass their score thresholds
  top_routes = semantic_router(query_text, limit=3)
  
  # To get scores for all routes regardless of threshold
  semantic_router.set_threshold(threshold=0.0)  # Set all route thresholds to 0
  all_route_scores = semantic_router(query_text, limit=None)
  ```
  
  When `limit=1` (the default), a single `RouteChoice` object is returned.
  When `limit=None` or `limit > 1`, a list of `RouteChoice` objects is returned.

  > **Important Note About `top_k`**: The `top_k` parameter (default: 5) can still limit the number of routes returned, regardless of the `limit` parameter. When using `limit > 1`, we recommend setting `top_k` to a higher value such as 100 or more. If you're using `limit=None` to get all possible results, make sure to set `top_k` to be equal to or greater than the total number of utterances shared across all of your routes.
  >
  > ```python
  > # Example: Setting top_k higher when retrieving multiple routes
  > semantic_router = SemanticRouter(encoder=encoder, routes=routes, top_k=100)
  > all_routes = semantic_router(query_text, limit=None)
  > ```

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