=Routers are the core components of Semantic Router that actually perform the intelligent routing of text to the appropriate handlers. They combine encoders and indexes to create a powerful semantic classification system.

## Understanding Routers

In Semantic Router, routers serve several key functions:

1. **Process incoming queries** into semantic representations
2. **Match queries to routes** based on similarity
3. **Make decisions** about which handler should process a query
4. **Manage routes** (adding, updating, removing)
5. **Provide confidence scores** for routing decisions

The router is the main interface you'll interact with when using Semantic Router, as it brings together all the other components (routes, encoders, and indexes) into a cohesive system.

## Router Types

Semantic Router provides two main types of routers:

### SemanticRouter

The `SemanticRouter` is the standard router that uses dense embeddings to match queries to routes. It's the most common and widely used router type.

**Example usage**:

```python
from semantic_router.routers import SemanticRouter
from semantic_router.encoders import OpenAIEncoder
from semantic_router.index import LocalIndex
from semantic_router import Route
import os

# Set up API key
os.environ["OPENAI_API_KEY"] = "your-api-key"

# Create routes
routes = [
    Route(name="weather", utterances=["How's the weather?", "Is it raining?"]),
    Route(name="politics", utterances=["Tell me about politics", "Who's the president?"])
]

# Initialize the router
router = SemanticRouter(
    encoder=OpenAIEncoder(),
    routes=routes,
    index=LocalIndex()
)

# Use the router to route a query
result = router("What's the weather like today?")
print(result.name)  # "weather"
print(result.score) # e.g., 0.92
```

### HybridRouter

The `HybridRouter` uses both dense and sparse embeddings for a more balanced approach, combining semantic understanding with keyword matching. This can improve accuracy in many cases, especially where exact keyword matching is important.

**Example usage**:

```python
from semantic_router.routers import HybridRouter
from semantic_router.encoders import OpenAIEncoder, AurelioSparseEncoder
from semantic_router.index import HybridLocalIndex
from semantic_router import Route
import os

# Set up API keys
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
os.environ["AURELIO_API_KEY"] = "your-aurelio-api-key"

# Create routes
routes = [
    Route(name="weather", utterances=["How's the weather?", "Is it raining?"]),
    Route(name="politics", utterances=["Tell me about politics", "Who's the president?"])
]

# Initialize the router with both dense and sparse encoders
router = HybridRouter(
    encoder=OpenAIEncoder(),
    sparse_encoder=AurelioSparseEncoder(),
    routes=routes,
    index=HybridLocalIndex(),
    alpha=0.3  # Balance between dense (0) and sparse (1) embeddings
)

# Use the router 
result = router("What's the weather like today?")
print(result.name)  # "weather"
```

## Key Router Features

### Route Management

Routers make it easy to add, update, and manage routes:

```python
# Adding a new route
new_route = Route(name="greetings", utterances=["Hello there", "Hi, how are you?"])
router.add(new_route)

# Getting a route by name
greeting_route = router.get("greetings")

# Listing all route names
route_names = router.list_route_names()
```

### Threshold Control

Control the sensitivity of your router by setting score thresholds:

```python
# Global threshold for all routes
router = SemanticRouter(
    encoder=OpenAIEncoder(),
    routes=routes,
    score_threshold=0.75  # Only match if similarity is above 0.75
)

# Per-route threshold
weather_route = Route(
    name="weather", 
    utterances=["How's the weather?", "Is it raining?"],
    score_threshold=0.8  # Higher threshold for this specific route
)
```

### Asynchronous Operation

Both router types support asynchronous operation for improved performance in async environments:

```python
# Async routing
result = await router.acall("What's the weather like today?")

# Async route addition
await router.aadd(new_route)
```

### Loading from Configuration

Routers can be loaded from configurations for easier deployment across environments:

```python
# Create a router from a YAML configuration file
router = SemanticRouter.from_yaml("router_config.yaml")

# Or from a RouterConfig object
from semantic_router.routers import RouterConfig

config = RouterConfig(routes=routes, encoder_type="openai")
router = SemanticRouter.from_config(config)
```

## Considerations for Choosing a Router

When selecting a router for your application, consider:

1. **Accuracy requirements**: HybridRouter typically provides better accuracy by combining semantic and keyword matching
2. **Performance needs**: SemanticRouter is more lightweight and can be faster
3. **Query characteristics**: If your queries often contain specific keywords, HybridRouter may perform better
4. **Resource constraints**: HybridRouter requires more computational resources
5. **Infrastructure**: Make sure you have the required API keys for the encoders used by your selected router

## Advanced Usage: Auto-Sync

Both router types support syncing routes between local and remote indexes:

```python
# Initialize router with auto-sync to remote index
router = SemanticRouter(
    encoder=encoder,
    routes=routes,
    index=remote_index,
    auto_sync="remote"  # Options: "local", "remote", None
)
```

## Advanced Usage: Hybrid Alpha

The `HybridRouter` allows fine-tuning the balance between dense and sparse embeddings:

```python
# More weight to dense embeddings (semantic matching)
router = HybridRouter(
    encoder=OpenAIEncoder(),
    sparse_encoder=AurelioSparseEncoder(),
    routes=routes,
    alpha=0.2  # 80% dense, 20% sparse
)

# Equal weight to both
router = HybridRouter(
    encoder=OpenAIEncoder(),
    sparse_encoder=AurelioSparseEncoder(),
    routes=routes,
    alpha=0.5  # 50% dense, 50% sparse
)

# More weight to sparse embeddings (keyword matching)
router = HybridRouter(
    encoder=OpenAIEncoder(),
    sparse_encoder=AurelioSparseEncoder(),
    routes=routes,
    alpha=0.8  # 20% dense, 80% sparse
)
```

## Router Return Values

When you call a router with a query, it returns a `RouteChoice` object with these key attributes:

- `name`: The name of the matched route (or empty string if no match)
- `score`: The confidence score of the match
- `function_schema`: Optional function schema associated with the route
- `metadata`: Any additional metadata associated with the route

For detailed information on routers and their configuration options, refer to the API documentation. 