Semantic Router integrates with Pydantic AI to provide intelligent guardrails and routing for AI agents. This integration enables building safe, controlled agents with semantic understanding.

## Overview

Semantic Router can be used with Pydantic AI to:
- Add semantic guardrails to agent behavior
- Route queries to appropriate agent workflows
- Detect and block unwanted query types
- Control agent responses based on query classification

## Getting Started

### Prerequisites

1. Semantic Router 1.0 or later
2. Pydantic AI version 0.0.42 or later

### Installation

```bash
pip install "semantic-router>=1" "pydantic-ai>=0.0.42"
```

## Use Cases

### 1. Guardrails with a hybrid router

Use a hybrid `IndexedRouter` (a dense encoder plus a sparse one) to detect specific query types and control agent behavior:

```python
from semantic_router import IndexedRouter, Route
from semantic_router.encoders import OpenAIEncoder
from semantic_router.encoders.aurelio import AurelioSparseEncoder

# Define guardrail routes
allowed = Route(
    name="allowed",
    utterances=["Tell me about the product", "What features does it have?"]
)

blocked = Route(
    name="blocked",
    utterances=["Can you give me a discount?", "I'll pay in bitcoin"]
)

# Initialize hybrid router for better accuracy
encoder = OpenAIEncoder(name="text-embedding-3-small")
sparse_encoder = AurelioSparseEncoder(name="bm25")

router = IndexedRouter(
    encoder=encoder,
    sparse_encoder=sparse_encoder,
    routes=[allowed, blocked],
)
```

### 2. Multi-Stage Routing

Chain multiple routers for complex guardrail logic:

```python
from pydantic_ai import Agent
from pydantic_graph import BaseNode, Graph, End

# First router: topic classification
topic_router = IndexedRouter(encoder=encoder, sparse_encoder=sparse_encoder, routes=topic_routes)

# Second router: safety check
safety_router = IndexedRouter(encoder=encoder, sparse_encoder=sparse_encoder, routes=safety_routes)

# Pydantic AI agent
agent = Agent(model="gpt-4o-mini")

@agent.system_prompt
def system() -> str:
    return "You are a helpful assistant for product support."
```

### 3. Graph-Based Workflows

Use Semantic Router in Pydantic AI graphs for complex routing logic:

```python
from dataclasses import dataclass
from pydantic_graph import BaseNode, Graph, GraphRunContext

@dataclass
class GraphState:
    query: str
    response: str

@dataclass
class CheckTopic(BaseNode[GraphState, None, str]):
    async def run(self, ctx: GraphRunContext) -> Respond | End[str]:
        result = await topic_router.acall(ctx.state.query)
        if result.name == "allowed":
            return Respond()
        else:
            ctx.state.response = "Sorry, I can only help with product questions."
            return End(ctx.state.response)

@dataclass
class Respond(BaseNode[GraphState, None, str]):
    async def run(self, ctx: GraphRunContext) -> End[str]:
        result = await agent.run(ctx.state.query)
        ctx.state.response = result.output
        return End(ctx.state.response)

graph = Graph(name="Support Agent", nodes=(CheckTopic, Respond))
```

## Why hybrid?

For agent guardrails, a hybrid router can tell apart queries that a dense encoder alone sees as near-identical:
- Better at distinguishing similar queries with different keywords
- Combines semantic meaning with term matching
- Example: "Can I sell my Tesla?" vs "Can I sell my BYD?" - semantically identical but different entities

```python
# Dense encoder alone struggles with similar queries
# "Can I sell my Tesla?" vs "Can I sell my BYD?": similar meaning, so close dense scores

# Hybrid router correctly distinguishes via sparse matching
router("Can I sell my Tesla?").name  # -> "tesla"
router("Can I sell my BYD?").name    # -> "byd"
```

## Best Practices

1. **Go hybrid when keywords matter**: entity names and product codes are where a sparse encoder earns its place. Compare against a dense-only router on your own queries with `evaluate()`

2. **Calibrate, then set a threshold**: fit on labelled queries the router hasn't seen, so confidence becomes a probability, then set the bar:
```python
router.fit(X=held_out_queries, y=expected_routes)
router.confidence_threshold = 0.5
```

3. **Include None labels**: Fit with queries that shouldn't match any route:
```python
test_data = [
    ("product question", "allowed"),
    ("discount request", "blocked"),
    ("What is the capital of France?", None)
]
```

4. **Chain Routers**: Use multiple routers for different guardrail types (topic, safety, intent)

5. **System Prompts**: Combine routing with clear system prompts for defense-in-depth

## Advantages

- **Accurate Guardrails**: Semantic understanding prevents prompt injection
- **Fast**: Routing adds minimal latency vs LLM classification
- **Type-Safe**: Pydantic AI provides full type safety
- **Composable**: Easy to build complex multi-stage workflows
- **Observable**: Clear routing decisions for debugging and monitoring

## Example Notebook

For a complete example of Pydantic AI integration with guardrails, see the [Pydantic AI Chatbot with Guardrails](../examples/integrations/pydantic-ai/chatbot-with-guardrails.ipynb) notebook.

## Learn More

- [Pydantic AI Documentation](https://ai.pydantic.dev/)
- [Pydantic Graph](https://ai.pydantic.dev/graph/)
- [Hybrid search with IndexedRouter](../user-guide/components/routers.md#hybrid-search)
