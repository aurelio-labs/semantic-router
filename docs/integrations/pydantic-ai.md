Semantic Router integrates with Pydantic AI to provide intelligent guardrails and routing for AI agents. This integration enables building safe, controlled agents with semantic understanding.

## Overview

Semantic Router can be used with Pydantic AI to:
- Add semantic guardrails to agent behavior
- Route queries to appropriate agent workflows
- Detect and block unwanted query types
- Control agent responses based on query classification

## Getting Started

### Prerequisites

1. Semantic Router with hybrid support
2. Pydantic AI version 0.0.42 or later

### Installation

```bash
pip install "semantic-router>=0.1.6" "pydantic-ai>=0.0.42"
```

## Use Cases

### 1. Guardrails with HybridRouter

Use `HybridRouter` to detect specific query types and control agent behavior:

```python
from semantic_router import Route
from semantic_router.routers import HybridRouter
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
encoder = OpenAIEncoder(name="text-embedding-3-small", score_threshold=0.3)
sparse_encoder = AurelioSparseEncoder(name="bm25")

router = HybridRouter(
    encoder=encoder,
    sparse_encoder=sparse_encoder,
    routes=[allowed, blocked],
    auto_sync="local"
)
```

### 2. Multi-Stage Routing

Chain multiple routers for complex guardrail logic:

```python
from pydantic_ai import Agent
from pydantic_graph import BaseNode, Graph, End

# First router: topic classification
topic_router = HybridRouter(encoder=encoder, sparse_encoder=sparse_encoder, routes=topic_routes)

# Second router: safety check
safety_router = HybridRouter(encoder=encoder, sparse_encoder=sparse_encoder, routes=safety_routes)

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
        result = topic_router(text=ctx.state.query)
        if result.name == "allowed":
            return Respond()
        else:
            ctx.state.response = "Sorry, I can only help with product questions."
            return End(ctx.state.response)

@dataclass
class Respond(BaseNode[GraphState, None, str]):
    async def run(self, ctx: GraphRunContext) -> End[str]:
        ctx.state.response = agent.run_sync(user_prompt=ctx.state.query)
        return End(ctx.state.response)

graph = Graph(name="Support Agent", nodes=(CheckTopic, Respond))
```

## Why Hybrid Router?

For agent guardrails, `HybridRouter` significantly outperforms dense-only routing:
- Better at distinguishing similar queries with different keywords
- Combines semantic meaning with term matching
- Example: "Can I sell my Tesla?" vs "Can I sell my BYD?" - semantically identical but different entities

```python
# Dense encoder alone struggles with similar queries
"Can I sell my Tesla?" ’ "Can I sell my BYD?"
# Similarity: 0.65+ (very similar!)

# Hybrid router correctly distinguishes via sparse matching
router("Can I sell my Tesla?").name  # -> "tesla"
router("Can I sell my BYD?").name    # -> "byd"
```

## Best Practices

1. **Use HybridRouter**: For guardrails, hybrid routing provides 10-20% better accuracy

2. **Threshold Optimization**: Use `router.fit()` to optimize thresholds on your data:
```python
router.fit(X=test_queries, y=expected_routes)
```

3. **Include None Routes**: Train with queries that shouldn't match any route:
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
- [Hybrid Router Guide](../user-guide/components/routers.md)
