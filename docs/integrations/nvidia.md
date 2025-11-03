Semantic Router integrates with NVIDIA NIM (NVIDIA Inference Microservices) embedding models through the `NimEncoder` class. This integration provides access to NVIDIA's optimized embedding models for high-performance semantic search.

## Overview

The `NimEncoder` is a subclass of `LiteLLMEncoder` that enables semantic routing using NVIDIA NIM embedding models. It supports both synchronous and asynchronous operations with built-in cost tracking.

## Getting Started

### Prerequisites

1. NVIDIA NIM API key ([https://build.nvidia.com/](https://build.nvidia.com/))
2. Semantic Router version 0.1.8 or later

### Installation

```bash
pip install "semantic-router>=0.1.8"
```

### Basic Usage

```python
import os
from semantic_router.encoders import NimEncoder

os.environ["NVIDIA_NIM_API_KEY"] = "your-api-key"

encoder = NimEncoder(
    name="nvidia_nim/nvidia/nv-embedqa-e5-v5",
    score_threshold=0.4,
    api_key=os.environ["NVIDIA_NIM_API_KEY"]
)
```

## Features

### Supported Models

The `NimEncoder` supports NVIDIA NIM embedding models:
- `nvidia_nim/nvidia/nv-embedqa-e5-v5` - Optimized QA embeddings (1024 dimensions)
- `nvidia_nim/nvidia/nv-embed-v1` - General-purpose embeddings
- Other NVIDIA NIM embedding models available on the platform

### GPU-Accelerated Inference

NVIDIA NIM models are optimized for GPU inference, providing:
- Low latency embedding generation
- High throughput for batch processing
- Efficient resource utilization

### Asynchronous Support

Full async/await support for high-throughput applications:

```python
# Synchronous encoding
embeddings = encoder(["your text here"])

# Asynchronous encoding
embeddings = await encoder.acall(["your text here"])
```

### Cost Tracking

Built-in cost tracking via LiteLLM integration:
- Automatic token counting
- Per-request cost calculation
- Model-specific pricing

## Integration with Routers

The `NimEncoder` works with both `SemanticRouter` and `HybridRouter`:

```python
from semantic_router.routers import SemanticRouter
from semantic_router.route import Route

routes = [
    Route(
        name="technical",
        utterances=["How does this work?", "Explain the architecture"]
    ),
    Route(
        name="support",
        utterances=["I need help", "Can you assist me?"]
    )
]

router = SemanticRouter(
    encoder=encoder,
    routes=routes,
    auto_sync="local"
)
```

## Best Practices

1. **Model Selection**: Choose the appropriate NIM model based on your use case (QA, general-purpose, etc.)

2. **API Key Management**: Store API keys securely using environment variables

3. **Score Threshold**: NVIDIA NIM embeddings typically work well with thresholds around 0.4

4. **Batch Processing**: Leverage GPU acceleration by processing batches of texts

5. **Latency**: NIM models are optimized for low latency - ideal for real-time applications

## Advantages

- **High Performance**: GPU-accelerated inference for fast embedding generation
- **Optimized Models**: NVIDIA-optimized models for specific tasks (QA, search, etc.)
- **Low Latency**: Suitable for real-time applications
- **Enterprise Ready**: NVIDIA's enterprise-grade infrastructure and support

## Example Usage

```python
from semantic_router.encoders import NimEncoder
from semantic_router.routers import SemanticRouter
from semantic_router.route import Route

encoder = NimEncoder(
    name="nvidia_nim/nvidia/nv-embedqa-e5-v5",
    score_threshold=0.4
)

routes = [
    Route(name="greeting", utterances=["hello", "hi", "hey"]),
    Route(name="goodbye", utterances=["bye", "goodbye", "see you"])
]

router = SemanticRouter(encoder=encoder, routes=routes, auto_sync="local")

print(router("hi there").name)  # -> greeting
```

## Example Notebook

For a complete example of using the NVIDIA NIM integration, see the [NVIDIA NIM Encoder Notebook](../encoders/nvidia_nim-encoder.ipynb).
