Semantic Router integrates with NVIDIA NIM (NVIDIA Inference Microservices) embedding models through the `NimEncoder` class. This integration provides access to NVIDIA's optimized embedding models for high-performance semantic search.

## Overview

The `NimEncoder` enables semantic routing using NVIDIA NIM embedding models. NIM exposes
an OpenAI-compatible embeddings endpoint, so the encoder drives it with the OpenAI SDK
pointed at `https://integrate.api.nvidia.com/v1` — it does not depend on LiteLLM. It
supports both synchronous and asynchronous operations.

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
    name="nvidia/nv-embedqa-e5-v5",
    score_threshold=0.4,
    api_key=os.environ["NVIDIA_NIM_API_KEY"]
)
```

Model names no longer need the `nvidia_nim/` prefix that LiteLLM required. The key comes
from the `api_key` parameter or the `NVIDIA_NIM_API_KEY` environment variable; if neither
is set, the constructor raises a `ValueError`.

### Self-Hosted NIM

To target a NIM container you run yourself, pass `base_url` or set
`NVIDIA_NIM_API_BASE`:

```python
encoder = NimEncoder(
    name="nvidia/nv-embedqa-e5-v5",
    base_url="http://localhost:8000/v1",
    api_key="not-used-but-required",
)
```

## Features

### Supported Models

The `NimEncoder` supports NVIDIA NIM embedding models:
- `nvidia/nv-embedqa-e5-v5` - Optimized QA embeddings (1024 dimensions)
- `nvidia/nv-embed-v1` - General-purpose embeddings
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

### Passing NIM Options

Standard OpenAI embedding parameters such as `dimensions` are forwarded as-is. Options
specific to NIM — `input_type`, `truncate` — travel in `extra_body`:

```python
embeddings = encoder.encode_queries(
    ["your text here"],
    extra_body={"truncate": "END"},
)
```

### Input Types

NVIDIA's `nv-embedqa` models are asymmetric — they expect a different `input_type` for
search queries than for the documents being searched. The encoder picks the right one for
you:
- `passage` - used by `encode_documents()`, for indexing route utterances
- `query` - used by `encode_queries()`, `__call__()` and `acall()`, for incoming queries

Anything you supply in `extra_body` is merged over that default, so
`extra_body={"input_type": "passage"}` will override it if you need to.

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
    name="nvidia/nv-embedqa-e5-v5",
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
