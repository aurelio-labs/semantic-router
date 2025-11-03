Semantic Router supports local embedding models through the `LocalEncoder` class. This integration allows you to generate embeddings entirely on your machine using sentence-transformers, with no API keys or internet connection required.

## Overview

The `LocalEncoder` provides access to open-source embedding models from sentence-transformers. All computation happens locally on your CPU, CUDA GPU, or Apple Silicon (MPS), making it ideal for privacy-sensitive applications or offline deployments.

## Getting Started

### Prerequisites

1. Semantic Router with local extras
2. Sufficient disk space for model downloads

### Installation

```bash
pip install "semantic-router[local]"
```

### Basic Usage

```python
from semantic_router.encoders.local import LocalEncoder

encoder = LocalEncoder()  # Uses default model: BAAI/bge-small-en-v1.5
print(f"Using model: {encoder.name}")
print(f"Device: {encoder.device}")
```

## Features

### Model Selection

You can specify any compatible sentence-transformers model:

```python
encoder = LocalEncoder(name="all-MiniLM-L6-v2")
```

Popular models include:
- `BAAI/bge-small-en-v1.5` (default) - Balanced performance and speed
- `all-MiniLM-L6-v2` - Fast and lightweight
- `all-mpnet-base-v2` - Higher quality, slower
- Any model from [sentence-transformers](https://www.sbert.net/docs/pretrained_models.html)

### Device Selection

The encoder automatically uses the best available device:
1. CUDA (NVIDIA GPUs)
2. MPS (Apple Silicon)
3. CPU (fallback)

You can also specify a device explicitly:

```python
encoder = LocalEncoder(device="cuda")  # or "mps", "cpu"
```

### Privacy and Offline Usage

Since all computation happens locally:
- No data is sent to external APIs
- Works completely offline (after initial model download)
- Full control over model versions and updates

## Integration with Routers

The `LocalEncoder` works seamlessly with both `SemanticRouter` and `HybridRouter`:

```python
from semantic_router.routers import SemanticRouter
from semantic_router.route import Route

routes = [
    Route(
        name="technical",
        utterances=["How does this work?", "Explain the architecture"]
    ),
    Route(
        name="general",
        utterances=["Hello", "How are you?"]
    )
]

router = SemanticRouter(
    encoder=encoder,
    routes=routes,
    auto_sync="local"
)
```

## Best Practices

1. **Model Selection**: Balance quality vs. speed based on your use case
2. **Hardware**: Use GPU acceleration when available for better performance
3. **First Run**: The first initialization downloads the model, which may take a few minutes
4. **Memory**: Larger models require more RAM/VRAM - monitor resource usage
5. **Embeddings**: All embeddings are L2-normalized by default

## Example Notebook

For a complete example of using the local encoder, see the [Local Encoder Notebook](../encoders/local-encoder.ipynb).
