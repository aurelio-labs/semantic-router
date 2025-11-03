Semantic Router integrates with AWS Bedrock's embedding models through the `BedrockEncoder` class. This integration provides access to Amazon's Titan and Cohere embedding models hosted on AWS Bedrock.

## Overview

The `BedrockEncoder` enables you to use AWS Bedrock embedding models for semantic routing tasks. It supports both synchronous and asynchronous encoding operations, making it suitable for various deployment scenarios.

## Getting Started

### Prerequisites

1. AWS credentials with access to Bedrock
2. Semantic Router version 0.0.40 or later

### Installation

```bash
pip install "semantic-router[bedrock]"
```

### Basic Usage

```python
import os
from semantic_router.encoders import BedrockEncoder

encoder = BedrockEncoder(
    name="amazon.titan-embed-text-v1",  # or amazon.titan-embed-text-v2, cohere.embed-english-v3
    score_threshold=0.5,
    access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    session_token=os.getenv("AWS_SESSION_TOKEN"),
    region=os.getenv("AWS_REGION")
)
```

## Features

### Supported Models

The `BedrockEncoder` supports the following AWS Bedrock embedding models:
- `amazon.titan-embed-text-v1`
- `amazon.titan-embed-text-v2`
- `amazon.titan-embed-image-v1`
- `cohere.embed-english-v3`

### AWS Credentials

You can provide AWS credentials in several ways:
- Pass directly to the encoder constructor
- Set environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, `AWS_REGION`)
- Use AWS credential profiles

### Asynchronous Support

The encoder supports both synchronous and asynchronous operations:

```python
# Synchronous encoding
embeddings = encoder(["your text here"])

# Asynchronous encoding
embeddings = await encoder.acall(["your text here"])
```

## Integration with Routers

The `BedrockEncoder` can be used with both `SemanticRouter` and `HybridRouter`:

```python
from semantic_router.routers import SemanticRouter
from semantic_router.route import Route

routes = [
    Route(
        name="support",
        utterances=["I need help", "Can you assist me?"]
    ),
    Route(
        name="sales",
        utterances=["I want to buy", "How much does it cost?"]
    )
]

router = SemanticRouter(
    encoder=encoder,
    routes=routes,
    auto_sync="local"
)
```

## Best Practices

1. **Credential Management**: Use AWS IAM roles or secure credential storage rather than hardcoding credentials
2. **Model Selection**: Choose the appropriate Bedrock model based on your use case and embedding dimensionality requirements
3. **Region Selection**: Select an AWS region close to your deployment for lower latency
4. **Batch Processing**: For multiple texts, use the list input format for better performance

## Example Notebook

For a complete example of using the Bedrock integration, see the [Bedrock Encoder Notebook](../encoders/bedrock.ipynb).
