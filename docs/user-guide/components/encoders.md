Encoders are essential components in Semantic Router that transform text (or other data) into numerical representations that capture semantic meaning. These numerical representations, called embeddings, allow the system to measure semantic similarity between texts, which is the core functionality of the routing process.

## Understanding Encoders

In Semantic Router, an encoder serves two primary purposes:

1. **Convert utterances from routes into embeddings** during initialization
2. **Convert incoming user queries into embeddings** during routing

By comparing these embeddings, Semantic Router can determine which route(s) best match the user's intent, even when the exact wording differs.

## Dense vs. Sparse Encoders

Semantic Router supports two main types of encoders:

### Dense Encoders

Dense encoders generate embeddings where every dimension has a value, resulting in a "dense" vector. These encoders typically:

- Produce fixed-size vectors (e.g., 1536 dimensions for OpenAI's text-embedding-3-small)
- Capture complex semantic relationships in the text
- Perform well on tasks requiring understanding of context and meaning

**Example usage**:

```python
from semantic_router.encoders import OpenAIEncoder
import os

# Set up API key
os.environ["OPENAI_API_KEY"] = "your-api-key"

# Initialize the encoder
encoder = OpenAIEncoder()

# Generate dense embeddings for documents
embeddings = encoder(["How's the weather today?", "Tell me about politics"])
```

### Sparse Encoders

Sparse encoders generate embeddings where most dimensions are zero, with only a few dimensions having non-zero values. These encoders typically:

- Focus on specific words or tokens in the text
- Excel at keyword matching and term frequency
- Can be more interpretable than dense encoders (non-zero dimensions often correspond to specific words)

**Example usage**:

```python
from semantic_router.encoders import AurelioSparseEncoder
from semantic_router import Route
import os

# Set up API key
os.environ["AURELIO_API_KEY"] = "your-api-key"

# Create some routes for routing
routes = [
    Route(name="weather", utterances=["How's the weather?", "Is it raining?"]),
    Route(name="politics", utterances=["Tell me about politics", "Who's the president?"])
]

# Initialize the sparse encoder
encoder = AurelioSparseEncoder()

# Generate sparse embeddings for documents
embeddings = encoder(["How's the weather today?", "Tell me about politics"])
```

## Hybrid Approaches

Semantic Router also allows combining both dense and sparse encoders in a hybrid approach through the `HybridRouter`. This can leverage the strengths of both encoding methods:

```python
from semantic_router.routers import HybridRouter
from semantic_router.encoders import OpenAIEncoder, AurelioSparseEncoder
import os

# Set up API keys
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
os.environ["AURELIO_API_KEY"] = "your-aurelio-api-key"

# Create dense and sparse encoders
dense_encoder = OpenAIEncoder()
sparse_encoder = AurelioSparseEncoder()

# Initialize the hybrid router
router = HybridRouter(
    encoder=dense_encoder,
    sparse_encoder=sparse_encoder,
    routes=routes,
    alpha=0.5  # Balance between dense (0) and sparse (1) embeddings
)
```

## Supported Encoders

### Dense Encoders

| Encoder | Description | Installation |
|---------|-------------|-------------|
| [OpenAIEncoder](https://semantic-router.aurelio.ai/api/encoders/openai) | Uses OpenAI's text embedding models | `pip install -qU semantic-router` |
| [AzureOpenAIEncoder](https://semantic-router.aurelio.ai/api/encoders/azure_openai) | Uses Azure OpenAI's text embedding models | `pip install -qU semantic-router` |
| [CohereEncoder](https://semantic-router.aurelio.ai/api/encoders/cohere) | Uses Cohere's text embedding models | `pip install -qU semantic-router` |
| [HuggingFaceEncoder](https://semantic-router.aurelio.ai/api/encoders/huggingface) | Uses local Hugging Face models | `pip install -qU "semantic-router[local]"` |
| [HFEndpointEncoder](https://semantic-router.aurelio.ai/api/encoders/huggingface) | Uses Hugging Face Inference API | `pip install -qU semantic-router` |
| [FastEmbedEncoder](https://semantic-router.aurelio.ai/api/encoders/fastembed) | Uses FastEmbed for local embeddings | `pip install -qU "semantic-router[local]"` |
| [MistralEncoder](https://semantic-router.aurelio.ai/api/encoders/mistral) | Uses Mistral's text embedding models | `pip install -qU semantic-router` |
| [GoogleEncoder](https://semantic-router.aurelio.ai/api/encoders/google) | Uses Google's text embedding models | `pip install -qU semantic-router` |
| [BedrockEncoder](https://semantic-router.aurelio.ai/api/encoders/bedrock) | Uses AWS Bedrock embedding models | `pip install -qU semantic-router` |
| [VitEncoder](https://semantic-router.aurelio.ai/api/encoders/vit) | Vision Transformer for image embeddings | `pip install -qU semantic-router` |
| [CLIPEncoder](https://semantic-router.aurelio.ai/api/encoders/clip) | Uses CLIP for image embeddings | `pip install -qU semantic-router` |

### Sparse Encoders

| Encoder | Description | Installation |
|---------|-------------|-------------|
| [BM25Encoder](https://semantic-router.aurelio.ai/api/encoders/bm25) | Implements BM25 algorithm for sparse embeddings | `pip install -qU semantic-router` |
| [TfidfEncoder](https://semantic-router.aurelio.ai/api/encoders/tfidf) | Implements TF-IDF for sparse embeddings | `pip install -qU semantic-router` |
| [AurelioSparseEncoder](https://semantic-router.aurelio.ai/api/encoders/aurelio) | Uses Aurelio's API for BM25 sparse embeddings | `pip install -qU semantic-router` |

## Using AutoEncoder

Semantic Router provides an `AutoEncoder` class that automatically selects the appropriate encoder based on the specified type:

```python
from semantic_router.encoders import AutoEncoder
from semantic_router.schema import EncoderType

# Create an encoder based on type
encoder = AutoEncoder(type=EncoderType.OPENAI.value, name="text-embedding-3-small").model

# Use the encoder
embeddings = encoder(["How can I help you today?"])
```

## Considerations for Choosing an Encoder

When selecting an encoder for your application, consider:

1. **Accuracy**: Dense encoders typically provide better semantic understanding but may miss exact keyword matches
2. **Speed**: Local encoders are faster but may be less accurate than cloud-based ones
3. **Cost**: Cloud-based encoders (OpenAI, Cohere, Aurelio AI) incur API costs
4. **Privacy**: Local encoders keep data within your environment
5. **Use case**: Hybrid approaches may work best for balanced retrieval

For more detailed information on specific encoders, refer to their respective documentation pages. 