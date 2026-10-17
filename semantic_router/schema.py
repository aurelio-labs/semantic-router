from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional, Union

import numpy as np
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    # type-only: importing aurelio_sdk calls logging.basicConfig
    from aurelio_sdk.schema import SparseEmbedding as BM25SparseEmbedding


class EncoderType(Enum):
    """The type of encoder."""

    AURELIO = "aurelio"
    AZURE = "azure"
    COHERE = "cohere"
    OPENAI = "openai"
    BM25 = "bm25"
    TFIDF = "tfidf"
    FASTEMBED = "fastembed"
    HUGGINGFACE = "huggingface"
    MISTRAL = "mistral"
    VIT = "vit"
    CLIP = "clip"
    GOOGLE = "google"
    BEDROCK = "bedrock"
    LITELLM = "litellm"
    OLLAMA = "ollama"
    JINA = "jina_ai"
    VOYAGE = "voyage"
    NIM = "nvidia_nim"
    LOCAL = "local"


class EncoderInfo(BaseModel):
    """Information about an encoder."""

    name: str
    token_limit: int


class MatchedUtterance(BaseModel):
    """One stored utterance that matched a query, with its raw similarity score.

    Populated by index-backed routers so downstream code can see which training examples
    pulled the route and tune thresholds or debug misrouting.

    :param text: The stored utterance text.
    :type text: str
    :param score: The raw similarity score between the query embedding and this utterance.
        Higher is more similar; the scale depends on the index metric (cosine, dotproduct,
        etc.) and the aggregation method.
    :type score: float
    :param route: The route this utterance belongs to.
    :type route: str
    """

    text: str
    score: float
    route: str


class RouteChoice(BaseModel):
    """A route choice output by the routers."""

    name: Optional[str] = None
    # the confidence behind the choice, and whether it is a probability
    confidence: Optional[float] = None
    calibrated: bool = False
    # utterance-level match details; None when the router does not expose them
    matched_utterances: Optional[list[MatchedUtterance]] = None


class Utterance(BaseModel):
    """One record in an index: a route name, one of its utterances, and the record's
    metadata (the route's own metadata plus the reserved ``sr_`` keys).
    """

    route: str
    utterance: Union[str, Any]
    metadata: Dict[str, Any] = {}


class Metric(Enum):
    """The metric to use in vector-based similarity search indexes."""

    COSINE = "cosine"
    DOTPRODUCT = "dotproduct"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"


class SparseEmbedding(BaseModel):
    """Sparse embedding interface. Primarily uses numpy operations for faster
    operations.
    """

    embedding: np.ndarray

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_compact_array(cls, array: np.ndarray):
        """Create a SparseEmbedding object from a compact array.

        :param array: A compact array.
        :type array: np.ndarray
        :return: A SparseEmbedding object.
        :rtype: SparseEmbedding
        """
        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError(
                f"Expected a 2D array with 2 columns, got a {array.ndim}D array with {array.shape[1]} columns. "
                "Column 0 should contain index positions, and column 1 should contain respective values."
            )
        return cls(embedding=array)

    @classmethod
    def from_vector(cls, vector: np.ndarray):
        """Consumes an array of sparse vectors containing zero-values.

        :param vector: A sparse vector.
        :type vector: np.ndarray
        :return: A SparseEmbedding object.
        :rtype: SparseEmbedding
        """
        if vector.ndim != 1:
            raise ValueError(f"Expected a 1D array, got a {vector.ndim}D array.")
        return cls.from_compact_array(np.array([np.arange(len(vector)), vector]).T)

    @classmethod
    def from_aurelio(cls, embedding: "BM25SparseEmbedding"):
        """Create a SparseEmbedding object from an AurelioSparseEmbedding object.

        :param embedding: An AurelioSparseEmbedding object.
        :type embedding: BM25SparseEmbedding
        :return: A SparseEmbedding object.
        :rtype: SparseEmbedding
        """
        arr = np.array([embedding.indices, embedding.values]).T
        return cls.from_compact_array(arr)

    @classmethod
    def from_dict(cls, sparse_dict: dict):
        """Create a SparseEmbedding object from a dictionary.

        :param sparse_dict: A dictionary of sparse values.
        :type sparse_dict: dict
        :return: A SparseEmbedding object.
        :rtype: SparseEmbedding
        """
        arr = np.array([list(sparse_dict.keys()), list(sparse_dict.values())]).T
        return cls.from_compact_array(arr)

    @classmethod
    def from_pinecone_dict(cls, sparse_dict: dict):
        """Create a SparseEmbedding object from a Pinecone dictionary.

        :param sparse_dict: A Pinecone dictionary.
        :type sparse_dict: dict
        :return: A SparseEmbedding object.
        :rtype: SparseEmbedding
        """
        arr = np.array([sparse_dict["indices"], sparse_dict["values"]]).T
        return cls.from_compact_array(arr)

    def to_dict(self):
        """Convert a SparseEmbedding object to a dictionary.

        :return: A dictionary of sparse values.
        :rtype: dict
        """
        return {
            i: v for i, v in zip(self.embedding[:, 0].astype(int), self.embedding[:, 1])
        }

    def to_pinecone(self):
        """Convert a SparseEmbedding object to a Pinecone dictionary.

        :return: A Pinecone dictionary.
        :rtype: dict
        """
        return {
            "indices": self.embedding[:, 0].astype(int).tolist(),
            "values": self.embedding[:, 1].tolist(),
        }

    # dictionary interface
    def items(self):
        """Return a list of (index, value) tuples from the SparseEmbedding object.

        :return: A list of (index, value) tuples.
        :rtype: list
        """
        return [
            (i, v)
            for i, v in zip(self.embedding[:, 0].astype(int), self.embedding[:, 1])
        ]
