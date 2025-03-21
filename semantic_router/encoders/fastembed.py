from typing import Any, List, Optional

import numpy as np
from pydantic import PrivateAttr

from semantic_router.encoders import DenseEncoder


class FastEmbedEncoder(DenseEncoder):
    """Dense encoder that uses local FastEmbed to embed documents. Supports text only.
    Requires the fastembed package which can be installed with `pip install 'semantic-router[fastembed]'`

    :param name: The name of the embedding model to use.
    :param max_length: The maximum length of the input text.
    :param cache_dir: The directory to cache the embedding model.
    :param threads: The number of threads to use for the embedding.
    """

    type: str = "fastembed"
    name: str = "BAAI/bge-small-en-v1.5"
    max_length: int = 512
    cache_dir: Optional[str] = None
    threads: Optional[int] = None
    _client: Any = PrivateAttr()

    def __init__(self, score_threshold: float = 0.5, **data):
        """Initialize the FastEmbed encoder.

        :param score_threshold: The threshold for the score of the embedding.
        :type score_threshold: float
        """
        # TODO default score_threshold not thoroughly tested, should optimize
        super().__init__(score_threshold=score_threshold, **data)
        self._client = self._initialize_client()

    def _initialize_client(self):
        """Initialize the FastEmbed library. Requires the fastembed package."""
        try:
            from fastembed import TextEmbedding
        except ImportError:
            raise ImportError(
                "Please install fastembed to use FastEmbedEncoder. "
                "You can install it with: "
                "`pip install 'semantic-router[fastembed]'`"
            )

        embedding_args = {
            "model_name": self.name,
            "max_length": self.max_length,
            "cache_dir": self.cache_dir,
            "threads": self.threads,
        }

        embedding_args = {k: v for k, v in embedding_args.items() if v is not None}

        embedding = TextEmbedding(**embedding_args)
        return embedding

    def __call__(self, docs: List[str]) -> List[List[float]]:
        """Embed a list of documents. Supports text only.

        :param docs: The documents to embed.
        :type docs: List[str]
        :raise ValueError: If the embedding fails.
        :return: The vector embeddings of the documents.
        :rtype: List[List[float]]
        """
        try:
            embeds: List[np.ndarray] = list(self._client.embed(docs))
            embeddings: List[List[float]] = [e.tolist() for e in embeds]
            return embeddings
        except Exception as e:
            raise ValueError(f"FastEmbed embed failed. Error: {e}") from e
