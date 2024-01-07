from typing import Any, Optional

import numpy as np
from pydantic import PrivateAttr

from semantic_router.encoders import BaseEncoder


class FastEmbedEncoder(BaseEncoder):
    type: str = "fastembed"
    name: str = "BAAI/bge-small-en-v1.5"
    max_length: int = 512
    cache_dir: Optional[str] = None
    threads: Optional[int] = None
    _client: Any = PrivateAttr()

    def __init__(
        self, score_threshold: float = 0.5, **data
    ):  # TODO default score_threshold not thoroughly tested, should optimize
        super().__init__(score_threshold=score_threshold, **data)
        self._client = self._initialize_client()

    def _initialize_client(self):
        try:
            from fastembed.embedding import FlagEmbedding as Embedding
        except ImportError:
            raise ImportError(
                "Please install fastembed to use FastEmbedEncoder. "
                "You can install it with: "
                "`pip install semantic-router[fastembed]`"
            )

        embedding_args = {
            "model_name": self.name,
            "max_length": self.max_length,
            "cache_dir": self.cache_dir,
            "threads": self.threads,
        }

        embedding_args = {k: v for k, v in embedding_args.items() if v is not None}

        embedding = Embedding(**embedding_args)
        return embedding

    def __call__(self, docs: list[str]) -> list[list[float]]:
        try:
            embeds: list[np.ndarray] = list(self._client.embed(docs))
            embeddings: list[list[float]] = [e.tolist() for e in embeds]
            return embeddings
        except Exception as e:
            raise ValueError(f"FastEmbed embed failed. Error: {e}")
