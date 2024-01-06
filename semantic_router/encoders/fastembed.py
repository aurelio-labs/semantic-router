from typing import Any, List, Optional

import numpy as np
from pydantic import BaseModel, PrivateAttr


class FastEmbedEncoder(BaseModel):
    type: str = "fastembed"
    model_name: str = "BAAI/bge-small-en-v1.5"
    max_length: int = 512
    cache_dir: Optional[str] = None
    threads: Optional[int] = None
    _client: Any = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
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
            "model_name": self.model_name,
            "max_length": self.max_length,
            "cache_dir": self.cache_dir,
            "threads": self.threads,
        }

        embedding_args = {k: v for k, v in embedding_args.items() if v is not None}

        embedding = Embedding(**embedding_args)
        return embedding

    def __call__(self, docs: list[str]) -> list[list[float]]:
        try:
            embeds: List[np.ndarray] = list(self._client.embed(docs))
            embeddings: List[List[float]] = [e.tolist() for e in embeds]
            return embeddings
        except Exception as e:
            raise ValueError(f"FastEmbed embed failed. Error: {e}")
