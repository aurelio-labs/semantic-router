from typing import List, Optional

import numpy as np
from semantic_router.encoders.base import BaseEncoder


class FastEmbedEncoder(BaseEncoder):
    model_name: str = "BAAI/bge-small-en-v1.5"
    max_length: int = 512
    cache_dir: Optional[str] = None
    threads: Optional[int] = None
    type: str = "fastembed"

    def init(self):
        try:
            from fastembed.embedding import FlagEmbedding as Embedding
        except ImportError:
            raise ImportError(
                "Please install fastembed to use FastEmbedEncoder"
                "You can install it with: `pip install fastembed`"
            )

        embedding_args = {
            "model_name": self.model_name,
            "max_length": self.max_length,
        }
        if self.cache_dir is not None:
            embedding_args["cache_dir"] = self.cache_dir
        if self.threads is not None:
            embedding_args["threads"] = self.threads

        self.client = Embedding(**embedding_args)

    def __call__(self, docs: list[str]) -> list[list[float]]:
        try:
            embeds: List[np.ndarray] = list(self.client.embed(docs))

            embeddings: List[List[float]] = [e.tolist() for e in embeds]

            return embeddings
        except Exception as e:
            raise ValueError(f"FastEmbed embed failed. Error: {e}")
