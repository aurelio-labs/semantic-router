import numpy as np
from pydantic import PrivateAttr

from semantic_router.encoders.torch import TorchAbstractDenseEncoder


class STEncoder(TorchAbstractDenseEncoder):
    """Base class for sentence-transformers bi-encoders. Our recommended encoder for
    generating dense embeddings locally.
    """
    name: str = "all-MiniLM-L6-v2"
    type: str = "sentence-transformers"
    dimensions: int = 384
    device: str | None = None
    _model: any = PrivateAttr()

    def __init__(self, **kwargs):
        if kwargs.get("score_threshold") is None:
            kwargs["score_threshold"] = 0.5
        super().__init__(**kwargs)
        self._model = self._initialize_st_model()

    def _initialize_st_model(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "Please install sentence-transformers to use STEncoder. "
                "You can install it with: "
                "`pip install semantic-router[local]`"
            )
        model = SentenceTransformer(self.name)
        model.to(self.device)
        return model

    def __call__(
        self,
        docs: list[any],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ) -> list[list[float]]:
        # compute document embeddings `xd`
        xd = self._model.encode(docs, batch_size=batch_size)
        if normalize_embeddings:
            # TODO not sure if required
            xd = xd / np.linalg.norm(xd, axis=0)
        return xd
