from typing import Any, List, Optional

import numpy as np

from semantic_router.encoders import DenseEncoder
from semantic_router.index.base import BaseIndex
from semantic_router.llms import BaseLLM
from semantic_router.route import Route
from semantic_router.routers.base import BaseRouter


class SemanticRouter(BaseRouter):
    def __init__(
        self,
        encoder: Optional[DenseEncoder] = None,
        llm: Optional[BaseLLM] = None,
        routes: Optional[List[Route]] = None,
        index: Optional[BaseIndex] = None,  # type: ignore
        top_k: int = 5,
        aggregation: str = "mean",
        auto_sync: Optional[str] = None,
    ):
        index = self._get_index(index=index)
        encoder = self._get_encoder(encoder=encoder)
        super().__init__(
            encoder=encoder,
            llm=llm,
            routes=routes if routes else [],
            index=index,
            top_k=top_k,
            aggregation=aggregation,
            auto_sync=auto_sync,
        )
        # run initialize index now if auto sync is active
        if self.auto_sync:
            self._init_index_state()

    def _encode(self, text: str) -> Any:
        """Given some text, encode it."""
        # create query vector
        xq = np.array(self.encoder([text]))
        xq = np.squeeze(xq)  # Reduce to 1d array.
        return xq

    async def _async_encode(self, text: str) -> Any:
        """Given some text, encode it."""
        # create query vector
        xq = np.array(await self.encoder.acall(docs=[text]))
        xq = np.squeeze(xq)  # Reduce to 1d array.
        return xq
