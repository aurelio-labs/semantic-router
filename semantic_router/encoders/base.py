from typing import Any, Coroutine, List, Optional

from pydantic import BaseModel, Field, field_validator
import numpy as np

from semantic_router.schema import SparseEmbedding


class DenseEncoder(BaseModel):
    name: str
    score_threshold: Optional[float] = None
    type: str = Field(default="base")

    class Config:
        arbitrary_types_allowed = True

    @field_validator("score_threshold")
    def set_score_threshold(cls, v):
        return float(v) if v is not None else None

    def __call__(self, docs: List[Any]) -> List[List[float]]:
        raise NotImplementedError("Subclasses must implement this method")

    def acall(self, docs: List[Any]) -> Coroutine[Any, Any, List[List[float]]]:
        raise NotImplementedError("Subclasses must implement this method")


class SparseEncoder(BaseModel):
    name: str
    type: str = Field(default="base")

    class Config:
        arbitrary_types_allowed = True

    def __call__(self, docs: List[str]) -> List[SparseEmbedding]:
        raise NotImplementedError("Subclasses must implement this method")

    async def acall(self, docs: List[str]) -> list[SparseEmbedding]:
        raise NotImplementedError("Subclasses must implement this method")

    def _array_to_sparse_embeddings(
        self, sparse_arrays: np.ndarray
    ) -> List[SparseEmbedding]:
        """Consumes several sparse vectors containing zero-values and returns a compact array."""
        if sparse_arrays.ndim != 2:
            raise ValueError(f"Expected a 2D array, got a {sparse_arrays.ndim}D array.")
        # get coordinates of non-zero values
        coords = np.nonzero(sparse_arrays)
        # create compact array
        compact_array = np.array([coords[0], coords[1], sparse_arrays[coords]]).T
        arr_range = range(compact_array[:, 0].max().astype(int) + 1)
        arrs = [compact_array[compact_array[:, 0] == i, :][:, 1:3] for i in arr_range]
        return [SparseEmbedding.from_compact_array(arr) for arr in arrs]
