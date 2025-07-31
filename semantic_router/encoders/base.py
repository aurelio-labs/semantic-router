from typing import Any, ClassVar, List, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from semantic_router.route import Route
from semantic_router.schema import SparseEmbedding


class DenseEncoder(BaseModel):
    name: str
    score_threshold: Optional[float] = None
    type: str = Field(default="base")

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("score_threshold")
    def set_score_threshold(cls, v: float | None) -> float | None:
        """Set the score threshold. If None, the score threshold is not used.

        :param v: The score threshold.
        :type v: float | None
        :return: The score threshold.
        :rtype: float | None
        """
        return float(v) if v is not None else None

    def __call__(self, docs: List[Any]) -> List[List[float]]:
        """Encode a list of documents. Documents can be any type, but the encoder must
        be built to handle that data type. Typically, these types are strings or
        arrays representing images.

        :param docs: The documents to encode.
        :type docs: List[Any]
        :return: The encoded documents.
        :rtype: List[List[float]]
        """
        raise NotImplementedError("Subclasses must implement this method")

    async def acall(self, docs: List[Any]) -> List[List[float]]:
        """Encode a list of documents asynchronously. Documents can be any type, but the
        encoder must be built to handle that data type. Typically, these types are
        strings or arrays representing images.

        :param docs: The documents to encode.
        :type docs: List[Any]
        :return: The encoded documents.
        :rtype: List[List[float]]
        """
        raise NotImplementedError("Subclasses must implement this method")


class SparseEncoder(BaseModel):
    """An encoder that encodes documents into a sparse format."""

    name: str
    type: str = Field(default="base")

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    def __call__(self, docs: List[str]) -> List[SparseEmbedding]:
        """Sparsely encode a list of documents. Documents can be any type, but the encoder must
        be built to handle that data type. Typically, these types are strings or
        arrays representing images.

        :param docs: The documents to encode.
        :type docs: List[Any]
        :return: The encoded documents.
        :rtype: List[SparseEmbedding]
        """
        raise NotImplementedError("Subclasses must implement this method")

    async def acall(self, docs: List[Any]) -> List[SparseEmbedding]:
        """Encode a list of documents asynchronously. Documents can be any type, but the
        encoder must be built to handle that data type. Typically, these types are
        strings or arrays representing images.

        :param docs: The documents to encode.
        :type docs: List[Any]
        :return: The encoded documents.
        :rtype: List[SparseEmbedding]
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _array_to_sparse_embeddings(
        self, sparse_arrays: np.ndarray
    ) -> List[SparseEmbedding]:
        """Consumes several sparse vectors containing zero-values and returns a compact
        array.

        :param sparse_arrays: The sparse arrays to compact.
        :type sparse_arrays: np.ndarray
        :return: The compact array.
        :rtype: List[SparseEmbedding]
        """
        # Handle PyTorch sparse tensors by converting to dense numpy arrays
        if hasattr(sparse_arrays, "to_dense"):
            sparse_arrays = sparse_arrays.to_dense().cpu().numpy()
        if sparse_arrays.ndim != 2:
            raise ValueError(f"Expected a 2D array, got a {sparse_arrays.ndim}D array.")
        # get coordinates of non-zero values
        coords = np.nonzero(sparse_arrays)
        if coords[0].size == 0:
            # Sparse Embeddings can be all zero, if query tokens do not appear in corpus at all
            return [SparseEmbedding(embedding=np.empty((1, 2)))]
        # create compact array
        compact_array = np.array([coords[0], coords[1], sparse_arrays[coords]]).T
        arr_range = range(compact_array[:, 0].max().astype(int) + 1)
        arrs = [compact_array[compact_array[:, 0] == i, :][:, 1:3] for i in arr_range]
        return [SparseEmbedding.from_compact_array(arr) for arr in arrs]


class FittableMixin:
    def fit(self, routes: list[Route]):
        pass


class AsymmetricDenseMixin:
    def encode_queries(self, docs: List[str]) -> List[List[float]]:
        """Convert query texts to dense embeddings optimized for querying"""
        raise NotImplementedError("Subclasses must implement this method")

    def encode_documents(self, docs: List[str]) -> List[List[float]]:
        """Convert document texts to dense embeddings optimized for storage"""
        raise NotImplementedError("Subclasses must implement this method")

    async def aencode_queries(self, docs: List[str]) -> List[List[float]]:
        """Async version of encode_queries"""
        raise NotImplementedError("Subclasses must implement this method")

    async def aencode_documents(self, docs: List[str]) -> List[List[float]]:
        """Async version of encode_documents"""
        raise NotImplementedError("Subclasses must implement this method")


class AsymmetricSparseMixin:
    def encode_queries(self, docs: List[str]) -> List[SparseEmbedding]:
        """Convert query texts to dense embeddings optimized for querying"""
        raise NotImplementedError("Subclasses must implement this method")

    def encode_documents(self, docs: List[str]) -> List[SparseEmbedding]:
        """Convert document texts to dense embeddings optimized for storage"""
        raise NotImplementedError("Subclasses must implement this method")

    async def aencode_queries(self, docs: List[str]) -> List[SparseEmbedding]:
        """Async version of encode_queries"""
        raise NotImplementedError("Subclasses must implement this method")

    async def aencode_documents(self, docs: List[str]) -> List[SparseEmbedding]:
        """Async version of encode_documents"""
        raise NotImplementedError("Subclasses must implement this method")
