from typing import Any, ClassVar, List

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from semantic_router.route import Route
from semantic_router.schema import SparseEmbedding


class DenseEncoder(BaseModel):
    name: str
    type: str = Field(default="base")

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

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
        embeddings = []
        for row in sparse_arrays:
            # get positions of non-zero values
            (idx,) = np.nonzero(row)
            if idx.size == 0:
                # a row can be all zero if none of its tokens are in the corpus:
                # keep a single zero entry, so there is one embedding per row
                embeddings.append(SparseEmbedding(embedding=np.zeros((1, 2))))
            else:
                compact_array = np.array([idx, row[idx]]).T
                embeddings.append(SparseEmbedding.from_compact_array(compact_array))
        return embeddings


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
