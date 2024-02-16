from typing import List

import numpy as np
from matplotlib import pyplot as plt
from nltk.tokenize import word_tokenize
from semantic_router.encoders.base import BaseEncoder

from semantic_router.schema import DocumentSplit
from semantic_router.splitters.base import BaseSplitter
from semantic_router.utils.logger import logger


class RollingWindowSplitter(BaseSplitter):
    """
    A splitter that divides documents into segments based on semantic similarity
    using a rolling window approach.
    It adjusts the similarity threshold dynamically.
    Splitting is based:
    - On the similarity threshold
    - On the maximum token limit for a split

    Attributes:
        encoder (Callable): A function to encode documents into semantic vectors.
        score_threshold (float): Initial threshold for similarity scores to decide
        splits.
        window_size (int): Size of the rolling window to calculate document context.
        plot_splits (bool): Whether to plot the similarity scores and splits for
        visualization.
        min_split_tokens (int): Minimum number of tokens for a valid document split.
        max_split_tokens (int): Maximum number of tokens a split can contain.
        split_tokens_tolerance (int): Tolerance in token count to still consider a split
        valid.
        threshold_step_size (float): Step size to adjust the similarity threshold during
        optimization.
    """

    def __init__(
        self,
        encoder: BaseEncoder,
        score_threshold=0.3,
        window_size=5,
        plot_splits=False,
        min_split_tokens=100,
        max_split_tokens=300,
        split_tokens_tolerance=10,
        threshold_step_size=0.01,
    ):
        self.encoder = encoder
        self.score_threshold = score_threshold
        self.window_size = window_size
        self.plot_splits = plot_splits
        self.min_split_tokens = min_split_tokens
        self.max_split_tokens = max_split_tokens
        self.split_tokens_tolerance = split_tokens_tolerance
        self.threshold_step_size = threshold_step_size

    def encode_documents(self, docs: list[str]) -> np.ndarray:
        return np.array(self.encoder(docs))

    def find_optimal_threshold(self, docs: list[str], encoded_docs: np.ndarray):
        logger.info(f"Number of documents for finding optimal threshold: {len(docs)}")
        token_counts = [len(word_tokenize(doc)) for doc in docs]
        low, high = 0, 1
        while low <= high:
            self.score_threshold = (low + high) / 2
            similarity_scores = self.calculate_similarity_scores(encoded_docs)
            split_indices = self.find_split_indices(similarity_scores)
            average_tokens = np.mean(
                [
                    sum(token_counts[start:end])
                    for start, end in zip(
                        [0] + split_indices, split_indices + [len(token_counts)]
                    )
                ]
            )
            if (
                self.min_split_tokens - self.split_tokens_tolerance
                <= average_tokens
                <= self.max_split_tokens + self.split_tokens_tolerance
            ):
                break
            elif average_tokens < self.min_split_tokens:
                high = self.score_threshold - self.threshold_step_size
            else:
                low = self.score_threshold + self.threshold_step_size

    def calculate_similarity_scores(self, encoded_docs: np.ndarray) -> list[float]:
        raw_similarities = []
        for idx in range(1, len(encoded_docs)):
            window_start = max(0, idx - self.window_size)
            cumulative_context = np.mean(encoded_docs[window_start:idx], axis=0)
            curr_sim_score = np.dot(cumulative_context, encoded_docs[idx]) / (
                np.linalg.norm(cumulative_context) * np.linalg.norm(encoded_docs[idx])
                + 1e-10
            )
            raw_similarities.append(curr_sim_score)
        return raw_similarities

    def find_split_indices(self, similarities: list[float]) -> list[int]:
        return [
            idx + 1
            for idx, sim in enumerate(similarities)
            if sim < self.score_threshold
        ]

    def split_documents(
        self, docs: list[str], split_indices: list[int], similarities: list[float]
    ) -> list[DocumentSplit]:
        """
        This method iterates through each document, appending it to the current split
        until it either reaches a split point (determined by split_indices) or exceeds
        the maximum token limit for a split (self.max_split_tokens).
        When a document causes the current token count to exceed this limit,
        or when a split point is reached and the minimum token requirement is met,
        the current split is finalized and added to the list of splits.
        """
        token_counts = [len(word_tokenize(doc)) for doc in docs]
        splits: List[DocumentSplit] = []
        current_split: List[str] = []
        current_tokens_count = 0

        for doc_idx, doc in enumerate(docs):
            doc_token_count = token_counts[doc_idx]
            # Check if current document causes token count to exceed max limit
            if (
                current_tokens_count + doc_token_count > self.max_split_tokens
                and current_tokens_count >= self.min_split_tokens
            ):
                splits.append(
                    DocumentSplit(docs=current_split.copy(), is_triggered=True)
                )
                logger.info(
                    f"Split finalized with {current_tokens_count} tokens due to "
                    f"exceeding token limit of {self.max_split_tokens}."
                )
                current_split, current_tokens_count = [], 0

            current_split.append(doc)
            current_tokens_count += doc_token_count

            # Check if current index is a split point based on similarity
            if doc_idx + 1 in split_indices or doc_idx == len(docs) - 1:
                if current_tokens_count >= self.min_split_tokens:
                    if doc_idx < len(similarities):
                        triggered_score = similarities[doc_idx]
                        splits.append(
                            DocumentSplit(
                                docs=current_split.copy(),
                                is_triggered=True,
                                triggered_score=triggered_score,
                            )
                        )
                        logger.info(
                            f"Split finalized with {current_tokens_count} tokens due to"
                            f" similarity score {triggered_score:.2f}."
                        )
                    else:
                        # This case handles the end of the document list
                        # where there's no similarity score
                        splits.append(
                            DocumentSplit(docs=current_split.copy(), is_triggered=False)
                        )
                        logger.info(
                            f"Split finalized with {current_tokens_count} tokens "
                            "at the end of the document list."
                        )
                    current_split, current_tokens_count = [], 0

        # Ensure any remaining documents are included in the final token count
        if current_split:
            splits.append(DocumentSplit(docs=current_split.copy(), is_triggered=False))
            logger.info(
                f"Final split added with {current_tokens_count} tokens "
                "due to remaining documents."
            )

        # Validation
        original_token_count = sum(token_counts)
        split_token_count = sum(
            [len(word_tokenize(doc)) for split in splits for doc in split.docs]
        )
        logger.debug(
            f"Original Token Count: {original_token_count}, "
            f"Split Token Count: {split_token_count}"
        )

        if original_token_count != split_token_count:
            logger.error(
                f"Token count mismatch: {original_token_count} != {split_token_count}"
            )
            for i, split in enumerate(splits):
                split_token_count = sum([len(word_tokenize(doc)) for doc in split.docs])
                logger.error(f"Split {i} Token Count: {split_token_count}")
            raise ValueError(
                f"Token count mismatch: {original_token_count} != {split_token_count}"
            )

        return splits

    # TODO: fix to plot split based on token count and final split
    def plot_similarity_scores(
        self, similarities: list[float], split_indices: list[int]
    ):
        if not self.plot_splits:
            return
        plt.figure(figsize=(12, 6))
        plt.plot(similarities, label="Similarity Scores", marker="o")
        for split_index in split_indices:
            plt.axvline(
                x=split_index - 1,
                color="r",
                linestyle="--",
                label="Split" if split_index == split_indices[0] else "",
            )
        plt.axhline(
            y=self.score_threshold,
            color="g",
            linestyle="-.",
            label="Threshold Similarity Score",
        )
        plt.xlabel("Document Segment Index")
        plt.ylabel("Similarity Score")
        plt.title(f"Threshold: {self.score_threshold}", loc="right", fontsize=10)
        plt.suptitle("Document Similarity Scores", fontsize=14)
        plt.legend()
        plt.show()

    def __call__(self, docs: list[str]) -> list[DocumentSplit]:
        encoded_docs = self.encode_documents(docs)
        self.find_optimal_threshold(docs, encoded_docs)
        similarities = self.calculate_similarity_scores(encoded_docs)
        split_indices = self.find_split_indices(similarities=similarities)
        splits = self.split_documents(docs, split_indices, similarities)

        self.plot_similarity_scores(similarities, split_indices)
        return splits
