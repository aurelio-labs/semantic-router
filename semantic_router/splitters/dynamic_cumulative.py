from typing import List

import numpy as np

from semantic_router.encoders import BaseEncoder
from semantic_router.schema import DocumentSplit
from semantic_router.splitters.base import BaseSplitter
from semantic_router.utils.logger import logger


class DynamicCumulativeSplitter(BaseSplitter):
    """
    Splits documents dynamically based on the cumulative similarity of document
    embeddings, adjusting thresholds and window sizes based on recent similarities.
    """

    def __init__(
        self,
        encoder: BaseEncoder,
        name: str = "dynamic_cumulative_similarity_splitter",
        score_threshold: float = 0.9,
    ):
        super().__init__(name=name, encoder=encoder, score_threshold=score_threshold)
        # Log the initialization details
        logger.info(
            f"Initialized {self.name} with score threshold: {self.score_threshold}"
        )

    def encode_documents(self, docs: List[str]) -> np.ndarray:
        # Encode the documents using the provided encoder and return as a numpy array
        encoded_docs = self.encoder(docs)
        logger.info(f"Encoded {len(docs)} documents")
        return np.array(encoded_docs)

    def adjust_threshold(self, similarities):
        # Adjust the similarity threshold based on recent similarities
        if len(similarities) <= 5:
            # If not enough data, return the default score threshold
            return self.score_threshold

        # Calculate mean and standard deviation of the last 5 similarities
        recent_similarities = similarities[-5:]
        mean_similarity, std_dev_similarity = np.mean(recent_similarities), np.std(
            recent_similarities
        )

        # Calculate the change in mean and standard deviation if enough data is
        # available
        delta_mean = delta_std_dev = 0
        if len(similarities) > 10:
            previous_similarities = similarities[-10:-5]
            delta_mean = mean_similarity - np.mean(previous_similarities)
            delta_std_dev = std_dev_similarity - np.std(previous_similarities)

        # Adjust the threshold based on the calculated metrics
        adjustment_factor = std_dev_similarity + abs(delta_mean) + abs(delta_std_dev)
        adjusted_threshold = mean_similarity - adjustment_factor
        dynamic_lower_bound = max(0.2, 0.2 + delta_mean - delta_std_dev)
        min_split_threshold = 0.3

        # Ensure the new threshold is within a sensible range
        new_threshold = max(
            np.clip(adjusted_threshold, dynamic_lower_bound, self.score_threshold),
            min_split_threshold,
        )
        logger.debug(
            f"Adjusted threshold to {new_threshold}, with dynamic lower "
            f"bound {dynamic_lower_bound}"
        )
        return new_threshold

    def calculate_dynamic_context_similarity(self, encoded_docs):
        # Calculate the dynamic context similarity to determine split indices
        split_indices, similarities = [0], []
        dynamic_window_size = 5  # Initial window size
        norms = np.linalg.norm(
            encoded_docs, axis=1
        )  # Pre-calculate norms for efficiency

        for idx in range(1, len(encoded_docs)):
            # Adjust window size based on the standard deviation of recent similarities
            if len(similarities) > 10:
                std_dev_recent = np.std(similarities[-10:])
                dynamic_window_size = 5 if std_dev_recent < 0.05 else 10

            # Calculate the similarity for the current document
            window_start = max(0, idx - dynamic_window_size)
            cumulative_context = np.mean(encoded_docs[window_start:idx], axis=0)
            cumulative_norm = np.linalg.norm(cumulative_context)
            curr_sim_score = np.dot(cumulative_context, encoded_docs[idx]) / (
                cumulative_norm * norms[idx] + 1e-10
            )

            similarities.append(curr_sim_score)
            # If the similarity is below the dynamically adjusted threshold,
            # mark a new split
            if curr_sim_score < self.adjust_threshold(similarities):
                split_indices.append(idx)

        return split_indices, similarities

    def __call__(self, docs: List[str]):
        # Main method to split the documents
        logger.info(f"Splitting {len(docs)} documents")
        encoded_docs = self.encode_documents(docs)
        split_indices, similarities = self.calculate_dynamic_context_similarity(
            encoded_docs
        )
        splits = []

        # Create DocumentSplit objects for each identified split
        last_idx = 0
        for idx in split_indices:
            if idx == 0:
                continue
            splits.append(
                DocumentSplit(
                    docs=docs[last_idx:idx],
                    is_triggered=(idx - last_idx > 1),
                    triggered_score=(
                        similarities[idx - 1] if idx - 1 < len(similarities) else None
                    ),
                )
            )
            last_idx = idx
        splits.append(
            DocumentSplit(
                docs=docs[last_idx:],
                is_triggered=(len(docs) - last_idx > 1),
                triggered_score=similarities[-1] if similarities else None,
            )
        )
        logger.info(f"Completed splitting documents into {len(splits)} splits")

        return splits
