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
        score_threshold: float = 0.3,
    ):
        super().__init__(name=name, encoder=encoder, score_threshold=score_threshold)
        logger.info(
            f"Initialized {self.name} with score threshold: {self.score_threshold}"
        )

    def encode_documents(self, docs: List[str]) -> np.ndarray:
        encoded_docs = self.encoder(docs)
        encoded_docs_array = np.array(encoded_docs)
        logger.info(f"Encoded {len(docs)} documents")
        return encoded_docs_array

    def adjust_threshold(self, similarities):
        if len(similarities) > 5:
            # Calculate recent mean similarity and standard deviation
            recent_similarities = similarities[-5:]
            mean_similarity = np.mean(recent_similarities)
            std_dev_similarity = np.std(recent_similarities)
            logger.debug(
                f"Recent mean similarity: {mean_similarity}, "
                f"std dev: {std_dev_similarity}"
            )

            # Calculate the rate of change (delta) for mean
            # similarity and standard deviation
            if len(similarities) > 10:
                previous_similarities = similarities[-10:-5]
                previous_mean_similarity = np.mean(previous_similarities)
                previous_std_dev_similarity = np.std(previous_similarities)
                delta_mean = mean_similarity - previous_mean_similarity
                delta_std_dev = std_dev_similarity - previous_std_dev_similarity
            else:
                delta_mean = delta_std_dev = 0

            # Adjust the threshold based on the deviation from the mean similarity
            # and the rate of change in mean similarity and standard deviation
            adjustment_factor = (
                std_dev_similarity + abs(delta_mean) + abs(delta_std_dev)
            )
            adjusted_threshold = mean_similarity - adjustment_factor

            # Dynamically set the lower bound based on the rate of change
            dynamic_lower_bound = max(0.2, 0.2 + delta_mean - delta_std_dev)

            # Introduce a minimum split threshold that is higher than the
            # dynamic lower bound
            min_split_threshold = 0.3

            # Ensure the new threshold is within a sensible range,
            # dynamically adjusting the lower bound
            # and considering the minimum split threshold
            new_threshold = max(
                np.clip(adjusted_threshold, dynamic_lower_bound, self.score_threshold),
                min_split_threshold,
            )

            logger.debug(
                f"Adjusted threshold to {new_threshold}, with dynamic lower "
                f"bound {dynamic_lower_bound}"
            )
            return new_threshold
        return self.score_threshold

    def calculate_dynamic_context_similarity(self, encoded_docs):
        split_indices = [0]
        similarities = []
        dynamic_window_size = 5  # Starting window size

        norms = np.linalg.norm(encoded_docs, axis=1)
        for idx in range(1, len(encoded_docs)):
            # Adjust window size based on recent variability
            if len(similarities) > 10:
                std_dev_recent = np.std(similarities[-10:])
                dynamic_window_size = 5 if std_dev_recent < 0.05 else 10

            window_start = max(0, idx - dynamic_window_size)
            cumulative_context = np.mean(encoded_docs[window_start:idx], axis=0)
            cumulative_norm = np.linalg.norm(cumulative_context)

            curr_sim_score = np.dot(cumulative_context, encoded_docs[idx]) / (
                cumulative_norm * norms[idx] + 1e-10
            )

            similarities.append(curr_sim_score)

            dynamic_threshold = self.adjust_threshold(similarities)
            if curr_sim_score < dynamic_threshold:
                split_indices.append(idx)

        return split_indices, similarities

    def __call__(self, docs: List[str]):
        logger.info(f"Splitting {len(docs)} documents")
        encoded_docs = self.encode_documents(docs)
        split_indices, similarities = self.calculate_dynamic_context_similarity(
            encoded_docs
        )
        splits = []

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
