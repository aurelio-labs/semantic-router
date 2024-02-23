from typing import List
import numpy as np

from semantic_router.encoders.base import BaseEncoder
from semantic_router.schema import DocumentSplit
from semantic_router.splitters.utils import split_to_sentences, tiktoken_length
from semantic_router.utils.logger import logger


class RollingWindowSplitter:
    def __init__(
        self,
        encoder: BaseEncoder,
        threshold_adjustment: float = 0.01,
        window_size=5,
        min_split_tokens=100,
        max_split_tokens=300,
        split_tokens_tolerance=10,
        plot_splits=False,
    ):
        self.calculated_threshold: float
        self.encoder = encoder
        self.threshold_adjustment = threshold_adjustment
        self.window_size = window_size
        self.plot_splits = plot_splits
        self.min_split_tokens = min_split_tokens
        self.max_split_tokens = max_split_tokens
        self.split_tokens_tolerance = split_tokens_tolerance

    def encode_documents(self, docs: List[str]) -> np.ndarray:
        try:
            embeddings = self.encoder(docs)
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Error encoding documents {docs}: {e}")
            raise

    def calculate_similarity_scores(self, encoded_docs: np.ndarray) -> List[float]:
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

    def find_split_indices(self, similarities: List[float]) -> List[int]:
        split_indices = []
        for idx in range(1, len(similarities)):
            if similarities[idx] < self.calculated_threshold:
                split_indices.append(idx + 1)
        return split_indices

    def find_optimal_threshold(self, docs: List[str], encoded_docs: np.ndarray):
        token_counts = [tiktoken_length(doc) for doc in docs]
        cumulative_token_counts = np.cumsum([0] + token_counts)
        similarity_scores = self.calculate_similarity_scores(encoded_docs)

        # Analyze the distribution of similarity scores to set initial bounds
        median_score = np.median(similarity_scores)
        std_dev = np.std(similarity_scores)

        # Set initial bounds based on median and standard deviation
        low = max(0.0, float(median_score - std_dev))
        high = min(1.0, float(median_score + std_dev))

        iteration = 0
        while low <= high:
            self.calculated_threshold = (low + high) / 2
            logger.info(
                f"Iteration {iteration}: Trying threshold: {self.calculated_threshold}"
            )
            split_indices = self.find_split_indices(similarity_scores)

            # Calculate the token counts for each split using the cumulative sums
            split_token_counts = [
                cumulative_token_counts[end] - cumulative_token_counts[start]
                for start, end in zip(
                    [0] + split_indices, split_indices + [len(token_counts)]
                )
            ]

            # Calculate the median token count for the splits
            median_tokens = np.median(split_token_counts)
            logger.info(
                f"Iteration {iteration}: Median tokens per split: {median_tokens}"
            )
            if (
                self.min_split_tokens - self.split_tokens_tolerance
                <= median_tokens
                <= self.max_split_tokens + self.split_tokens_tolerance
            ):
                logger.info(
                    f"Iteration {iteration}: "
                    f"Optimal threshold {self.calculated_threshold} found "
                    f"with median tokens ({median_tokens}) in target range "
                    f" {self.min_split_tokens}-{self.max_split_tokens}."
                )
                break
            elif median_tokens < self.min_split_tokens:
                high = self.calculated_threshold - self.threshold_adjustment
                logger.info(f"Iteration {iteration}: Adjusting high to {high}")
            else:
                low = self.calculated_threshold + self.threshold_adjustment
                logger.info(f"Iteration {iteration}: Adjusting low to {low}")
            iteration += 1

        logger.info(f"Final optimal threshold: {self.calculated_threshold}")
        return self.calculated_threshold

    def split_documents(
        self, docs: List[str], split_indices: List[int], similarities: List[float]
    ) -> List[DocumentSplit]:
        """
        This method iterates through each document, appending it to the current split
        until it either reaches a split point (determined by split_indices) or exceeds
        the maximum token limit for a split (self.max_split_tokens).
        When a document causes the current token count to exceed this limit,
        or when a split point is reached and the minimum token requirement is met,
        the current split is finalized and added to the List of splits.
        """
        token_counts = [tiktoken_length(doc) for doc in docs]
        splits, current_split = [], []
        current_tokens_count = 0

        for doc_idx, doc in enumerate(docs):
            doc_token_count = token_counts[doc_idx]

            # Check if current index is a split point based on similarity
            if doc_idx + 1 in split_indices:
                if current_tokens_count + doc_token_count >= self.min_split_tokens:
                    # Include the current document before splitting
                    # if it doesn't exceed the max limit
                    current_split.append(doc)
                    current_tokens_count += doc_token_count

                    triggered_score = (
                        similarities[doc_idx] if doc_idx < len(similarities) else None
                    )
                    splits.append(
                        DocumentSplit(
                            docs=current_split.copy(),
                            is_triggered=True,
                            triggered_score=triggered_score,
                            token_count=current_tokens_count,
                        )
                    )
                    logger.info(
                        f"Split finalized with {current_tokens_count} tokens due to "
                        f"threshold {self.calculated_threshold}."
                    )
                    current_split, current_tokens_count = [], 0
                    continue  # Move to the next document after splitting

            # Check if adding the current document exceeds the max token limit
            if current_tokens_count + doc_token_count > self.max_split_tokens:
                if current_tokens_count >= self.min_split_tokens:
                    splits.append(
                        DocumentSplit(
                            docs=current_split.copy(),
                            is_triggered=False,
                            triggered_score=None,
                            token_count=current_tokens_count,
                        )
                    )
                    logger.info(
                        f"Split finalized with {current_tokens_count} tokens due to "
                        f"exceeding token limit of {self.max_split_tokens}."
                    )
                    current_split, current_tokens_count = [], 0

            current_split.append(doc)
            current_tokens_count += doc_token_count

        # Handle the last split
        if current_split:
            splits.append(
                DocumentSplit(
                    docs=current_split.copy(),
                    is_triggered=False,
                    triggered_score=None,
                    token_count=current_tokens_count,
                )
            )
            logger.info(
                f"Final split added with {current_tokens_count} "
                "tokens due to remaining documents."
            )

        # Validation to ensure no tokens are lost during the split
        original_token_count = sum(token_counts)
        split_token_count = sum(
            [tiktoken_length(doc) for split in splits for doc in split.docs]
        )
        if original_token_count != split_token_count:
            logger.error(
                f"Token count mismatch: {original_token_count} != {split_token_count}"
            )
            raise ValueError(
                f"Token count mismatch: {original_token_count} != {split_token_count}"
            )

        return splits

    def plot_similarity_scores(
        self, similarities: List[float], split_indices: List[int]
    ):
        try:
            from matplotlib import pyplot as plt
        except ImportError:
            logger.warning("Plotting is disabled. Please `pip install matplotlib`.")
            return

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
            y=self.calculated_threshold,
            color="g",
            linestyle="-.",
            label="Threshold Similarity Score",
        )

        # Annotating each similarity score
        for i, score in enumerate(similarities):
            plt.annotate(
                f"{score:.2f}",  # Formatting to two decimal places
                (i, score),
                textcoords="offset points",
                xytext=(0, 10),  # Positioning the text above the point
                ha="center",
            )  # Center-align the text

        plt.xlabel("Document Segment Index")
        plt.ylabel("Similarity Score")
        plt.title(
            f"Threshold: {self.calculated_threshold} |"
            f" Window Size: {self.window_size}",
            loc="right",
            fontsize=10,
        )
        plt.suptitle("Document Similarity Scores", fontsize=14)
        plt.legend()
        plt.show()

    def plot_sentence_similarity_scores(
        self, docs: List[str], threshold: float, window_size: int
    ):
        try:
            from matplotlib import pyplot as plt
        except ImportError:
            logger.warning("Plotting is disabled. Please `pip install matplotlib`.")
            return
        """
        Computes similarity scores between the average of the last
        'window_size' sentences and the next one,
        plots a graph of these similarity scores, and prints the first
        sentence after a similarity score below
        a specified threshold.
        """
        sentences = [sentence for doc in docs for sentence in split_to_sentences(doc)]
        encoded_sentences = self.encode_documents(sentences)
        similarity_scores = []

        for i in range(window_size, len(encoded_sentences)):
            window_avg_encoding = np.mean(
                encoded_sentences[i - window_size : i], axis=0
            )
            sim_score = np.dot(window_avg_encoding, encoded_sentences[i]) / (
                np.linalg.norm(window_avg_encoding)
                * np.linalg.norm(encoded_sentences[i])
                + 1e-10
            )
            similarity_scores.append(sim_score)

        plt.figure(figsize=(10, 8))
        plt.plot(similarity_scores, marker="o", linestyle="-", color="b")
        plt.title("Sliding Window Sentence Similarity Scores")
        plt.xlabel("Sentence Index")
        plt.ylabel("Similarity Score")
        plt.grid(True)
        plt.axhline(y=threshold, color="r", linestyle="--", label="Threshold")
        plt.show()

        for i, score in enumerate(similarity_scores):
            if score < threshold:
                print(
                    f"First sentence after similarity score "
                    f"below {threshold}: {sentences[i + window_size]}"
                )

    def __call__(self, docs: List[str]) -> List[DocumentSplit]:
        if not docs:
            raise ValueError("At least one document is required for splitting.")

        if len(docs) == 1:
            token_count = tiktoken_length(docs[0])
            if token_count > self.max_split_tokens:
                logger.warning(
                    f"Single document exceeds the maximum token limit "
                    f"of {self.max_split_tokens}. "
                    "Splitting to sentences before semantically splitting."
                )
            docs = split_to_sentences(docs[0])
        encoded_docs = self.encode_documents(docs)
        self.find_optimal_threshold(docs, encoded_docs)
        similarities = self.calculate_similarity_scores(encoded_docs)
        split_indices = self.find_split_indices(similarities=similarities)
        splits = self.split_documents(docs, split_indices, similarities)
        self.plot_similarity_scores(similarities, split_indices)
        return splits
