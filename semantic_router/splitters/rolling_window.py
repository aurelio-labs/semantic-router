from dataclasses import dataclass
from typing import List
from warnings import warn

import numpy as np

from semantic_router.encoders.base import BaseEncoder
from semantic_router.schema import DocumentSplit
from semantic_router.splitters.base import BaseSplitter
from semantic_router.splitters.utils import split_to_sentences, tiktoken_length
from semantic_router.utils.logger import logger


@dataclass
class SplitStatistics:
    total_documents: int
    total_splits: int
    splits_by_threshold: int
    splits_by_max_chunk_size: int
    splits_by_last_split: int
    min_token_size: int
    max_token_size: int
    splits_by_similarity_ratio: float

    def __str__(self):
        return (
            f"Splitting Statistics:\n"
            f"  - Total Documents: {self.total_documents}\n"
            f"  - Total Splits: {self.total_splits}\n"
            f"  - Splits by Threshold: {self.splits_by_threshold}\n"
            f"  - Splits by Max Chunk Size: {self.splits_by_max_chunk_size}\n"
            f"  - Last Split: {self.splits_by_last_split}\n"
            f"  - Minimum Token Size of Split: {self.min_token_size}\n"
            f"  - Maximum Token Size of Split: {self.max_token_size}\n"
            f"  - Similarity Split Ratio: {self.splits_by_similarity_ratio:.2f}"
        )


class RollingWindowSplitter(BaseSplitter):
    def __init__(
        self,
        encoder: BaseEncoder,
        name="rolling_window_splitter",
        threshold_adjustment=0.01,
        dynamic_threshold: bool = True,
        window_size=5,
        min_split_tokens=100,
        max_split_tokens=300,
        split_tokens_tolerance=10,
        plot_splits=False,
        enable_statistics=False,
    ):
        warn(
            "Splitters are being deprecated. They have moved to their own "
            "package. Please migrate to the `semantic-chunkers` package. More "
            "information can be found at:\n"
            "https://github.com/aurelio-labs/semantic-chunkers",
            stacklevel=2,
        )
        super().__init__(name=name, encoder=encoder)
        self.calculated_threshold: float
        self.encoder = encoder
        self.threshold_adjustment = threshold_adjustment
        self.dynamic_threshold = dynamic_threshold
        self.window_size = window_size
        self.plot_splits = plot_splits
        self.min_split_tokens = min_split_tokens
        self.max_split_tokens = max_split_tokens
        self.split_tokens_tolerance = split_tokens_tolerance
        self.enable_statistics = enable_statistics
        self.statistics: SplitStatistics

    def __call__(self, docs: List[str]) -> List[DocumentSplit]:
        """Split documents into smaller chunks based on semantic similarity.

        :param docs: list of text documents to be split, if only wanted to
            split a single document, pass it as a list with a single element.

        :return: list of DocumentSplit objects containing the split documents.
        """
        if not docs:
            raise ValueError("At least one document is required for splitting.")

        if len(docs) == 1:
            token_count = tiktoken_length(docs[0])
            if token_count > self.max_split_tokens:
                logger.info(
                    f"Single document exceeds the maximum token limit "
                    f"of {self.max_split_tokens}. "
                    "Splitting to sentences before semantically splitting."
                )
            docs = split_to_sentences(docs[0])
        encoded_docs = self._encode_documents(docs)
        similarities = self._calculate_similarity_scores(encoded_docs)
        if self.dynamic_threshold:
            self._find_optimal_threshold(docs, similarities)
        else:
            if self.encoder.score_threshold is None:
                raise ValueError(
                    "No score threshold provided for encoder. Please set the score threshold "
                    "in the encoder config."
                )
            self.calculated_threshold = self.encoder.score_threshold
        split_indices = self._find_split_indices(similarities=similarities)
        splits = self._split_documents(docs, split_indices, similarities)

        if self.plot_splits:
            self.plot_similarity_scores(similarities, split_indices, splits)

        if self.enable_statistics:
            print(self.statistics)

        return splits

    def _encode_documents(self, docs: List[str]) -> np.ndarray:
        """
        Encodes a list of documents into embeddings. If the number of documents exceeds 2000,
        the documents are split into batches to avoid overloading the encoder. OpenAI has a
        limit of len(array) < 2048.

        :param docs: List of text documents to be encoded.
        :return: A numpy array of embeddings for the given documents.
        """
        max_docs_per_batch = 2000
        embeddings = []

        for i in range(0, len(docs), max_docs_per_batch):
            batch_docs = docs[i : i + max_docs_per_batch]
            try:
                batch_embeddings = self.encoder(batch_docs)
                embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error encoding documents {batch_docs}: {e}")
                raise

        return np.array(embeddings)

    def _calculate_similarity_scores(self, encoded_docs: np.ndarray) -> List[float]:
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

    def _find_split_indices(self, similarities: List[float]) -> List[int]:
        split_indices = []
        for idx, score in enumerate(similarities):
            logger.debug(f"Similarity score at index {idx}: {score}")
            if score < self.calculated_threshold:
                logger.debug(
                    f"Adding to split_indices due to score < threshold: "
                    f"{score} < {self.calculated_threshold}"
                )
                # Split after the document at idx
                split_indices.append(idx + 1)
        return split_indices

    def _find_optimal_threshold(self, docs: List[str], similarity_scores: List[float]):
        token_counts = [tiktoken_length(doc) for doc in docs]
        cumulative_token_counts = np.cumsum([0] + token_counts)

        # Analyze the distribution of similarity scores to set initial bounds
        median_score = np.median(similarity_scores)
        std_dev = np.std(similarity_scores)

        # Set initial bounds based on median and standard deviation
        low = max(0.0, float(median_score - std_dev))
        high = min(1.0, float(median_score + std_dev))

        iteration = 0
        median_tokens = 0
        while low <= high:
            self.calculated_threshold = (low + high) / 2
            split_indices = self._find_split_indices(similarity_scores)
            logger.debug(
                f"Iteration {iteration}: Trying threshold: {self.calculated_threshold}"
            )

            # Calculate the token counts for each split using the cumulative sums
            split_token_counts = [
                cumulative_token_counts[end] - cumulative_token_counts[start]
                for start, end in zip(
                    [0] + split_indices, split_indices + [len(token_counts)]
                )
            ]

            # Calculate the median token count for the splits
            median_tokens = np.median(split_token_counts)
            logger.debug(
                f"Iteration {iteration}: Median tokens per split: {median_tokens}"
            )
            if (
                self.min_split_tokens - self.split_tokens_tolerance
                <= median_tokens
                <= self.max_split_tokens + self.split_tokens_tolerance
            ):
                logger.debug("Median tokens in target range. Stopping iteration.")
                break
            elif median_tokens < self.min_split_tokens:
                high = self.calculated_threshold - self.threshold_adjustment
                logger.debug(f"Iteration {iteration}: Adjusting high to {high}")
            else:
                low = self.calculated_threshold + self.threshold_adjustment
                logger.debug(f"Iteration {iteration}: Adjusting low to {low}")
            iteration += 1

        logger.debug(
            f"Optimal threshold {self.calculated_threshold} found "
            f"with median tokens ({median_tokens}) in target range "
            f"({self.min_split_tokens}-{self.max_split_tokens})."
        )

        return self.calculated_threshold

    def _split_documents(
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

        # Statistics
        splits_by_threshold = 0
        splits_by_max_chunk_size = 0
        splits_by_last_split = 0

        for doc_idx, doc in enumerate(docs):
            doc_token_count = token_counts[doc_idx]
            logger.debug(f"Accumulative token count: {current_tokens_count} tokens")
            logger.debug(f"Document token count: {doc_token_count} tokens")
            # Check if current index is a split point based on similarity
            if doc_idx + 1 in split_indices:
                if (
                    self.min_split_tokens
                    <= current_tokens_count + doc_token_count
                    < self.max_split_tokens
                ):
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
                    logger.debug(
                        f"Split finalized with {current_tokens_count} tokens due to "
                        f"threshold {self.calculated_threshold}."
                    )
                    current_split, current_tokens_count = [], 0
                    splits_by_threshold += 1
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
                    splits_by_max_chunk_size += 1
                    logger.debug(
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
            splits_by_last_split += 1
            logger.debug(
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

        # Statistics
        total_splits = len(splits)
        splits_by_similarity_ratio = (
            splits_by_threshold / total_splits if total_splits else 0
        )
        min_token_size = max_token_size = 0
        if splits:
            token_counts = [
                split.token_count for split in splits if split.token_count is not None
            ]
            min_token_size, max_token_size = min(token_counts, default=0), max(
                token_counts, default=0
            )

        self.statistics = SplitStatistics(
            total_documents=len(docs),
            total_splits=total_splits,
            splits_by_threshold=splits_by_threshold,
            splits_by_max_chunk_size=splits_by_max_chunk_size,
            splits_by_last_split=splits_by_last_split,
            min_token_size=min_token_size,
            max_token_size=max_token_size,
            splits_by_similarity_ratio=splits_by_similarity_ratio,
        )

        return splits

    def plot_similarity_scores(
        self,
        similarities: List[float],
        split_indices: List[int],
        splits: List[DocumentSplit],
    ):
        try:
            from matplotlib import pyplot as plt
        except ImportError:
            logger.warning(
                "Plotting is disabled. Please `pip install "
                "semantic-router[processing]`."
            )
            return

        _, axs = plt.subplots(2, 1, figsize=(12, 12))  # Adjust for two plots

        # Plot 1: Similarity Scores
        axs[0].plot(similarities, label="Similarity Scores", marker="o")
        for split_index in split_indices:
            axs[0].axvline(
                x=split_index - 1,
                color="r",
                linestyle="--",
                label="Split" if split_index == split_indices[0] else "",
            )
        axs[0].axhline(
            y=self.calculated_threshold,
            color="g",
            linestyle="-.",
            label="Threshold Similarity Score",
        )

        # Annotating each similarity score
        for i, score in enumerate(similarities):
            axs[0].annotate(
                f"{score:.2f}",  # Formatting to two decimal places
                (i, score),
                textcoords="offset points",
                xytext=(0, 10),  # Positioning the text above the point
                ha="center",
            )  # Center-align the text

        axs[0].set_xlabel("Document Segment Index")
        axs[0].set_ylabel("Similarity Score")
        axs[0].set_title(
            f"Threshold: {self.calculated_threshold} |"
            f" Window Size: {self.window_size}",
            loc="right",
            fontsize=10,
        )
        axs[0].legend()

        # Plot 2: Split Token Size Distribution
        token_counts = [split.token_count for split in splits]
        axs[1].bar(range(len(token_counts)), token_counts, color="lightblue")
        axs[1].set_title("Split Token Sizes")
        axs[1].set_xlabel("Split Index")
        axs[1].set_ylabel("Token Count")
        axs[1].set_xticks(range(len(token_counts)))
        axs[1].set_xticklabels([str(i) for i in range(len(token_counts))])
        axs[1].grid(True)

        # Annotate each bar with the token size
        for idx, token_count in enumerate(token_counts):
            if not token_count:
                continue
            axs[1].text(
                idx, token_count + 0.01, str(token_count), ha="center", va="bottom"
            )

        plt.tight_layout()
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
        encoded_sentences = self._encode_documents(sentences)
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
