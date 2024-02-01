from typing import List
from semantic_router.splitters.base import BaseSplitter
from semantic_router.encoders import BaseEncoder
import numpy as np
from semantic_router.schema import DocumentSplit

class ConsecutiveSimSplitter(BaseSplitter):
    
    """
    Called "consecutive sim splitter" because we check the similarities of consecutive document embeddings (compare ith to i+1th document embedding).
    """

    def __init__(
        self,
        encoder: BaseEncoder,
        name: str = "consecutive_similarity_splitter",
        similarity_threshold: float = 0.45
    ):
        super().__init__(
            name=name, 
            similarity_threshold=similarity_threshold,
            encoder=encoder
            )

    def __call__(self, docs: List[str]):
        doc_embeds = self.encoder(docs)
        norm_embeds = doc_embeds / np.linalg.norm(doc_embeds, axis=1, keepdims=True)
        sim_matrix = np.matmul(norm_embeds, norm_embeds.T)
        total_docs = len(docs)
        splits = []
        curr_split_start_idx = 0
        curr_split_num = 1

        for idx in range(1, total_docs):
            curr_sim_score = sim_matrix[idx - 1][idx]
            if idx < len(sim_matrix) and curr_sim_score < self.similarity_threshold:
                splits.append(
                    DocumentSplit(
                        docs=list(docs[curr_split_start_idx:idx]),
                        is_triggered=True,
                        triggered_score=curr_sim_score,
                    )
                )
                curr_split_start_idx = idx
                curr_split_num += 1
        splits.append(DocumentSplit(docs=list(docs[curr_split_start_idx:])))
        return splits
    

class ConsecutiveAvgSimSplitter(BaseSplitter):
    def __init__(
        self,
        encoder: BaseEncoder,
        name: str = "consecutive_similarity_splitter",
        similarity_threshold: float = 0.45,
        drop_threshold: float = 0.1  # Additional parameter to control the drop threshold
    ):
        super().__init__(
            name=name, 
            similarity_threshold=similarity_threshold,
            encoder=encoder
        )

    def __call__(self, docs: List[str], drop_threshold):
        doc_embeds = self.encoder(docs)
        norm_embeds = doc_embeds / np.linalg.norm(doc_embeds, axis=1, keepdims=True)
        sim_matrix = np.matmul(norm_embeds, norm_embeds.T)
        total_docs = len(docs)
        splits = []
        curr_split_start_idx = 0

        # Calculate similarity scores between consecutive documents
        sim_scores = [sim_matrix[i][i+1] for i in range(total_docs - 1)]

        # Calculate running average of similarity scores
        running_avg = [np.mean(sim_scores[:i+1]) for i in range(len(sim_scores))]

        for idx, curr_sim_score in enumerate(sim_scores):
            # Check for a significant drop in similarity compared to the running average
            if idx > 0 and (running_avg[idx-1] - curr_sim_score) > drop_threshold:
                splits.append(
                    DocumentSplit(
                        docs=list(docs[curr_split_start_idx:idx+1]),  # Include current doc in the split
                        is_triggered=True,
                        triggered_score=curr_sim_score,
                    )
                )
                curr_split_start_idx = idx + 1  # Update the start index for the next split

        # Add the last split
        if curr_split_start_idx < total_docs:
            splits.append(DocumentSplit(docs=list(docs[curr_split_start_idx:])))

        return splits
    

class ConsecutiveAvgSimSplitter2(BaseSplitter):
    def __init__(
        self,
        encoder: BaseEncoder,
        name: str = "consecutive_similarity_splitter",
        similarity_threshold: float = 0.45,
        drop_threshold: float = 0.1  # Additional parameter to control the drop threshold
    ):
        super().__init__(
            name=name, 
            similarity_threshold=similarity_threshold,
            encoder=encoder
        )

    def __call__(self, docs: List[str], drop_threshold):
        doc_embeds = self.encoder(docs)
        norm_embeds = doc_embeds / np.linalg.norm(doc_embeds, axis=1, keepdims=True)
        sim_matrix = np.matmul(norm_embeds, norm_embeds.T)
        total_docs = len(docs)
        splits = []
        curr_split_start_idx = 0

        # Initialize an empty list to store similarity scores for the current topic segment
        segment_sim_scores = []

        for idx in range(total_docs - 1):
            curr_sim_score = sim_matrix[idx][idx + 1]
            segment_sim_scores.append(curr_sim_score)

            # Calculate running average of similarity scores for the current segment
            running_avg = np.mean(segment_sim_scores)

            # Check for a significant drop in similarity compared to the running average
            if idx > 0 and (running_avg - curr_sim_score) > drop_threshold:
                splits.append(
                    DocumentSplit(
                        docs=list(docs[curr_split_start_idx:idx + 1]),  # Include current doc in the split
                        is_triggered=True,
                        triggered_score=curr_sim_score,
                    )
                )
                curr_split_start_idx = idx + 1
                # Reset the similarity scores for the new segment
                segment_sim_scores = [curr_sim_score]

        # Add the last split
        if curr_split_start_idx < total_docs:
            splits.append(DocumentSplit(docs=list(docs[curr_split_start_idx:])))

        return splits