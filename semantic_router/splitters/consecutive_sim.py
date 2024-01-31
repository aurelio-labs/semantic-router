from typing import List
from semantic_router.splitters import BaseSplitter
import numpy as np
from semantic_router.utils import DocumentSplit

class ConsecutiveSimSplitter(BaseSplitter):
    
    """
    Called "consecutive sim splitter" because we check the similarities of consecutive document embeddings (compare ith to i+1th document embedding).
    """

    def __init__(
        self,
        docs: List[str],
        name: str = "consecutive_similarity_splitter",
        similarity_threshold: float = 0.45,
    ):
        super().__init__(
            docs=docs,
            name=name, 
            similarity_threshold=similarity_threshold,
            )

    def __call__(self):
        doc_embeds = self.encoder(self.docs)
        norm_embeds = doc_embeds / np.linalg.norm(doc_embeds, axis=1, keepdims=True)
        sim_matrix = np.matmul(norm_embeds, norm_embeds.T)
        total_docs = len(self.docs)
        splits = []
        curr_split_start_idx = 0
        curr_split_num = 1

        for idx in range(1, total_docs):
            curr_sim_score = sim_matrix[idx - 1][idx]
            if idx < len(sim_matrix) and curr_sim_score < self.similarity_threshold:
                splits.append(
                    DocumentSplit(
                        docs=list(self.docs[curr_split_start_idx:idx]),
                        is_triggered=True,
                        triggered_score=curr_sim_score,
                    )
                )
                curr_split_start_idx = idx
                curr_split_num += 1
        splits.append(DocumentSplit(docs=list(self.docs[curr_split_start_idx:])))
        return splits