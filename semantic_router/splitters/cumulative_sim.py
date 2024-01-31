from typing import List
from semantic_router.splitters import BaseSplitter
import numpy as np
from semantic_router.utils import DocumentSplit

class CumulativeSimSplitter(BaseSplitter):
    
    """
    Called "cumulative sim" because we check the similarities of the embeddings of cumulative concatenated documents with the next document.
    """

    def __init__(
        self,
        docs: List[str],
        name: str = "cumulative_similarity_splitter",
        similarity_threshold: float = 0.45,
    ):
        super().__init__(
            docs=docs,
            name=name, 
            similarity_threshold=similarity_threshold,
            )

    def __call__(self):
        total_docs = len(self.docs)
        splits = []
        curr_split_start_idx = 0
        curr_split_num = 1

        for idx in range(1, total_docs):
            if idx + 1 < total_docs:
                curr_split_docs = "\n".join(self.docs[curr_split_start_idx : idx + 1])
                next_doc = self.docs[idx + 1]

                curr_split_docs_embed = self.encoder([curr_split_docs])[0]
                next_doc_embed = self.encoder([next_doc])[0]

                curr_sim_score = np.dot(curr_split_docs_embed, next_doc_embed) / (
                    np.linalg.norm(curr_split_docs_embed)
                    * np.linalg.norm(next_doc_embed)
                )

                if curr_sim_score < self.similarity_threshold:
                    splits.append(
                        DocumentSplit(
                            docs=list(self.docs[curr_split_start_idx : idx + 1]),
                            is_triggered=True,
                            triggered_score=curr_sim_score,
                        )
                    )
                    curr_split_start_idx = idx + 1
                    curr_split_num += 1

        splits.append(DocumentSplit(docs=list(self.docs[curr_split_start_idx:])))
        return splits