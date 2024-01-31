from typing import List
from semantic_router.splitters.base import BaseSplitter
import numpy as np
from semantic_router.schema import DocumentSplit
from semantic_router.encoders import BaseEncoder

class CAVSimSplitter(BaseSplitter):
    
    """
    The CAVSimSplitter class is a document splitter that uses the concept of Cumulative Average Vectors (CAV) to determine where to split a sequence of documents based on their semantic similarity.

    For example, consider a sequence of documents [A, B, C, D, E, F]. The CAVSimSplitter works as follows:

    1. It starts with the first document (A) and calculates the cosine similarity between the embedding of A and the average embedding of the next two documents (B, C) if they exist, or the next one document (B) if only one exists.
    - Cosine Similarity: cos_sim(A, avg(B, C))

    2. It then moves to the next document (B), calculates the average embedding of the current documents (A, B), and calculates the cosine similarity with the average embedding of the next two documents (C, D) if they exist, or the next one document (C) if only one exists.
    - Cosine Similarity: cos_sim(avg(A, B), avg(C, D))

    3. This process continues, with the average embedding being calculated for the current cumulative documents and the next one or two documents. For example, at document C:
    - Cosine Similarity: cos_sim(avg(A, B, C), avg(D, E))

    4. If the similarity score between the average embedding of the current cumulative documents and the average embedding of the next one or two documents falls below the specified similarity threshold, a split is triggered. In our example, let's say the similarity score falls below the threshold between the average of documents A, B, C and the average of D, E. The splitter will then create a split, resulting in two groups of documents: [A, B, C] and [D, E].

    5. After a split occurs, the process restarts with the next document in the sequence. For example, after the split between C and D, the process restarts with D and calculates the cosine similarity between the embedding of D and the average embedding of the next two documents if they exist.
    - Cosine Similarity: cos_sim(D, avg(E, F))

    6. Then we start accumulating and averaging from the left again. On the right there is only one more document left, F:
    - Cosine Similarity: cos_sim(avg(D, E), F)

    7. The process continues until all documents have been processed.

    The result is a list of DocumentSplit objects, each representing a group of semantically similar documents.
    """

    def __init__(
        self,
        encoder: BaseEncoder,
        name: str = "cav_similarity_splitter",
        similarity_threshold: float = 0.45,
    ):
        super().__init__(
            name=name, 
            similarity_threshold=similarity_threshold,
            encoder=encoder
            )

    def __call__(self, docs: List[str]):
        total_docs = len(docs)
        splits = []
        curr_split_start_idx = 0
        curr_split_num = 1
        doc_embeds = self.encoder(docs)

        for idx in range(1, total_docs):
            curr_split_docs_embeds = doc_embeds[curr_split_start_idx : idx + 1]
            avg_embedding = np.mean(curr_split_docs_embeds, axis=0)

            # Compute the average embedding for the next two documents, if available
            if idx + 3 <= total_docs:  # Check if the next two indices are within the range
                next_doc_embeds = doc_embeds[idx + 1 : idx + 3]
                next_avg_embed = np.mean(next_doc_embeds, axis=0)
            elif idx + 2 <= total_docs:  # Check if the next index is within the range
                next_avg_embed = doc_embeds[idx + 1]
            else:
                next_avg_embed = None

            if next_avg_embed is not None:
                curr_sim_score = np.dot(avg_embedding, next_avg_embed) / (
                    np.linalg.norm(avg_embedding)
                    * np.linalg.norm(next_avg_embed)
                )

                if curr_sim_score < self.similarity_threshold:
                    splits.append(
                        DocumentSplit(
                            docs=list(docs[curr_split_start_idx : idx + 1]),
                            is_triggered=True,
                            triggered_score=curr_sim_score,
                        )
                    )
                    curr_split_start_idx = idx + 1
                    curr_split_num += 1

        splits.append(DocumentSplit(docs=list(docs[curr_split_start_idx:])))
        return splits