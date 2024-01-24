from typing import List, Literal, Optional

import numpy as np
from pydantic.v1 import BaseModel

from semantic_router.encoders import BaseEncoder


class DocumentSplit(BaseModel):
    docs: List[str]
    is_triggered: bool = False
    triggered_score: Optional[float] = None


def semantic_splitter(
    encoder: BaseEncoder,
    docs: List[str],
    threshold: float,
    split_method: Literal[
        "consecutive_similarity_drop", "cumulative_similarity_drop"
    ] = "consecutive_similarity_drop",
) -> List[DocumentSplit]:
    """
    Splits a list of documents base on semantic similarity changes.

    Method 1: "consecutive_similarity_drop" - This method splits documents based on
    the changes in similarity scores between consecutive documents.
    Method 2: "cumulative_similarity_drop" - This method segments the documents based
    on the changes in cumulative similarity score of the documents within the same
    split.

    Args:
        encoder (BaseEncoder): Encoder for document embeddings.
        docs (List[str]): Documents to split.
        threshold (float): The similarity drop value that will trigger a new document
        split.
        split_method (str): The method to use for splitting.

    Returns:
        Dict[str, List[str]]: Splits with corresponding documents.
    """
    total_docs = len(docs)
    splits = []
    curr_split_start_idx = 0
    curr_split_num = 1

    if split_method == "consecutive_similarity_drop":
        doc_embeds = encoder(docs)
        norm_embeds = doc_embeds / np.linalg.norm(doc_embeds, axis=1, keepdims=True)
        sim_matrix = np.matmul(norm_embeds, norm_embeds.T)

        for idx in range(1, total_docs):
            curr_sim_score = sim_matrix[idx - 1][idx]
            if idx < len(sim_matrix) and curr_sim_score < threshold:
                splits.append(
                    DocumentSplit(
                        docs=docs[curr_split_start_idx:idx],
                        is_triggered=True,
                        triggered_score=curr_sim_score,
                    )
                )
                curr_split_start_idx = idx
                curr_split_num += 1

    elif split_method == "cumulative_similarity_drop":
        for idx in range(1, total_docs):
            if idx + 1 < total_docs:
                curr_split_docs = "\n".join(docs[curr_split_start_idx : idx + 1])
                next_doc = docs[idx + 1]

                curr_split_docs_embed = encoder([curr_split_docs])[0]
                next_doc_embed = encoder([next_doc])[0]

                curr_sim_score = np.dot(curr_split_docs_embed, next_doc_embed) / (
                    np.linalg.norm(curr_split_docs_embed)
                    * np.linalg.norm(next_doc_embed)
                )

                if curr_sim_score < threshold:
                    splits.append(
                        DocumentSplit(
                            docs=docs[curr_split_start_idx : idx + 1],
                            is_triggered=True,
                            triggered_score=curr_sim_score,
                        )
                    )
                    curr_split_start_idx = idx + 1
                    curr_split_num += 1

    else:
        raise ValueError(
            "Invalid 'split_method'. Choose either 'consecutive_similarity_drop' or"
            " 'cumulative_similarity_drop'."
        )

    splits.append(DocumentSplit(docs=docs[curr_split_start_idx:]))
    return splits
