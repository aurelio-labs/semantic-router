import numpy as np

from semantic_router.encoders import BaseEncoder
from semantic_router.encoders.cohere import CohereEncoder


def semantic_splitter(
    docs: list[str],
    encoder: BaseEncoder = CohereEncoder(),
    threshold: float = 0.5,
    split_method: str = "consecutive_similarity_drop",
) -> dict[str, list[str]]:
    """
    Splits a list of documents base on semantic similarity changes.

    Method 1: "consecutive_similarity_drop" - This method splits documents based on
    the changes in similarity scores between consecutive documents.
    Method 2: "cumulative_similarity_drop" - This method segments the documents based
    on the changes in cumulative similarity score of the documents within the same
    split.

    Args:
        encoder (BaseEncoder): Encoder for document embeddings.
        docs (list[str]): Documents to split.
        threshold (float): The similarity drop value that will trigger a new document
        split.
        split_method (str): The method to use for splitting.

    Returns:
        Dict[str, list[str]]: Splits with corresponding documents.
    """
    total_docs = len(docs)
    splits = {}
    curr_split_start_idx = 0
    curr_split_num = 1

    if split_method == "consecutive_similarity_drop":
        doc_embeds = encoder(docs)
        norm_embeds = doc_embeds / np.linalg.norm(doc_embeds, axis=1, keepdims=True)
        sim_matrix = np.matmul(norm_embeds, norm_embeds.T)

        for idx in range(1, total_docs):
            if idx < len(sim_matrix) and sim_matrix[idx - 1][idx] < threshold:
                splits[f"split {curr_split_num}"] = docs[curr_split_start_idx:idx]
                curr_split_start_idx = idx
                curr_split_num += 1

    elif split_method == "cumulative_similarity_drop":
        for idx in range(1, total_docs):
            if idx + 1 < total_docs:
                curr_split_docs = "\n".join(docs[curr_split_start_idx : idx + 1])
                next_doc = docs[idx + 1]

                curr_split_docs_embed = encoder([curr_split_docs])[0]
                next_doc_embed = encoder([next_doc])[0]

                similarity = np.dot(curr_split_docs_embed, next_doc_embed) / (
                    np.linalg.norm(curr_split_docs_embed)
                    * np.linalg.norm(next_doc_embed)
                )

                if similarity < threshold:
                    splits[f"split {curr_split_num}"] = docs[
                        curr_split_start_idx : idx + 1
                    ]
                    curr_split_start_idx = idx + 1
                    curr_split_num += 1

    else:
        raise ValueError(
            "Invalid 'split_method'. Choose either 'consecutive_similarity_drop' or"
            " 'cumulative_similarity_drop'."
        )

    splits[f"split {curr_split_num}"] = docs[curr_split_start_idx:]
    return splits


def colorize_splits(splits):
    colors = ["red", "blue", "green", "purple", "orange"]
    colorized_splits = {}
    for i, (split, text) in enumerate(splits.items()):
        color = colors[i % len(colors)]
        # Join the list of strings into a single string
        colorized_text = "".join(
            [f'<p style="color: {color};">{line}</p>' for line in text]
        )
        colorized_splits[split] = colorized_text
    return colorized_splits


def colorize_and_concatenate_splits(splits):
    colors = ["red", "blue", "green", "purple", "orange"]
    colorized_full_text = ""
    for i, (split, text) in enumerate(splits.items()):
        color = colors[i % len(colors)]
        # Join the list of strings into a single string and concatenate to the full text
        colorized_full_text += "".join(
            [f'<p style="color: {color};">{line}</p>' for line in text]
        )
    return colorized_full_text
