from typing import Tuple

import numpy as np
from numpy.linalg import norm


def similarity_matrix(xq: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Compute the similarity scores between a query vector and a set of vectors.

    Args:
        xq: A query vector (1d ndarray)
        index: A set of vectors.

    Returns:
        The similarity between the query vector and the set of vectors.
    """

    index_norm = norm(index, axis=1)
    xq_norm = norm(xq.T)
    sim = np.dot(index, xq.T) / (index_norm * xq_norm)
    return sim


def top_scores(sim: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    # get indices of top_k records
    top_k = min(top_k, sim.shape[0])
    idx = np.argpartition(sim, -top_k)[-top_k:]
    scores = sim[idx]

    return scores, idx
