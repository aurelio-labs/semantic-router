from semantic_router.encoders.sentence_transformers import STEncoder


class LocalEncoder(STEncoder):
    """The local encoder uses the underlying STEncoder (ie a sentence-transformers
    bi-encoder). Designed as our recommended local encoder option for generating dense
    embeddings.
    """

    pass
