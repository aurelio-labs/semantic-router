from semantic_router.retrievers import BaseRetriever


class HuggingFaceRetriever(BaseRetriever):
    def __init__(self, name: str):
        self.name = name

    def __call__(self, docs: list[str]) -> list[float]:
        raise NotImplementedError
