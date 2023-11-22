from semantic_router.encoders import BaseEncoder


class HuggingFaceEncoder(BaseEncoder):
    def __init__(self, name: str):
        self.name = name

    def __call__(self, docs: list[str]) -> list[float]:
        raise NotImplementedError
