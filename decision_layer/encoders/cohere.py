from decision_layer.encoders import BaseEncoder

class CohereEncoder(BaseEncoder):
    def __init__(self, name: str):
        super().__init__(name)

    def __call__(self, texts: list[str]) -> list[float]:
        raise NotImplementedError