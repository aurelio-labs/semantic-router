from typing import List

from pydantic.v1 import BaseModel

from semantic_router.encoders import BaseEncoder
from semantic_router.schema import DocumentSplit
from termcolor import colored


class BaseSplitter(BaseModel):
    name: str
    encoder: BaseEncoder
    score_threshold: float

    def __call__(self, docs: List[str]) -> List[List[float]]:
        raise NotImplementedError("Subclasses must implement this method")

    def print_colored_splits(self, splits: List[DocumentSplit]):
        colors = ["red", "green", "blue", "magenta", "cyan"]
        for i, split in enumerate(splits):
            color = colors[i % len(colors)]
            for doc in split.docs:
                print(colored(doc, color))  # type: ignore
            print("Triggered score:", split.triggered_score)
            print("\n")
