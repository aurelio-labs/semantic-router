from itertools import cycle
from typing import List, Optional

from pydantic.v1 import BaseModel
from termcolor import colored

from semantic_router.encoders import BaseEncoder
from semantic_router.schema import DocumentSplit


class BaseSplitter(BaseModel):
    name: str
    encoder: BaseEncoder
    score_threshold: float
    min_split_tokens: Optional[int] = None
    max_split_tokens: Optional[int] = None

    def __call__(self, docs: List[str]) -> List[DocumentSplit]:
        raise NotImplementedError("Subclasses must implement this method")

    def print_splits(self, splits: list[DocumentSplit]):
        colors = cycle(["red", "green", "blue", "magenta", "cyan"])
        for i, split in enumerate(splits):
            triggered_text = (
                "Triggered " + format(split.triggered_score, ".2f")
                if split.triggered_score
                else "Not Triggered"
            )
            header = f"Split {i+1} - ({triggered_text})"
            if split.triggered_score:
                print(colored(header, "red"))
            else:
                print(colored(header, "blue"))
            print(colored(split.docs, next(colors)))  # type: ignore
            print("\n" + "-" * 50 + "\n")
