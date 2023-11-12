from pydantic import BaseModel

from semantic_router.retrievers import BaseRetriever
from semantic_router.rankers import BaseRanker
from semantic_router.schema import Decision


class BaseMatcher(BaseModel):
    retriever: BaseRetriever | None
    ranker: BaseRanker | None
    top_k: int | None
    top_n: int | None

    class Config:
        arbitrary_types_allowed = True

    def __call__(self, query: str, decisions: list[Decision]) -> str:
        raise NotImplementedError("Subclasses must implement this method")