from enum import Enum
from typing import List, Literal, Optional, Tuple

from pydantic.v1 import BaseModel
from pydantic.v1.dataclasses import dataclass

from semantic_router.encoders import (
    BaseEncoder,
    CohereEncoder,
    FastEmbedEncoder,
    OpenAIEncoder,
)
# from semantic_router.utils.splitters import DocumentSplit, semantic_splitter
from semantic_router.splitters.consecutive_sim import ConsecutiveSimSplitter
from semantic_router.splitters.cumulative_sim import CumulativeSimSplitter
from semantic_router.splitters.cav_sim import CAVSimSplitter


class EncoderType(Enum):
    HUGGINGFACE = "huggingface"
    FASTEMBED = "fastembed"
    OPENAI = "openai"
    COHERE = "cohere"


class RouteChoice(BaseModel):
    name: Optional[str] = None
    function_call: Optional[dict] = None
    similarity_score: Optional[float] = None
    trigger: Optional[bool] = None


@dataclass
class Encoder:
    type: EncoderType
    name: Optional[str]
    model: BaseEncoder

    def __init__(self, type: str, name: Optional[str]):
        self.type = EncoderType(type)
        self.name = name
        if self.type == EncoderType.HUGGINGFACE:
            raise NotImplementedError
        elif self.type == EncoderType.FASTEMBED:
            self.model = FastEmbedEncoder(name=name)
        elif self.type == EncoderType.OPENAI:
            self.model = OpenAIEncoder(name=name)
        elif self.type == EncoderType.COHERE:
            self.model = CohereEncoder(name=name)
        else:
            raise ValueError

    def __call__(self, texts: List[str]) -> List[List[float]]:
        return self.model(texts)


class Message(BaseModel):
    role: str
    content: str

    def to_openai(self):
        if self.role.lower() not in ["user", "assistant", "system"]:
            raise ValueError("Role must be either 'user', 'assistant' or 'system'")
        return {"role": self.role, "content": self.content}

    def to_cohere(self):
        return {"role": self.role, "message": self.content}

    def to_llamacpp(self):
        return {"role": self.role, "content": self.content}


class DocumentSplit(BaseModel):
    docs: List[str]
    is_triggered: bool = False
    triggered_score: Optional[float] = None

class Conversation(BaseModel):
    messages: List[Message]
    topics: List[Tuple[int, str]] = []
    splitter = None

    def add_new_messages(self, new_messages: List[Message]):
        self.messages.extend(new_messages)

    def configure_splitter(
        self,
        encoder: BaseEncoder,
        threshold: float = 0.5,
        split_method: Literal[
            "consecutive_similarity", "cumulative_similarity", "cav_similarity"
        ] = "consecutive_similarity",
    ):
        if split_method == "consecutive_similarity":
            self.splitter = ConsecutiveSimSplitter(encoder=encoder, similarity_threshold=threshold)
        elif split_method == "cumulative_similarity":
            self.splitter = CumulativeSimSplitter(encoder=encoder, similarity_threshold=threshold)
        elif split_method == "cav_similarity":
            self.splitter = CAVSimSplitter(encoder=encoder, similarity_threshold=threshold)
        else:
            raise ValueError(f"Invalid split method: {split_method}")

    def split_by_topic(self):
        if self.splitter is None:
            raise ValueError("Splitter is not configured. Please call configure_splitter first.")
        
        # Get the messages that haven't been clustered into topics yet
        unclustered_messages = self.messages[len(self.topics):]
        
        # Check if there are any messages that have been assigned topics
        if len(self.topics) >= 1:
            # Include the last message in the docs
            docs = [self.topics[-1][1]]
        else:
            # No messages have been assigned topics yet
            docs = []
        
        # Add the unclustered messages to the docs
        docs.extend([f"{m.role}: {m.content}" for m in unclustered_messages])
        
        # Use the splitter to split the documents
        new_topics = self.splitter(docs)
        
        # Check if the first new topic includes the first new message.
        # This means that the first new message shares the same topic as the last old message to have been assigned a topic.
        if docs[-len(unclustered_messages)] in new_topics[0].docs:
            start = self.topics[-1][0]
        else:
            start = len(self.topics) + 1
        
        # Add the new topics to the list of topics with unique IDs
        for i, topic in enumerate(new_topics, start=start):
            for message in topic.docs:
                self.topics.append((i, message))
        
        return new_topics