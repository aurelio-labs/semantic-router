from pydantic.v1 import BaseModel, Field
from typing import Union, List, Literal, Tuple
from semantic_router.splitters.consecutive_sim import ConsecutiveSimSplitter
from semantic_router.splitters.cumulative_sim import CumulativeSimSplitter
from semantic_router.splitters.running_avg_sim import RunningAvgSimSplitter
from semantic_router.encoders import BaseEncoder
from semantic_router.schema import Message

# Define a type alias for the splitter to simplify the annotation
SplitterType = Union[ConsecutiveSimSplitter, CumulativeSimSplitter, RunningAvgSimSplitter, None]

class Conversation(BaseModel):
    messages: List[Message] = Field(default_factory=list) # Ensure this is initialized as an empty list
    topics: List[Tuple[int, str]] = []
    splitter: SplitterType = None

    def add_new_messages(self, new_messages: List[Message]):
        self.messages.extend(new_messages)

    def remove_topics(self):
        self.topics = []

    def configure_splitter(
        self,
        encoder: BaseEncoder,
        threshold: float = 0.5,
        split_method: Literal[
            "consecutive_similarity", "cumulative_similarity", "running_avg_similarity"
        ] = "consecutive_similarity",
    ):
        if split_method == "consecutive_similarity":
            self.splitter = ConsecutiveSimSplitter(encoder=encoder, similarity_threshold=threshold)
        elif split_method == "cumulative_similarity":
            self.splitter = CumulativeSimSplitter(encoder=encoder, similarity_threshold=threshold)
        elif split_method == "running_avg_similarity":
            self.splitter = RunningAvgSimSplitter(encoder=encoder, similarity_threshold=threshold)
        else:
            raise ValueError(f"Invalid split method: {split_method}")
    

    def split_by_topic(self):
        if self.splitter is None:
            raise ValueError("Splitter is not configured. Please call configure_splitter first.")
        new_topics = []

        # Get the messages that haven't been clustered into topics yet
        unclustered_messages = self.messages[len(self.topics):]
        
        # If there are no unclustered messages, return early
        if not unclustered_messages:
            print("No unclustered messages to process.")
            return self.topics, new_topics

        # Extract the last topic ID and message from the previous splitting, if they exist.
        if self.topics:
            last_topic_id_from_last_splitting, last_message_from_last_splitting = self.topics[-1]
        else:
            last_topic_id_from_last_splitting, last_message_from_last_splitting = None, None

        # Initialize docs with the last message from the last topic if it exists
        docs = [last_message_from_last_splitting] if last_message_from_last_splitting else []
        
        # Add the unclustered messages to the docs
        docs.extend([f"{m.role}: {m.content}" for m in unclustered_messages])

        # Use the splitter to split the documents
        new_topics = self.splitter(docs)

        # Ensure there are new topics before proceeding
        if not new_topics:
            return self.topics, []
    
        # Check if there are any previously assigned topics
        if self.topics and new_topics:
            # Check if the first new topic includes the last message that was assigned a topic in the previous splitting.
            # This indicates that the new messages may continue the same topic as the last message from the previous split.
            if last_topic_id_from_last_splitting and last_message_from_last_splitting and last_message_from_last_splitting in new_topics[0].docs:
                start = last_topic_id_from_last_splitting
            else:
                start = self.topics[-1][0] + 1
        else:
            start = 0  # Start from 0 if no previous topics

        # If the last message from the previous splitting is found in the first new topic, remove it
        if self.topics and new_topics[0].docs[0] == self.topics[-1][1]:
            new_topics[0].docs.pop(0)

        # Add the new topics to the list of topics with unique IDs
        for i, topic in enumerate(new_topics, start=start):
            for message in topic.docs:
                self.topics.append((i, message))
        
        return self.topics, new_topics
   