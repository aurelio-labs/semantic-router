from datetime import datetime
from difflib import Differ
from enum import Enum
from typing import List, Optional, Union, Any, Dict, Tuple
from pydantic.v1 import BaseModel, Field


class EncoderType(Enum):
    AZURE = "azure"
    COHERE = "cohere"
    OPENAI = "openai"
    BM25 = "bm25"
    TFIDF = "tfidf"
    FASTEMBED = "fastembed"
    HUGGINGFACE = "huggingface"
    MISTRAL = "mistral"
    VIT = "vit"
    CLIP = "clip"
    GOOGLE = "google"
    BEDROCK = "bedrock"

    def to_list():
        return [encoder.value for encoder in EncoderType]


class EncoderInfo(BaseModel):
    name: str
    token_limit: int
    threshold: Optional[float] = None


class RouteChoice(BaseModel):
    name: Optional[str] = None
    function_call: Optional[List[Dict]] = None
    similarity_score: Optional[float] = None


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

    def to_mistral(self):
        return {"role": self.role, "content": self.content}

    def __str__(self):
        return f"{self.role}: {self.content}"


class DocumentSplit(BaseModel):
    docs: List[Union[str, Any]]
    is_triggered: bool = False
    triggered_score: Optional[float] = None
    token_count: Optional[int] = None
    metadata: Optional[Dict] = None

    @property
    def content(self) -> str:
        return " ".join([doc if isinstance(doc, str) else "" for doc in self.docs])


class ConfigParameter(BaseModel):
    field: str
    value: str
    namespace: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_pinecone(self, dimensions: int):
        if self.namespace is None:
            namespace = ""
        return {
            "id": f"{self.field}#{namespace}",
            "values": [0.1] * dimensions,
            "metadata": {
                "value": self.value,
                "created_at": self.created_at,
                "namespace": namespace,
                "field": self.field,
            },
        }


class Utterance(BaseModel):
    route: str
    utterance: str
    function_schemas: Optional[List[Dict]] = None
    metadata: Optional[Dict] = None
    diff_tag: str = " "

    @classmethod
    def from_tuple(cls, tuple_obj: Tuple):
        """Create an Utterance object from a tuple. The tuple must contain
        route and utterance as the first two elements. Then optionally
        function schemas and metadata as the third and fourth elements
        respectively. If this order is not followed an invalid Utterance
        object will be returned.

        :param tuple_obj: A tuple containing route, utterance, function schemas and metadata.
        :type tuple_obj: Tuple
        :return: An Utterance object.
        :rtype: Utterance
        """
        route, utterance = tuple_obj[0], tuple_obj[1]
        function_schemas = tuple_obj[2] if len(tuple_obj) > 2 else None
        metadata = tuple_obj[3] if len(tuple_obj) > 3 else None
        return cls(
            route=route,
            utterance=utterance,
            function_schemas=function_schemas,
            metadata=metadata
        )

    def to_tuple(self):
        return (
            self.route,
            self.utterance,
            self.function_schemas,
            self.metadata,
        )

    def to_str(self, include_metadata: bool = False):
        if include_metadata:
            return f"{self.route}: {self.utterance} | {self.function_schemas} | {self.metadata}"
        return f"{self.route}: {self.utterance}"

    def to_diff_str(self):
        return f"{self.diff_tag} {self.to_str()}"


class SyncMode(Enum):
    """Synchronization modes for local (route layer) and remote (index)
    instances.
    """
    ERROR = "error"
    REMOTE = "remote"
    LOCAL = "local"
    MERGE_FORCE_REMOTE = "merge-force-remote"
    MERGE_FORCE_LOCAL = "merge-force-local"
    MERGE = "merge"

    def to_list() -> List[str]:
        return [mode.value for mode in SyncMode]

class UtteranceDiff(BaseModel):
    diff: List[Utterance]

    @classmethod
    def from_utterances(
        cls,
        local_utterances: List[Utterance],
        remote_utterances: List[Utterance]
    ):
        local_utterances_map = {x.to_str(): x for x in local_utterances}
        remote_utterances_map = {x.to_str(): x for x in remote_utterances}
        # sort local and remote utterances
        local_utterances_str = list(local_utterances_map.keys())
        local_utterances_str.sort()
        remote_utterances_str = list(remote_utterances_map.keys())
        remote_utterances_str.sort()
        # get diff
        differ = Differ()
        diff_obj = list(differ.compare(local_utterances_str, remote_utterances_str))
        # create UtteranceDiff list
        utterance_diffs = []
        for line in diff_obj:
            utterance_str = line[2:]
            utterance_diff_tag = line[0]
            utterance = remote_utterances_map[utterance_str] if utterance_diff_tag == "+" else local_utterances_map[utterance_str]
            utterance.diff_tag = utterance_diff_tag
            utterance_diffs.append(utterance)
        return UtteranceDiff(diff=utterance_diffs)

    def to_utterance_str(self) -> List[str]:
        """Outputs the utterance diff as a list of diff strings. Returns a list
        of strings showing what is different in the remote when compared to the
        local. For example:

        ["  route1: utterance1",
         "  route1: utterance2",
         "- route2: utterance3",
         "- route2: utterance4"]

        Tells us that the remote is missing "route2: utterance3" and "route2:
        utterance4", which do exist locally. If we see:

        ["  route1: utterance1",
         "  route1: utterance2",
         "+ route2: utterance3",
         "+ route2: utterance4"]

        This diff tells us that the remote has "route2: utterance3" and
        "route2: utterance4", which do not exist locally.
        """
        return [x.to_diff_str() for x in self.diff]

    def get_tag(self, diff_tag: str) -> List[Utterance]:
        """Get all utterances with a given diff tag.

        :param diff_tag: The diff tag to filter by. Must be one of "+", "-", or
        " ".
        :type diff_tag: str
        :return: A list of Utterance objects.
        :rtype: List[Utterance]
        """
        if diff_tag not in ["+", "-", " "]:
            raise ValueError("diff_tag must be one of '+', '-', or ' '")
        return [x for x in self.diff if x.diff_tag == diff_tag]

    def get_sync_strategy(self, sync_mode: str) -> dict:
        """Generates the optimal synchronization plan for local and remote
        instances.

        :param sync_mode: The mode to sync the routes with the remote index.
        :type sync_mode: str
        :return: A dictionary describing the synchronization strategy.
        :rtype: dict
        """
        if sync_mode not in SyncMode.to_list():
            raise ValueError(f"sync_mode must be one of {SyncMode.to_list()}")
        local_only = self.get_tag("-")
        remote_only = self.get_tag("+")
        local_and_remote = self.get_tag(" ")
        if sync_mode == "error":
            if len(local_only) > 0 or len(remote_only) > 0:
                raise ValueError(
                    "There are utterances that exist in the local or remote "
                    "instance that do not exist in the other instance. Please "
                    "sync the routes before running this command."
                )
            else:
                return {
                    "remote": {
                        "upsert": [],
                        "delete": []
                    },
                    "local": {
                        "upsert": [],
                        "delete": []
                    }
                }
        elif sync_mode == "local":
            return {
                "remote": {
                    "upsert": local_only,
                    "delete": remote_only
                },
                "local": {
                    "upsert": [],
                    "delete": []
                }
            }
        elif sync_mode == "remote":
            return {
                "remote": {
                    "upsert": [],
                    "delete": []
                },
                "local": {
                    "upsert": remote_only,
                    "delete": local_only
                }
            }
        elif sync_mode == "merge-force-remote":
            # get set of route names that exist in both local and remote
            routes_in_both = set([utt.route for utt in local_and_remote])
            # get remote utterances that belong to routes_in_both
            remote_to_keep = [utt for utt in remote_only if utt.route in routes_in_both]
            # get remote utterances that do NOT belong to routes_in_both
            remote_to_delete = [utt for utt in remote_only if utt.route not in routes_in_both]
            return {
                "remote": {
                    "upsert": local_only,
                    "delete": remote_to_delete
                },
                "local": {
                    "upsert": remote_to_keep,
                    "delete": []
                }
            }
        elif sync_mode == "merge-force-local":
            # get set of route names that exist in both local and remote
            routes_in_both = set([utt.route for utt in local_and_remote])
            # get local utterances that belong to routes_in_both
            local_to_keep = [utt for utt in local_only if utt.route in routes_in_both]
            # get local utterances that do NOT belong to routes_in_both
            local_to_delete = [utt for utt in local_only if utt.route not in routes_in_both]
            return {
                "remote": {
                    "upsert": local_to_keep,
                    "delete": []
                },
                "local": {
                    "upsert": remote_only,
                    "delete": local_to_delete
                }
            }
        elif sync_mode == "merge":
            return {
                "remote": {
                    "upsert": local_only,
                    "delete": []
                },
                "local": {
                    "upsert": remote_only,
                    "delete": []
                }
            }
        


class Metric(Enum):
    COSINE = "cosine"
    DOTPRODUCT = "dotproduct"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
