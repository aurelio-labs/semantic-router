from datetime import datetime
from difflib import Differ
from enum import Enum
from typing import List, Optional, Union, Any, Dict, Tuple
from pydantic.v1 import BaseModel, Field
from semantic_router.utils.logger import logger


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
    utterance: Union[str, Any]
    function_schemas: Optional[List[Dict]] = None
    metadata: dict = {}
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
        if isinstance(function_schemas, dict):
            function_schemas = [function_schemas]
        metadata = tuple_obj[3] if len(tuple_obj) > 3 else {}
        return cls(
            route=route,
            utterance=utterance,
            function_schemas=function_schemas,
            metadata=metadata,
        )

    def to_tuple(self):
        """Convert an Utterance object to a tuple.

        :return: A tuple containing (route, utterance, function schemas, metadata).
        :rtype: Tuple
        """
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

    def to_diff_str(self, include_metadata: bool = False):
        return f"{self.diff_tag} {self.to_str(include_metadata=include_metadata)}"


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


SYNC_MODES = [x.value for x in SyncMode]


class UtteranceDiff(BaseModel):
    diff: List[Utterance]

    @classmethod
    def from_utterances(
        cls, local_utterances: List[Utterance], remote_utterances: List[Utterance]
    ):
        local_utterances_map = {
            x.to_str(include_metadata=True): x for x in local_utterances
        }
        remote_utterances_map = {
            x.to_str(include_metadata=True): x for x in remote_utterances
        }
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
            if utterance_diff_tag == "?":
                # this is a new line from diff string, we can ignore
                continue
            utterance = (
                remote_utterances_map[utterance_str]
                if utterance_diff_tag == "+"
                else local_utterances_map[utterance_str]
            )
            utterance.diff_tag = utterance_diff_tag
            utterance_diffs.append(utterance)
        return UtteranceDiff(diff=utterance_diffs)

    def to_utterance_str(self, include_metadata: bool = False) -> List[str]:
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
        return [x.to_diff_str(include_metadata=include_metadata) for x in self.diff]

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
        if sync_mode not in SYNC_MODES:
            raise ValueError(f"sync_mode must be one of {SYNC_MODES}")
        local_only = self.get_tag("-")
        local_only_mapper = {
            utt.route: (utt.function_schemas, utt.metadata) for utt in local_only
        }
        remote_only = self.get_tag("+")
        remote_only_mapper = {
            utt.route: (utt.function_schemas, utt.metadata) for utt in remote_only
        }
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
                    "remote": {"upsert": [], "delete": []},
                    "local": {"upsert": [], "delete": []},
                }
        elif sync_mode == "local":
            return {
                "remote": {
                    "upsert": local_only,  # + remote_updates,
                    "delete": remote_only,
                },
                "local": {"upsert": [], "delete": []},
            }
        elif sync_mode == "remote":
            return {
                "remote": {"upsert": [], "delete": []},
                "local": {"upsert": remote_only, "delete": local_only},
            }
        elif sync_mode == "merge-force-local":  # merge-to-local merge-join-local
            # PRIORITIZE LOCAL
            # get set of route names that exist in local (we keep these if
            # they are in remote)
            local_route_names = set([utt.route for utt in local_only])
            # if we see route: utterance exists in local, we do not pull it in
            # from remote
            local_route_utt_strs = set([utt.to_str() for utt in local_only])
            # get remote utterances that are in local
            remote_to_keep = [
                utt
                for utt in remote_only
                if (
                    utt.route in local_route_names
                    and utt.to_str() not in local_route_utt_strs
                )
            ]
            # overwrite remote routes with local metadata and function schemas
            logger.info(f"local_only_mapper: {local_only_mapper}")
            remote_to_update = [
                Utterance(
                    route=utt.route,
                    utterance=utt.utterance,
                    metadata=local_only_mapper[utt.route][1],
                    function_schemas=local_only_mapper[utt.route][0],
                )
                for utt in remote_only
                if (
                    utt.route in local_only_mapper
                    and (
                        utt.metadata != local_only_mapper[utt.route][1]
                        or utt.function_schemas != local_only_mapper[utt.route][0]
                    )
                )
            ]
            remote_to_keep = [
                Utterance(
                    route=utt.route,
                    utterance=utt.utterance,
                    metadata=local_only_mapper[utt.route][1],
                    function_schemas=local_only_mapper[utt.route][0],
                )
                for utt in remote_to_keep
                if utt.to_str() not in [x.to_str() for x in remote_to_update]
            ]
            # get remote utterances that are NOT in local
            remote_to_delete = [
                utt for utt in remote_only if utt.route not in local_route_names
            ]
            return {
                "remote": {
                    "upsert": local_only + remote_to_update,
                    "delete": remote_to_delete,
                },
                "local": {"upsert": remote_to_keep, "delete": []},
            }
        elif sync_mode == "merge-force-remote":  # merge-to-remote merge-join-remote
            # get set of route names that exist in remote (we keep these if
            # they are in local)
            remote_route_names = set([utt.route for utt in remote_only])
            # if we see route: utterance exists in remote, we do not pull it in
            # from local
            remote_route_utt_strs = set([utt.to_str() for utt in remote_only])
            # get local utterances that are in remote
            local_to_keep = [
                utt
                for utt in local_only
                if (
                    utt.route in remote_route_names
                    and utt.to_str() not in remote_route_utt_strs
                )
            ]
            # overwrite remote routes with local metadata and function schemas
            local_to_keep = [
                Utterance(
                    route=utt.route,
                    utterance=utt.utterance,
                    metadata=remote_only_mapper[utt.route][1],
                    function_schemas=remote_only_mapper[utt.route][0],
                )
                for utt in local_to_keep
            ]
            # get local utterances that are NOT in remote
            local_to_delete = [
                utt for utt in local_only if utt.route not in remote_route_names
            ]
            return {
                "remote": {"upsert": local_to_keep, "delete": []},
                "local": {"upsert": remote_only, "delete": local_to_delete},
            }
        elif sync_mode == "merge":
            # overwrite remote routes with local metadata and function schemas
            remote_only_updated = [
                (
                    Utterance(
                        route=utt.route,
                        utterance=utt.utterance,
                        metadata=local_only_mapper[utt.route][1],
                        function_schemas=local_only_mapper[utt.route][0],
                    )
                    if utt.route in local_only_mapper
                    else utt
                )
                for utt in remote_only
            ]
            # propogate same to shared routes
            shared_updated = [
                Utterance(
                    route=utt.route,
                    utterance=utt.utterance,
                    metadata=local_only_mapper[utt.route][1],
                    function_schemas=local_only_mapper[utt.route][0],
                )
                for utt in local_and_remote
                if (
                    utt.route in local_only_mapper
                    and (
                        utt.metadata != local_only_mapper[utt.route][1]
                        or utt.function_schemas != local_only_mapper[utt.route][0]
                    )
                )
            ]
            return {
                "remote": {
                    "upsert": local_only + shared_updated + remote_only_updated,
                    "delete": [],
                },
                "local": {"upsert": remote_only_updated + shared_updated, "delete": []},
            }
        else:
            raise ValueError(f"sync_mode must be one of {SYNC_MODES}")


class Metric(Enum):
    COSINE = "cosine"
    DOTPRODUCT = "dotproduct"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
