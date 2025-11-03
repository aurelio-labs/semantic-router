import json
from datetime import datetime, timezone
from difflib import Differ
from enum import Enum
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
from aurelio_sdk.schema import SparseEmbedding as BM25SparseEmbedding
from pydantic import BaseModel, ConfigDict, Field

from semantic_router.utils.logger import logger


class EncoderType(Enum):
    """The type of encoder."""

    AURELIO = "aurelio"
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
    LITELLM = "litellm"
    OLLAMA = "ollama"
    JINA = "jina_ai"
    VOYAGE = "voyage"
    NIM = "nvidia_nim"


class EncoderInfo(BaseModel):
    """Information about an encoder."""

    name: str
    token_limit: int
    threshold: Optional[float] = None


class RouteChoice(BaseModel):
    """A route choice typically output by the routers."""

    name: Optional[str] = None
    function_call: Optional[List[Dict]] = None
    similarity_score: Optional[float] = None


class Message(BaseModel):
    """A message in a conversation, includes the role and content fields."""

    role: str
    content: str

    def to_openai(self):
        """Convert the message to an OpenAI-compatible format."""
        if self.role.lower() not in ["user", "assistant", "system", "tool"]:
            raise ValueError(
                "Role must be either 'user', 'assistant', 'system' or 'tool'"
            )
        return {"role": self.role, "content": self.content}

    def to_cohere(self):
        """Convert the message to a Cohere-compatible format."""
        return {"role": self.role, "message": self.content}

    def to_llamacpp(self):
        """Convert the message to a LlamaCPP-compatible format."""
        return {"role": self.role, "content": self.content}

    def to_mistral(self):
        """Convert the message to a Mistral-compatible format."""
        return {"role": self.role, "content": self.content}

    def to_voyage(self):
        """Convert the message to a Voyage-compatible format."""
        return {"role": self.role, "content": self.content}

    def to_jina(self):
        """Convert the message to a Jina-compatible format."""
        return {"role": self.role, "content": self.content}

    def __str__(self):
        """Convert the message to a string."""
        return f"{self.role}: {self.content}"


class ConfigParameter(BaseModel):
    """A configuration parameter for a route. Used for remote router metadata such as
    router hashes, sync locks, etc.
    """

    field: str
    value: str
    scope: Optional[str] = None
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_pinecone(self, dimensions: int):
        """Convert the configuration parameter to a Pinecone-compatible format. Should
        be used when upserting configuration parameters to a separate config namespace
        within your Pinecone index.

        :param dimensions: The dimensions of the Pinecone index.
        :type dimensions: int
        :return: A Pinecone-compatible configuration parameter.
        :rtype: dict
        """
        namespace = self.scope or ""
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
    """An utterance in a conversation, includes the route, utterance, function
    schemas, metadata, and diff tag.
    """

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
        """Convert an Utterance object to a string. Used for comparisons during sync
        check operations.

        :param include_metadata: Whether to include metadata in the string.
        :type include_metadata: bool
        :return: A string representation of the Utterance object.
        :rtype: str
        """
        if include_metadata:
            # we sort the dicts to ensure consistent order as we need this to compare
            # stringified function schemas accurately
            if self.function_schemas is not None:
                function_schemas_sorted: List[str] | None = [
                    json.dumps(schema, sort_keys=True)
                    for schema in self.function_schemas
                ]
            else:
                function_schemas_sorted = None
            # we must do the same for metadata
            metadata_sorted = json.dumps(self.metadata, sort_keys=True)
            return f"{self.route}: {self.utterance} | {function_schemas_sorted} | {metadata_sorted}"
        return f"{self.route}: {self.utterance}"

    def to_diff_str(self, include_metadata: bool = False):
        return f"{self.diff_tag} {self.to_str(include_metadata=include_metadata)}"


class SyncMode(Enum):
    """Synchronization modes for local (route layer) and remote (index) instances."""

    ERROR = "error"
    REMOTE = "remote"
    LOCAL = "local"
    MERGE_FORCE_REMOTE = "merge-force-remote"
    MERGE_FORCE_LOCAL = "merge-force-local"
    MERGE = "merge"


SYNC_MODES = [x.value for x in SyncMode]


class UtteranceDiff(BaseModel):
    """A list of Utterance objects that represent the differences between local and
    remote utterances.
    """

    diff: List[Utterance]

    @classmethod
    def from_utterances(
        cls, local_utterances: List[Utterance], remote_utterances: List[Utterance]
    ):
        """Create a UtteranceDiff object from two lists of Utterance objects.

        :param local_utterances: A list of Utterance objects.
        :type local_utterances: List[Utterance]
        :param remote_utterances: A list of Utterance objects.
        :type remote_utterances: List[Utterance]
        """
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

        :param include_metadata: Whether to include metadata in the string.
        :type include_metadata: bool
        :return: A list of diff strings.
        :rtype: List[str]
        """
        return [x.to_diff_str(include_metadata=include_metadata) for x in self.diff]

    def get_tag(self, diff_tag: str) -> List[Utterance]:
        """Get all utterances with a given diff tag.

        :param diff_tag: The diff tag to filter by. Must be one of "+", "-", or " ".
        :type diff_tag: str
        :return: A list of Utterance objects.
        :rtype: List[Utterance]
        """
        if diff_tag not in ["+", "-", " "]:
            raise ValueError("diff_tag must be one of '+', '-', or ' '")
        return [x for x in self.diff if x.diff_tag == diff_tag]

    def get_sync_strategy(self, sync_mode: str) -> dict:
        """Generates the optimal synchronization plan for local and remote instances.

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
    """The metric to use in vector-based similarity search indexes."""

    COSINE = "cosine"
    DOTPRODUCT = "dotproduct"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"


class SparseEmbedding(BaseModel):
    """Sparse embedding interface. Primarily uses numpy operations for faster
    operations.
    """

    embedding: np.ndarray

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_compact_array(cls, array: np.ndarray):
        """Create a SparseEmbedding object from a compact array.

        :param array: A compact array.
        :type array: np.ndarray
        :return: A SparseEmbedding object.
        :rtype: SparseEmbedding
        """
        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError(
                f"Expected a 2D array with 2 columns, got a {array.ndim}D array with {array.shape[1]} columns. "
                "Column 0 should contain index positions, and column 1 should contain respective values."
            )
        return cls(embedding=array)

    @classmethod
    def from_vector(cls, vector: np.ndarray):
        """Consumes an array of sparse vectors containing zero-values.

        :param vector: A sparse vector.
        :type vector: np.ndarray
        :return: A SparseEmbedding object.
        :rtype: SparseEmbedding
        """
        if vector.ndim != 1:
            raise ValueError(f"Expected a 1D array, got a {vector.ndim}D array.")
        return cls.from_compact_array(np.array([np.arange(len(vector)), vector]).T)

    @classmethod
    def from_aurelio(cls, embedding: BM25SparseEmbedding):
        """Create a SparseEmbedding object from an AurelioSparseEmbedding object.

        :param embedding: An AurelioSparseEmbedding object.
        :type embedding: BM25SparseEmbedding
        :return: A SparseEmbedding object.
        :rtype: SparseEmbedding
        """
        arr = np.array([embedding.indices, embedding.values]).T
        return cls.from_compact_array(arr)

    @classmethod
    def from_dict(cls, sparse_dict: dict):
        """Create a SparseEmbedding object from a dictionary.

        :param sparse_dict: A dictionary of sparse values.
        :type sparse_dict: dict
        :return: A SparseEmbedding object.
        :rtype: SparseEmbedding
        """
        arr = np.array([list(sparse_dict.keys()), list(sparse_dict.values())]).T
        return cls.from_compact_array(arr)

    @classmethod
    def from_pinecone_dict(cls, sparse_dict: dict):
        """Create a SparseEmbedding object from a Pinecone dictionary.

        :param sparse_dict: A Pinecone dictionary.
        :type sparse_dict: dict
        :return: A SparseEmbedding object.
        :rtype: SparseEmbedding
        """
        arr = np.array([sparse_dict["indices"], sparse_dict["values"]]).T
        return cls.from_compact_array(arr)

    def to_dict(self):
        """Convert a SparseEmbedding object to a dictionary.

        :return: A dictionary of sparse values.
        :rtype: dict
        """
        return {
            i: v for i, v in zip(self.embedding[:, 0].astype(int), self.embedding[:, 1])
        }

    def to_pinecone(self):
        """Convert a SparseEmbedding object to a Pinecone dictionary.

        :return: A Pinecone dictionary.
        :rtype: dict
        """
        return {
            "indices": self.embedding[:, 0].astype(int).tolist(),
            "values": self.embedding[:, 1].tolist(),
        }

    # dictionary interface
    def items(self):
        """Return a list of (index, value) tuples from the SparseEmbedding object.

        :return: A list of (index, value) tuples.
        :rtype: list
        """
        return [
            (i, v)
            for i, v in zip(self.embedding[:, 0].astype(int), self.embedding[:, 1])
        ]
