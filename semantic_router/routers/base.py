import hashlib
import importlib
import json
import os
import random
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml  # type: ignore
from pydantic import BaseModel, ConfigDict, Field
from tqdm.auto import tqdm
from typing_extensions import deprecated

from semantic_router.encoders import (
    AutoEncoder,
    DenseEncoder,
    OpenAIEncoder,
    SparseEncoder,
)
from semantic_router.encoders.encode_input_type import EncodeInputType
from semantic_router.index.base import BaseIndex
from semantic_router.index.local import LocalIndex
from semantic_router.index.pinecone import PineconeIndex
from semantic_router.index.postgres import PostgresIndex
from semantic_router.index.qdrant import QdrantIndex
from semantic_router.llms import BaseLLM, OpenAILLM
from semantic_router.route import Route
from semantic_router.schema import (
    ConfigParameter,
    EncoderType,
    RouteChoice,
    Utterance,
    UtteranceDiff,
)
from semantic_router.utils.defaults import EncoderDefault
from semantic_router.utils.logger import logger


def is_valid(layer_config: str) -> bool:
    """Make sure the given string is json format and contains the 3 keys:
    ["encoder_name", "encoder_type", "routes"]"""
    try:
        output_json = json.loads(layer_config)
        required_keys = ["encoder_name", "encoder_type", "routes"]

        if isinstance(output_json, list):
            for item in output_json:
                missing_keys = [key for key in required_keys if key not in item]
                if missing_keys:
                    logger.warning(
                        f"Missing keys in layer config: {', '.join(missing_keys)}"
                    )
                    return False
            return True
        else:
            missing_keys = [key for key in required_keys if key not in output_json]
            if missing_keys:
                logger.warning(
                    f"Missing keys in layer config: {', '.join(missing_keys)}"
                )
                return False
            else:
                return True
    except json.JSONDecodeError as e:
        logger.error(e)
        return False


class RouterConfig:
    """Generates a RouterConfig object that can be used for initializing routers."""

    routes: List[Route] = Field(default_factory=list)

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        routes: List[Route] = [],
        encoder_type: str = "openai",
        encoder_name: Optional[str] = None,
    ):
        """Initialize a RouterConfig object.

        :param routes: A list of routes.
        :type routes: List[Route]
        :param encoder_type: The type of encoder to use.
        :type encoder_type: str
        :param encoder_name: The name of the encoder to use.
        :type encoder_name: Optional[str]
        """
        self.encoder_type = encoder_type
        if encoder_name is None:
            for encode_type in EncoderType:
                if encode_type.value == self.encoder_type:
                    if self.encoder_type == EncoderType.HUGGINGFACE.value:
                        raise NotImplementedError(
                            "HuggingFace encoder not supported by RouterConfig yet."
                        )
                    encoder_name = EncoderDefault[encode_type.name].value[
                        "embedding_model"
                    ]
                    break
            logger.info(f"Using default {encoder_type} encoder: {encoder_name}")
        self.encoder_name = encoder_name
        self.routes = routes

    @classmethod
    def from_file(cls, path: str) -> "RouterConfig":
        """Initialize a RouterConfig from a file. Expects a JSON or YAML file with file
        extension .json, .yaml, or .yml.

        :param path: The path to the file to load the RouterConfig from.
        :type path: str
        """
        logger.info(f"Loading route config from {path}")
        _, ext = os.path.splitext(path)
        with open(path, "r", encoding="utf-8") as f:
            if ext == ".json":
                layer = json.load(f)
            elif ext in [".yaml", ".yml"]:
                layer = yaml.safe_load(f)
            else:
                raise ValueError(
                    "Unsupported file type. Only .json and .yaml are supported"
                )

            if not is_valid(json.dumps(layer)):
                raise Exception("Invalid config JSON or YAML")

            encoder_type = layer["encoder_type"]
            encoder_name = layer["encoder_name"]
            routes = []
            for route_data in layer["routes"]:
                # Handle the 'llm' field specially if it exists
                if "llm" in route_data and route_data["llm"] is not None:
                    llm_data = route_data.pop(
                        "llm"
                    )  # Remove 'llm' from route_data and handle it separately
                    # Use the module path directly from llm_data without modification
                    llm_module_path = llm_data["module"]
                    # Dynamically import the module and then the class from that module
                    llm_module = importlib.import_module(llm_module_path)
                    llm_class = getattr(llm_module, llm_data["class"])
                    # Instantiate the LLM class with the provided model name
                    llm = llm_class(name=llm_data["model"])
                    # Reassign the instantiated llm object back to route_data
                    route_data["llm"] = llm

                # Dynamically create the Route object using the remaining route_data
                route = Route(**route_data)
                routes.append(route)

            return cls(
                encoder_type=encoder_type, encoder_name=encoder_name, routes=routes
            )

    @classmethod
    def from_tuples(
        cls,
        route_tuples: List[
            Tuple[str, str, Optional[List[Dict[str, Any]]], Dict[str, Any]]
        ],
        encoder_type: str = "openai",
        encoder_name: Optional[str] = None,
    ):
        """Initialize a RouterConfig from a list of tuples of routes and
        utterances.

        :param route_tuples: A list of tuples, each containing a route name and an
            associated utterance.
        :type route_tuples: List[Tuple[str, str]]
        :param encoder_type: The type of encoder to use, defaults to "openai".
        :type encoder_type: str, optional
        :param encoder_name: The name of the encoder to use, defaults to None.
        :type encoder_name: Optional[str], optional
        """
        routes_dict: Dict[str, Route] = {}
        # first create a dictionary of route names to Route objects
        # TODO: duplicated code with BaseIndex.get_routes()
        for route_name, utterance, function_schema, metadata in route_tuples:
            # if the route is not in the dictionary, add it
            if route_name not in routes_dict:
                routes_dict[route_name] = Route(
                    name=route_name,
                    utterances=[utterance],
                    function_schemas=function_schema,
                    metadata=metadata,
                )
            else:
                # otherwise, add the utterance to the route
                routes_dict[route_name].utterances.append(utterance)
        # then create a list of routes from the dictionary
        routes: List[Route] = []
        for route_name, route in routes_dict.items():
            routes.append(route)
        return cls(routes=routes, encoder_type=encoder_type, encoder_name=encoder_name)

    @classmethod
    def from_index(
        cls,
        index: BaseIndex,
        encoder_type: str = "openai",
        encoder_name: Optional[str] = None,
    ):
        """Initialize a RouterConfig from a BaseIndex object.

        :param index: The index to initialize the RouterConfig from.
        :type index: BaseIndex
        :param encoder_type: The type of encoder to use, defaults to "openai".
        :type encoder_type: str, optional
        :param encoder_name: The name of the encoder to use, defaults to None.
        :type encoder_name: Optional[str], optional
        """
        remote_routes = index.get_utterances(include_metadata=True)
        return cls.from_tuples(
            route_tuples=[utt.to_tuple() for utt in remote_routes],
            encoder_type=encoder_type,
            encoder_name=encoder_name,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the RouterConfig to a dictionary.

        :return: A dictionary representation of the RouterConfig.
        :rtype: Dict[str, Any]
        """
        return {
            "encoder_type": self.encoder_type,
            "encoder_name": self.encoder_name,
            "routes": [route.to_dict() for route in self.routes],
        }

    def to_file(self, path: str):
        """Save the routes to a file in JSON or YAML format.

        :param path: The path to save the RouterConfig to.
        :type path: str
        """
        logger.info(f"Saving route config to {path}")
        _, ext = os.path.splitext(path)

        # Check file extension before creating directories or files
        if ext not in [".json", ".yaml", ".yml"]:
            raise ValueError(
                "Unsupported file type. Only .json and .yaml are supported"
            )

        dir_name = os.path.dirname(path)

        # Create the directory if it doesn't exist and dir_name is not an empty string
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)

        with open(path, "w") as f:
            if ext == ".json":
                json.dump(self.to_dict(), f, indent=4)
            elif ext in [".yaml", ".yml"]:
                yaml.safe_dump(self.to_dict(), f)

    def to_utterances(self) -> List[Utterance]:
        """Convert the routes to a list of Utterance objects.

        :return: A list of Utterance objects.
        :rtype: List[Utterance]
        """
        utterances = []
        for route in self.routes:
            utterances.extend(
                [
                    Utterance(
                        route=route.name,
                        utterance=x,
                        function_schemas=route.function_schemas,
                        metadata=route.metadata or {},
                    )
                    for x in route.utterances
                ]
            )
        return utterances

    def add(self, route: Route):
        """Add a route to the RouterConfig.

        :param route: The route to add.
        :type route: Route
        """
        self.routes.append(route)
        logger.info(f"Added route `{route.name}`")

    def get(self, name: str) -> Optional[Route]:
        """Get a route from the RouterConfig by name.

        :param name: The name of the route to get.
        :type name: str
        :return: The route if found, otherwise None.
        :rtype: Optional[Route]
        """
        for route in self.routes:
            if route.name == name:
                return route
        logger.error(f"Route `{name}` not found")
        return None

    def remove(self, name: str):
        """Remove a route from the RouterConfig by name.

        :param name: The name of the route to remove.
        :type name: str
        """
        if name not in [route.name for route in self.routes]:
            logger.error(f"Route `{name}` not found")
        else:
            self.routes = [route for route in self.routes if route.name != name]
            logger.info(f"Removed route `{name}`")

    def get_hash(self) -> ConfigParameter:
        """Get the hash of the RouterConfig. Used for syncing.

        :return: The hash of the RouterConfig.
        :rtype: ConfigParameter
        """
        layer = self.to_dict()
        return ConfigParameter(
            field="sr_hash",
            value=hashlib.sha256(json.dumps(layer).encode()).hexdigest(),
        )


def xq_reshape(xq: List[float] | np.ndarray) -> np.ndarray:
    """Reshape the query vector to be a 2D numpy array.

    :param xq: The query vector.
    :type xq: List[float] | np.ndarray
    :return: The reshaped query vector.
    :rtype: np.ndarray
    """
    # convert to numpy array if not already
    if not isinstance(xq, np.ndarray):
        xq = np.array(xq)
    # check if vector is 1D and expand to 2D if necessary
    if len(xq.shape) == 1:
        xq = np.expand_dims(xq, axis=0)
    if xq.shape[0] != 1:
        raise ValueError(
            f"Expected (1, x) dimensional input for query, got {xq.shape}."
        )
    return xq


class BaseRouter(BaseModel):
    """Base class for all routers."""

    encoder: DenseEncoder = Field(default_factory=OpenAIEncoder)
    sparse_encoder: Optional[SparseEncoder] = Field(default=None)
    index: BaseIndex = Field(default_factory=BaseIndex)
    score_threshold: Optional[float] = Field(default=None)
    routes: List[Route] = Field(default_factory=list)
    llm: Optional[BaseLLM] = None
    top_k: int = 5
    aggregation: str = "mean"
    aggregation_method: Optional[Callable] = None
    auto_sync: Optional[str] = None
    init_async_index: bool = False

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        encoder: Optional[DenseEncoder] = None,
        sparse_encoder: Optional[SparseEncoder] = None,
        llm: Optional[BaseLLM] = None,
        routes: Optional[List[Route]] = None,
        index: Optional[BaseIndex] = None,  # type: ignore
        top_k: int = 5,
        aggregation: str = "mean",
        auto_sync: Optional[str] = None,
        init_async_index: bool = False,
    ):
        """Initialize a BaseRouter object. Expected to be used as a base class only,
        not directly instantiated.

        :param encoder: The encoder to use.
        :type encoder: Optional[DenseEncoder]
        :param sparse_encoder: The sparse encoder to use.
        :type sparse_encoder: Optional[SparseEncoder]
        :param llm: The LLM to use.
        :type llm: Optional[BaseLLM]
        :param routes: The routes to use.
        :type routes: Optional[List[Route]]
        :param index: The index to use.
        :type index: Optional[BaseIndex]
        :param top_k: The number of routes to return.
        :type top_k: int
        :param aggregation: The aggregation method to use.
        :type aggregation: str
        :param auto_sync: The auto sync mode to use.
        :type auto_sync: Optional[str]
        """
        routes = routes.copy() if routes else []
        super().__init__(
            encoder=encoder,
            sparse_encoder=sparse_encoder,
            llm=llm,
            routes=routes,
            index=index,
            top_k=top_k,
            aggregation=aggregation,
            auto_sync=auto_sync,
        )
        self.encoder = self._get_encoder(encoder=encoder)
        self.sparse_encoder = self._get_sparse_encoder(sparse_encoder=sparse_encoder)
        self.llm = llm
        self.routes = routes
        # initialize index
        self.index = self._get_index(index=index)
        # set score threshold using default method
        self._set_score_threshold()
        self.top_k = top_k
        if self.top_k < 1:
            raise ValueError(f"top_k needs to be >= 1, but was: {self.top_k}.")
        self.aggregation = aggregation
        if self.aggregation not in ["sum", "mean", "max"]:
            raise ValueError(
                f"Unsupported aggregation method chosen: {aggregation}. Choose either 'SUM', 'MEAN', or 'MAX'."
            )
        self.aggregation_method = self._set_aggregation_method(self.aggregation)
        # to keep PostgresIndex backwards compatible we will default auto_sync to "local",
        # this will be removed in 0.2.0
        if isinstance(self.index, PostgresIndex):
            self.auto_sync = "local"
        else:
            self.auto_sync = auto_sync

        # set route score thresholds if not already set
        for route in self.routes:
            if route.score_threshold is None:
                route.score_threshold = self.score_threshold
        # initialize index
        if not init_async_index:
            self._init_index_state()

    def _get_index(self, index: Optional[BaseIndex]) -> BaseIndex:
        """Get the index to use.

        :param index: The index to use.
        :type index: Optional[BaseIndex]
        :return: The index to use.
        :rtype: BaseIndex
        """
        if index is None:
            logger.warning("No index provided. Using default LocalIndex.")
            index = LocalIndex()
        else:
            index = index
        return index

    def _get_encoder(self, encoder: Optional[DenseEncoder]) -> DenseEncoder:
        """Get the dense encoder to be used for creating dense vector embeddings.

        :param encoder: The encoder to use.
        :type encoder: Optional[DenseEncoder]
        :return: The encoder to use.
        :rtype: DenseEncoder
        """
        if encoder is None:
            logger.warning("No encoder provided. Using default OpenAIEncoder.")
            encoder = OpenAIEncoder()
        else:
            encoder = encoder
        return encoder

    def _get_sparse_encoder(
        self, sparse_encoder: Optional[SparseEncoder]
    ) -> Optional[SparseEncoder]:
        """Get the sparse encoder to be used for creating sparse vector embeddings.

        :param sparse_encoder: The sparse encoder to use.
        :type sparse_encoder: Optional[SparseEncoder]
        :return: The sparse encoder to use.
        :rtype: Optional[SparseEncoder]
        """
        if sparse_encoder is None:
            return None
        raise NotImplementedError(
            f"Sparse encoder not implemented for {self.__class__.__name__}"
        )

    def _init_index_state(self):
        """Initializes an index (where required) and runs auto_sync if active."""
        # initialize index now, check if we need dimensions
        if self.index.dimensions is None:
            dims = len(self.encoder(["test"])[0])
            self.index.dimensions = dims
        # now init index
        if isinstance(self.index, PineconeIndex) or isinstance(
            self.index, PostgresIndex
        ):
            # _init_index will not create index if already exists — it will also check
            # for required attributes like self.index.host and self.index.dimensions and
            # fetch them if not set
            self.index.index = self.index._init_index(force_create=True)

        # run auto sync if active
        if self.auto_sync:
            local_utterances = self.to_config().to_utterances()
            remote_utterances = self.index.get_utterances(include_metadata=True)
            diff = UtteranceDiff.from_utterances(
                local_utterances=local_utterances,
                remote_utterances=remote_utterances,
            )
            sync_strategy = diff.get_sync_strategy(self.auto_sync)
            self._execute_sync_strategy(sync_strategy)

    async def _async_init_index_state(self):
        """Asynchronously initializes an index (where required) and runs auto_sync if active."""
        # initialize index now, check if we need dimensions
        if self.index is None or self.index.dimensions is None:
            dims = len(self.encoder(["test"])[0])
            self.index.dimensions = dims
        # now init index
        if isinstance(self.index, PineconeIndex) or isinstance(
            self.index, PostgresIndex
        ):
            await self.index._init_async_index(force_create=True)

        # run auto sync if active
        if self.auto_sync:
            local_utterances = self.to_config().to_utterances()
            remote_utterances = await self.index.aget_utterances(include_metadata=True)
            diff = UtteranceDiff.from_utterances(
                local_utterances=local_utterances,
                remote_utterances=remote_utterances,
            )
            sync_strategy = diff.get_sync_strategy(self.auto_sync)
            await self._async_execute_sync_strategy(sync_strategy)

    def _set_score_threshold(self):
        """Set the score threshold for the layer based on the encoder
        score threshold.

        When no score threshold is used a default `None` value
        is used, which means that a route will always be returned when
        the layer is called."""
        if self.encoder.score_threshold is not None:
            self.score_threshold = self.encoder.score_threshold
            if self.score_threshold is None:
                logger.warning(
                    "No score threshold value found in encoder. Using the default "
                    "'None' value can lead to unexpected results."
                )

    def check_for_matching_routes(self, top_class: str) -> Optional[Route]:
        """Check for a matching route in the routes list.

        :param top_class: The top class to check for.
        :type top_class: str
        :return: The matching route if found, otherwise None.
        :rtype: Optional[Route]
        """
        matching_route = next(
            (route for route in self.routes if route.name == top_class), None
        )
        if matching_route is None:
            logger.error(
                f"No route found with name {top_class}. Check to see if any Routes "
                "have been defined."
            )
            return None
        return matching_route

    def __call__(
        self,
        text: Optional[str] = None,
        vector: Optional[List[float] | np.ndarray] = None,
        simulate_static: bool = False,
        route_filter: Optional[List[str]] = None,
        limit: int | None = 1,
    ) -> RouteChoice | List[RouteChoice]:
        """Call the router to get a route choice.

        :param text: The text to route.
        :type text: Optional[str]
        :param vector: The vector to route.
        :type vector: Optional[List[float] | np.ndarray]
        :param simulate_static: Whether to simulate a static route.
        :type simulate_static: bool
        :param route_filter: The route filter to use.
        :type route_filter: Optional[List[str]]
        :param limit: The number of routes to return, defaults to 1. If set to None, no
            limit is applied and all routes are returned.
        :type limit: int | None
        :return: The route choice.
        :rtype: RouteChoice | List[RouteChoice]
        """
        if not self.index.is_ready():
            raise ValueError("Index is not ready.")
        # if no vector provided, encode text to get vector
        if vector is None:
            if text is None:
                raise ValueError("Either text or vector must be provided")
            vector = self._encode(text=[text], input_type="queries")
        # convert to numpy array if not already
        vector = xq_reshape(vector)
        # get scores and routes
        scores, routes = self.index.query(
            vector=vector[0], top_k=self.top_k, route_filter=route_filter
        )
        query_results = [
            {"route": d, "score": s.item()} for d, s in zip(routes, scores)
        ]
        # decide most relevant routes
        scored_routes = self._score_routes(query_results=query_results)
        return self._pass_routes(
            scored_routes=scored_routes,
            simulate_static=simulate_static,
            text=text,
            limit=limit,
        )

    def _pass_routes(
        self,
        scored_routes: List[Tuple[str, float, List[float]]],
        simulate_static: bool,
        text: Optional[str],
        limit: int | None,
    ) -> RouteChoice | list[RouteChoice]:
        """Returns a list of RouteChoice objects that passed the thresholds set.

        :param scored_routes: The scored routes to pass.
        :type scored_routes: List[Tuple[str, float, List[float]]]
        :param simulate_static: Whether to simulate a static route.
        :type simulate_static: bool
        :param text: The text to route.
        :type text: Optional[str]
        :param limit: The number of routes to return, defaults to 1. If set to None, no
            limit is applied and all routes are returned.
        :type limit: int | None
        :return: The route choice.
        :rtype: RouteChoice | list[RouteChoice]
        """
        passed_routes: list[RouteChoice] = []
        for route_name, total_score, scores in scored_routes:
            route = self.check_for_matching_routes(top_class=route_name)
            if route is None:
                # if no route is found we cannot use it
                continue
            if current_threshold := (
                route.score_threshold
                if route.score_threshold is not None
                else self.score_threshold
            ):
                # check if our route score exceeds the set threshold
                passed = total_score >= current_threshold
            else:
                # if no threshold is set, we always pass
                passed = True
            if passed and route is not None and not simulate_static:
                if route.function_schemas and text is None:
                    raise ValueError(
                        "Route has a function schema, but no text was provided."
                    )
                if route.function_schemas and not isinstance(route.llm, BaseLLM):
                    if not self.llm:
                        logger.warning(
                            "No LLM provided for dynamic route, will use OpenAI LLM "
                            "default. Ensure API key is set in OPENAI_API_KEY environment "
                            "variable."
                        )

                        self.llm = OpenAILLM()
                        route.llm = self.llm
                    else:
                        route.llm = self.llm
                # call dynamic route to generate the function_call content
                route_choice = route(query=text)
                if route_choice is not None and route_choice.similarity_score is None:
                    route_choice.similarity_score = total_score
                passed_routes.append(route_choice)
            elif passed and route is not None and simulate_static:
                passed_routes.append(
                    RouteChoice(
                        name=route.name,
                        function_call=None,
                        similarity_score=None,
                    )
                )
            if limit is None:
                # without limit we continue through all routes
                continue
            if len(passed_routes) >= limit:
                if limit == 1:
                    # maintain backward compatibility and return single route choice
                    return passed_routes[0]
                else:
                    # otherwise return all passed routes
                    return passed_routes
        if len(passed_routes) == 1:
            return passed_routes[0]
        elif len(passed_routes) > 1:
            return passed_routes
        else:
            # if no route passes threshold, return empty route choice
            return RouteChoice()

    async def _async_pass_routes(
        self,
        scored_routes: List[Tuple[str, float, List[float]]],
        simulate_static: bool,
        text: Optional[str],
        limit: int | None,
    ) -> RouteChoice | list[RouteChoice]:
        """Returns a list of RouteChoice objects that passed the thresholds set. Runs any
        dynamic route calls asynchronously. If there are no dynamic routes this method is
        equivalent to _pass_routes.

        :param scored_routes: The scored routes to pass.
        :type scored_routes: List[Tuple[str, float, List[float]]]
        :param simulate_static: Whether to simulate a static route.
        :type simulate_static: bool
        :param text: The text to route.
        :type text: Optional[str]
        :param limit: The number of routes to return, defaults to 1. If set to None, no
            limit is applied and all routes are returned.
        :type limit: int | None
        :return: The route choice.
        :rtype: RouteChoice | list[RouteChoice]
        """
        passed_routes: list[RouteChoice] = []
        for route_name, total_score, scores in scored_routes:
            route = self.check_for_matching_routes(top_class=route_name)
            if route is None:
                # if no route is found we cannot use it
                continue
            if current_threshold := (
                route.score_threshold
                if route.score_threshold is not None
                else self.score_threshold
            ):
                # check if our route score exceeds the set threshold
                passed = total_score >= current_threshold
            else:
                # if no threshold is set, we always pass
                passed = True
            if passed and route is not None and not simulate_static:
                if route.function_schemas and text is None:
                    raise ValueError(
                        "Route has a function schema, but no text was provided."
                    )
                if route.function_schemas and not isinstance(route.llm, BaseLLM):
                    if not self.llm:
                        logger.warning(
                            "No LLM provided for dynamic route, will use OpenAI LLM "
                            "default. Ensure API key is set in OPENAI_API_KEY environment "
                            "variable."
                        )

                        self.llm = OpenAILLM()
                        route.llm = self.llm
                    else:
                        route.llm = self.llm
                # TODO need to move to asyncio tasks and gather
                route_choice = await route.acall(query=text)
                if route_choice is not None and route_choice.similarity_score is None:
                    route_choice.similarity_score = total_score
                passed_routes.append(route_choice)
            elif passed and route is not None and simulate_static:
                passed_routes.append(
                    RouteChoice(
                        name=route.name,
                        function_call=None,
                        similarity_score=None,
                    )
                )
            if limit is None:
                # without limit we continue through all routes
                continue
            if len(passed_routes) >= limit:
                if limit == 1:
                    # maintain backward compatibility and return single route choice
                    return passed_routes[0]
                else:
                    # otherwise return all passed routes
                    return passed_routes
        if len(passed_routes) == 1:
            return passed_routes[0]
        elif len(passed_routes) > 1:
            return passed_routes
        else:
            # if no route passes threshold, return empty route choice
            return RouteChoice()

    async def acall(
        self,
        text: Optional[str] = None,
        vector: Optional[List[float] | np.ndarray] = None,
        limit: int | None = 1,
        simulate_static: bool = False,
        route_filter: Optional[List[str]] = None,
    ) -> RouteChoice | list[RouteChoice]:
        """Asynchronously call the router to get a route choice.

        :param text: The text to route.
        :type text: Optional[str]
        :param vector: The vector to route.
        :type vector: Optional[List[float] | np.ndarray]
        :param simulate_static: Whether to simulate a static route (ie avoid dynamic route
            LLM calls during fit or evaluate).
        :type simulate_static: bool
        :param route_filter: The route filter to use.
        :type route_filter: Optional[List[str]]
        :return: The route choice.
        :rtype: RouteChoice
        """
        if not (await self.index.ais_ready()):
            await self._async_init_index_state()
        # if no vector provided, encode text to get vector
        if vector is None:
            if text is None:
                raise ValueError("Either text or vector must be provided")
            vector = await self._async_encode(text=[text], input_type="queries")
        # convert to numpy array if not already
        vector = xq_reshape(vector)
        # get scores and routes
        scores, routes = await self.index.aquery(
            vector=vector[0], top_k=self.top_k, route_filter=route_filter
        )
        query_results = [
            {"route": d, "score": s.item()} for d, s in zip(routes, scores)
        ]
        scored_routes = self._score_routes(query_results=query_results)
        return await self._async_pass_routes(
            scored_routes=scored_routes,
            simulate_static=simulate_static,
            text=text,
            limit=limit,
        )

    def _index_ready(self) -> bool:
        """Method to check if the index is ready to be used.

        :return: True if the index is ready, False otherwise.
        :rtype: bool
        """
        if self.index.index is None or self.routes is None:
            return False
        if isinstance(self.index, QdrantIndex):
            info = self.index.describe()
            if info.vectors == 0:
                return False
        return True

    def sync(self, sync_mode: str, force: bool = False, wait: int = 0) -> List[str]:
        """Runs a sync of the local routes with the remote index.

        :param sync_mode: The mode to sync the routes with the remote index.
        :type sync_mode: str
        :param force: Whether to force the sync even if the local and remote
            hashes already match. Defaults to False.
        :type force: bool, optional
        :param wait: The number of seconds to wait for the index to be unlocked
        before proceeding with the sync. If set to 0, will raise an error if
        index is already locked/unlocked.
        :type wait: int
        :return: A list of diffs describing the addressed differences between
            the local and remote route layers.
        :rtype: List[str]
        """
        if not force and self.is_synced():
            logger.warning("Local and remote route layers are already synchronized.")
            # create utterance diff to return, but just using local instance
            # for speed
            local_utterances = self.to_config().to_utterances()
            diff = UtteranceDiff.from_utterances(
                local_utterances=local_utterances,
                remote_utterances=local_utterances,
            )
            return diff.to_utterance_str()
        # otherwise we continue with the sync, first locking the index
        try:
            diff_utt_str: list[str] = []
            _ = self.index.lock(value=True, wait=wait)
            try:
                # first creating a diff
                local_utterances = self.to_config().to_utterances()
                remote_utterances = self.index.get_utterances(include_metadata=True)
                diff = UtteranceDiff.from_utterances(
                    local_utterances=local_utterances,
                    remote_utterances=remote_utterances,
                )
                # generate sync strategy
                sync_strategy = diff.get_sync_strategy(sync_mode=sync_mode)
                # and execute
                self._execute_sync_strategy(sync_strategy)
                diff_utt_str = diff.to_utterance_str()
            except Exception as e:
                logger.error(f"Failed to create diff: {e}")
                raise e
            finally:
                # unlock index after sync
                _ = self.index.lock(value=False)
        except Exception as e:
            logger.error(f"Failed to lock index for sync: {e}")
            raise e
        return diff_utt_str

    async def async_sync(
        self, sync_mode: str, force: bool = False, wait: int = 0
    ) -> List[str]:
        """Runs a sync of the local routes with the remote index.

        :param sync_mode: The mode to sync the routes with the remote index.
        :type sync_mode: str
        :param force: Whether to force the sync even if the local and remote
            hashes already match. Defaults to False.
        :type force: bool, optional
        :param wait: The number of seconds to wait for the index to be unlocked
        before proceeding with the sync. If set to 0, will raise an error if
        index is already locked/unlocked.
        :type wait: int
        :return: A list of diffs describing the addressed differences between
            the local and remote route layers.
        :rtype: List[str]
        """
        if not force and await self.async_is_synced():
            logger.warning("Local and remote route layers are already synchronized.")
            # create utterance diff to return, but just using local instance
            # for speed
            local_utterances = self.to_config().to_utterances()
            diff = UtteranceDiff.from_utterances(
                local_utterances=local_utterances,
                remote_utterances=local_utterances,
            )
            return diff.to_utterance_str()
        # otherwise we continue with the sync, first locking the index
        try:
            diff_utt_str: list[str] = []
            _ = await self.index.alock(value=True, wait=wait)
            try:
                # first creating a diff
                local_utterances = self.to_config().to_utterances()
                remote_utterances = await self.index.aget_utterances(
                    include_metadata=True
                )
                diff = UtteranceDiff.from_utterances(
                    local_utterances=local_utterances,
                    remote_utterances=remote_utterances,
                )
                # generate sync strategy
                sync_strategy = diff.get_sync_strategy(sync_mode=sync_mode)
                # and execute
                await self._async_execute_sync_strategy(sync_strategy)
                diff_utt_str = diff.to_utterance_str()
            except Exception as e:
                logger.error(f"Failed to create diff: {e}")
                raise e
            finally:
                # unlock index after sync
                _ = await self.index.alock(value=False)
        except Exception as e:
            logger.error(f"Failed to lock index for sync: {e}")
            raise e
        return diff_utt_str

    def _execute_sync_strategy(self, strategy: Dict[str, Dict[str, List[Utterance]]]):
        """Executes the provided sync strategy, either deleting or upserting
        routes from the local and remote instances as defined in the strategy.

        :param strategy: The sync strategy to execute.
        :type strategy: Dict[str, Dict[str, List[Utterance]]]
        """
        if strategy["remote"]["delete"]:
            data_to_delete = {}  # type: ignore
            for utt_obj in strategy["remote"]["delete"]:
                data_to_delete.setdefault(utt_obj.route, []).append(utt_obj.utterance)
            # TODO: switch to remove without sync??
            self.index._remove_and_sync(data_to_delete)
        if strategy["remote"]["upsert"]:
            utterances_text = [utt.utterance for utt in strategy["remote"]["upsert"]]
            self.index.add(
                embeddings=self.encoder(utterances_text),
                routes=[utt.route for utt in strategy["remote"]["upsert"]],
                utterances=utterances_text,
                function_schemas=[
                    utt.function_schemas  # type: ignore
                    for utt in strategy["remote"]["upsert"]
                ],
                metadata_list=[utt.metadata for utt in strategy["remote"]["upsert"]],
            )
        if strategy["local"]["delete"]:
            self._local_delete(utterances=strategy["local"]["delete"])
        if strategy["local"]["upsert"]:
            self._local_upsert(utterances=strategy["local"]["upsert"])
        # update hash
        self._write_hash()

    async def _async_execute_sync_strategy(
        self, strategy: Dict[str, Dict[str, List[Utterance]]]
    ):
        """Executes the provided sync strategy, either deleting or upserting
        routes from the local and remote instances as defined in the strategy.

        :param strategy: The sync strategy to execute.
        :type strategy: Dict[str, Dict[str, List[Utterance]]]
        """
        if strategy["remote"]["delete"]:
            data_to_delete = {}  # type: ignore
            for utt_obj in strategy["remote"]["delete"]:
                data_to_delete.setdefault(utt_obj.route, []).append(utt_obj.utterance)
            # TODO: switch to remove without sync??
            await self.index._async_remove_and_sync(data_to_delete)
        if strategy["remote"]["upsert"]:
            utterances_text = [utt.utterance for utt in strategy["remote"]["upsert"]]
            await self.index.aadd(
                embeddings=await self.encoder.acall(docs=utterances_text),
                routes=[utt.route for utt in strategy["remote"]["upsert"]],
                utterances=utterances_text,
                function_schemas=[
                    utt.function_schemas  # type: ignore
                    for utt in strategy["remote"]["upsert"]
                ],
                metadata_list=[utt.metadata for utt in strategy["remote"]["upsert"]],
            )
        if strategy["local"]["delete"]:
            # assumption is that with simple local delete we don't benefit from async
            self._local_delete(utterances=strategy["local"]["delete"])
        if strategy["local"]["upsert"]:
            # same assumption as with local delete above
            self._local_upsert(utterances=strategy["local"]["upsert"])
        # update hash
        await self._async_write_hash()

    def _local_upsert(self, utterances: List[Utterance]):
        """Adds new routes to the SemanticRouter.

        :param utterances: The utterances to add to the local SemanticRouter.
        :type utterances: List[Utterance]
        """
        new_routes = {route.name: route for route in self.routes}
        for utt_obj in utterances:
            if utt_obj.route not in new_routes.keys():
                new_routes[utt_obj.route] = Route(
                    name=utt_obj.route,
                    utterances=[utt_obj.utterance],
                    function_schemas=utt_obj.function_schemas,
                    metadata=utt_obj.metadata,
                )
            else:
                if utt_obj.utterance not in new_routes[utt_obj.route].utterances:
                    new_routes[utt_obj.route].utterances.append(utt_obj.utterance)
                new_routes[utt_obj.route].function_schemas = utt_obj.function_schemas
                new_routes[utt_obj.route].metadata = utt_obj.metadata
        self.routes = list(new_routes.values())

    def _local_delete(self, utterances: List[Utterance]):
        """Deletes routes from the local SemanticRouter.

        :param utterances: The utterances to delete from the local SemanticRouter.
        :type utterances: List[Utterance]
        """
        # create dictionary of route names to utterances
        route_dict: dict[str, List[str]] = {}
        for utt in utterances:
            route_dict.setdefault(utt.route, []).append(utt.utterance)
        # iterate over current routes and delete specific utterance if found
        new_routes = []
        for route in self.routes:
            if route.name in route_dict.keys():
                # drop utterances that are in route_dict deletion list
                new_utterances = list(
                    set(route.utterances) - set(route_dict[route.name])
                )
                if len(new_utterances) == 0:
                    # the route is now empty, so we skip it
                    continue
                else:
                    new_routes.append(
                        Route(
                            name=route.name,
                            utterances=new_utterances,
                            # use existing function schemas and metadata
                            function_schemas=route.function_schemas,
                            metadata=route.metadata,
                        )
                    )
            else:
                # the route is not in the route_dict, so we keep it as is
                new_routes.append(route)

        self.routes = new_routes

    def __str__(self):
        return (
            f"{self.__class__.__name__}(encoder={self.encoder}, "
            f"score_threshold={self.score_threshold}, "
            f"routes={self.routes})"
        )

    @classmethod
    def from_json(cls, file_path: str):
        """Load a RouterConfig from a JSON file.

        :param file_path: The path to the JSON file.
        :type file_path: str
        :return: The RouterConfig object.
        :rtype: RouterConfig
        """
        config = RouterConfig.from_file(file_path)
        encoder = AutoEncoder(type=config.encoder_type, name=config.encoder_name).model
        if isinstance(encoder, DenseEncoder):
            return cls(encoder=encoder, routes=config.routes)
        else:
            raise ValueError(f"{type(encoder)} not supported for loading from JSON.")

    @classmethod
    def from_yaml(cls, file_path: str):
        """Load a RouterConfig from a YAML file.

        :param file_path: The path to the YAML file.
        :type file_path: str
        :return: The RouterConfig object.
        :rtype: RouterConfig
        """
        config = RouterConfig.from_file(file_path)
        encoder = AutoEncoder(type=config.encoder_type, name=config.encoder_name).model
        if isinstance(encoder, DenseEncoder):
            return cls(encoder=encoder, routes=config.routes)
        else:
            raise ValueError(f"{type(encoder)} not supported for loading from YAML.")

    @classmethod
    def from_config(cls, config: RouterConfig, index: Optional[BaseIndex] = None):
        """Create a Router from a RouterConfig object.

        :param config: The RouterConfig object.
        :type config: RouterConfig
        :param index: The index to use.
        :type index: Optional[BaseIndex]
        """
        encoder = AutoEncoder(type=config.encoder_type, name=config.encoder_name).model
        if isinstance(encoder, DenseEncoder):
            return cls(encoder=encoder, routes=config.routes, index=index)
        else:
            raise ValueError(f"{type(encoder)} not supported for loading from config.")

    def add(self, routes: List[Route] | Route):
        """Add a route to the local SemanticRouter and index.

        :param route: The route to add.
        :type route: Route
        """
        raise NotImplementedError("This method must be implemented by subclasses.")

    async def aadd(self, routes: List[Route] | Route):
        """Add a route to the local SemanticRouter and index asynchronously.

        :param route: The route to add.
        :type route: Route
        """
        logger.warning("Async method not implemented.")
        return self.add(routes)

    def list_route_names(self) -> List[str]:
        return [route.name for route in self.routes]

    def update(
        self,
        name: str,
        threshold: Optional[float] = None,
        utterances: Optional[List[str]] = None,
    ):
        """Updates the route specified in name. Allows the update of
        threshold and/or utterances. If no values are provided via the
        threshold or utterances parameters, those fields are not updated.
        If neither field is provided raises a ValueError.

        The name must exist within the local SemanticRouter, if not a
        KeyError will be raised.

        :param name: The name of the route to update.
        :type name: str
        :param threshold: The threshold to update.
        :type threshold: Optional[float]
        :param utterances: The utterances to update.
        :type utterances: Optional[List[str]]
        """
        # TODO JB: should modify update to take a Route object
        current_local_hash = self._get_hash()
        current_remote_hash = self.index._read_hash()
        if current_remote_hash.value == "":
            # if remote hash is empty, the index is to be initialized
            current_remote_hash = current_local_hash

        if threshold is None and utterances is None:
            raise ValueError(
                "At least one of 'threshold' or 'utterances' must be provided."
            )
        if utterances:
            raise NotImplementedError(
                "The update method cannot be used for updating utterances yet."
            )

        route = self.get(name)
        if route:
            if threshold:
                old_threshold = route.score_threshold
                route.score_threshold = threshold
                logger.info(
                    f"Updated threshold for route '{route.name}' from {old_threshold} to {threshold}"
                )
        else:
            raise ValueError(f"Route '{name}' not found. Nothing updated.")

        if current_local_hash.value == current_remote_hash.value:
            self._write_hash()  # update current hash in index
        else:
            logger.warning(
                "Local and remote route layers were not aligned. Remote hash "
                f"not updated. Use `{self.__class__.__name__}.get_utterance_diff()` "
                "to see details."
            )

    def delete(self, route_name: str):
        """Deletes a route given a specific route name.

        :param route_name: the name of the route to be deleted
        :type str:
        """
        # ensure index is not locked
        if self.index._is_locked():
            raise ValueError("Index is locked. Cannot delete route.")
        current_local_hash = self._get_hash()
        current_remote_hash = self.index._read_hash()
        if current_remote_hash.value == "":
            # if remote hash is empty, the index is to be initialized
            current_remote_hash = current_local_hash

        if route_name not in [route.name for route in self.routes]:
            err_msg = f"Route `{route_name}` not found in {self.__class__.__name__}"
            logger.warning(err_msg)
            try:
                self.index.delete(route_name=route_name)
            except Exception as e:
                logger.error(f"Failed to delete route from the index: {e}")
        else:
            self.routes = [route for route in self.routes if route.name != route_name]
            self.index.delete(route_name=route_name)

        if current_local_hash.value == current_remote_hash.value:
            self._write_hash()  # update current hash in index
        else:
            logger.warning(
                "Local and remote route layers were not aligned. Remote hash "
                f"not updated. Use `{self.__class__.__name__}.get_utterance_diff()` "
                "to see details."
            )

    async def adelete(self, route_name: str):
        """Deletes a route given a specific route name asynchronously.

        :param route_name: the name of the route to be deleted
        :type str:
        """
        # ensure index is not locked
        if await self.index._ais_locked():
            raise ValueError("Index is locked. Cannot delete route.")
        current_local_hash = self._get_hash()
        current_remote_hash = await self.index._async_read_hash()
        if current_remote_hash.value == "":
            # if remote hash is empty, the index is to be initialized
            current_remote_hash = current_local_hash

        if route_name not in [route.name for route in self.routes]:
            err_msg = f"Route `{route_name}` not found in {self.__class__.__name__}"
            logger.warning(err_msg)
            try:
                await self.index.adelete(route_name=route_name)
            except Exception as e:
                logger.error(f"Failed to delete route from the index: {e}")
        else:
            self.routes = [route for route in self.routes if route.name != route_name]
            await self.index.adelete(route_name=route_name)

        if current_local_hash.value == current_remote_hash.value:
            await self._async_write_hash()  # update current hash in index
        else:
            logger.warning(
                "Local and remote route layers were not aligned. Remote hash "
                f"not updated. Use `{self.__class__.__name__}.get_utterance_diff()` "
                "to see details."
            )

    def _refresh_routes(self):
        """Pulls out the latest routes from the index.

        Not yet implemented for BaseRouter.
        """
        raise NotImplementedError("This method has not yet been implemented.")
        route_mapping = {route.name: route for route in self.routes}
        index_routes = self.index.get_utterances()
        new_routes_names = []
        new_routes = []
        for route_name, utterance in index_routes:
            if route_name in route_mapping:
                if route_name not in new_routes_names:
                    existing_route = route_mapping[route_name]
                    new_routes.append(existing_route)

                new_routes.append(Route(name=route_name, utterances=[utterance]))
            route = route_mapping[route_name]
            self.routes.append(route)

    def _get_hash(self) -> ConfigParameter:
        """Get the hash of the current routes.

        :return: The hash of the current routes.
        :rtype: ConfigParameter
        """
        config = self.to_config()
        return config.get_hash()

    def _write_hash(self) -> ConfigParameter:
        """Write the hash of the current routes to the index.

        :return: The hash of the current routes.
        :rtype: ConfigParameter
        """
        config = self.to_config()
        hash_config = config.get_hash()
        self.index._write_config(config=hash_config)
        return hash_config

    async def _async_write_hash(self) -> ConfigParameter:
        """Write the hash of the current routes to the index asynchronously.

        :return: The hash of the current routes.
        :rtype: ConfigParameter
        """
        config = self.to_config()
        hash_config = config.get_hash()
        await self.index._async_write_config(config=hash_config)
        return hash_config

    def is_synced(self) -> bool:
        """Check if the local and remote route layer instances are
        synchronized.

        :return: True if the local and remote route layers are synchronized,
            False otherwise.
        :rtype: bool
        """
        # first check hash
        local_hash = self._get_hash()
        remote_hash = self.index._read_hash()
        if local_hash.value == remote_hash.value:
            return True
        else:
            return False

    async def async_is_synced(self) -> bool:
        """Check if the local and remote route layer instances are
        synchronized asynchronously.

        :return: True if the local and remote route layers are synchronized,
            False otherwise.
        :rtype: bool
        """
        # first check hash
        local_hash = self._get_hash()
        remote_hash = await self.index._async_read_hash()
        if local_hash.value == remote_hash.value:
            return True
        else:
            return False

    def get_utterance_diff(self, include_metadata: bool = False) -> List[str]:
        """Get the difference between the local and remote utterances. Returns
        a list of strings showing what is different in the remote when compared
        to the local. For example:

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

        :param include_metadata: Whether to include metadata in the diff.
        :type include_metadata: bool
        :return: A list of strings showing the difference between the local and remote
            utterances.
        :rtype: List[str]
        """
        # first we get remote and local utterances
        remote_utterances = self.index.get_utterances(include_metadata=include_metadata)
        local_utterances = self.to_config().to_utterances()

        diff_obj = UtteranceDiff.from_utterances(
            local_utterances=local_utterances, remote_utterances=remote_utterances
        )
        return diff_obj.to_utterance_str(include_metadata=include_metadata)

    async def aget_utterance_diff(self, include_metadata: bool = False) -> List[str]:
        """Get the difference between the local and remote utterances asynchronously.
        Returns a list of strings showing what is different in the remote when
        compared to the local. For example:

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

        :param include_metadata: Whether to include metadata in the diff.
        :type include_metadata: bool
        :return: A list of strings showing the difference between the local and remote
            utterances.
        :rtype: List[str]
        """
        # first we get remote and local utterances
        remote_utterances = await self.index.aget_utterances(
            include_metadata=include_metadata
        )
        local_utterances = self.to_config().to_utterances()

        diff_obj = UtteranceDiff.from_utterances(
            local_utterances=local_utterances, remote_utterances=remote_utterances
        )
        return diff_obj.to_utterance_str(include_metadata=include_metadata)

    def _extract_routes_details(
        self, routes: List[Route], include_metadata: bool = False
    ) -> Tuple:
        """Extract the routes details.

        :param routes: The routes to extract the details from.
        :type routes: List[Route]
        :param include_metadata: Whether to include metadata in the details.
        :type include_metadata: bool
        :return: A tuple of the route names, utterances, and function schemas.
        """

        route_names = [route.name for route in routes for _ in route.utterances]
        utterances = [utterance for route in routes for utterance in route.utterances]
        function_schemas = [
            (
                route.function_schemas[0]
                if route.function_schemas and len(route.function_schemas) > 0
                else {}
            )
            for route in routes
            for _ in route.utterances
        ]

        if include_metadata:
            metadata = [route.metadata for route in routes for _ in route.utterances]
            return route_names, utterances, function_schemas, metadata
        return route_names, utterances, function_schemas

    def _encode(
        self,
        text: list[str],
        input_type: EncodeInputType,
    ) -> Any:
        """Generates embeddings for a given text.

        Must be implemented by a subclass.

        :param text: The text to encode.
        :type text: list[str]
        :param input_type: Specify whether encoding 'queries' or 'documents', used in asymmetric retrieval
        :type input_type: semantic_router.encoders.encode_input_type.EncodeInputType
        :return: The embeddings of the text.
        :rtype: Any
        """
        # TODO: should encode "content" rather than text
        raise NotImplementedError("This method should be implemented by subclasses.")

    async def _async_encode(
        self,
        text: list[str],
        input_type: EncodeInputType,
    ) -> Any:
        """Asynchronously generates embeddings for a given text.

        Must be implemented by a subclass.

        :param text: The text to encode.
        :type text: list[str]
        :param input_type: Specify whether encoding 'queries' or 'documents', used in asymmetric retrieval
        :type input_type: semantic_router.encoders.encode_input_type.EncodeInputType
        :return: The embeddings of the text.
        :rtype: Any
        """
        # TODO: should encode "content" rather than text
        raise NotImplementedError("This method should be implemented by subclasses.")

    def _set_aggregation_method(self, aggregation: str = "sum"):
        """Set the aggregation method.

        :param aggregation: The aggregation method to use.
        :type aggregation: str
        :return: The aggregation method.
        :rtype: Callable
        """
        # TODO is this really needed?
        if aggregation == "sum":
            return lambda x: sum(x)
        elif aggregation == "mean":
            return lambda x: np.mean(x)
        elif aggregation == "max":
            return lambda x: max(x)
        else:
            raise ValueError(
                f"Unsupported aggregation method chosen: {aggregation}. Choose either 'SUM', 'MEAN', or 'MAX'."
            )

    def _score_routes(
        self, query_results: list[dict]
    ) -> list[tuple[str, float, list[float]]]:
        """Score the routes based on the query results.

        :param query_results: The query results to score.
        :type query_results: List[Dict]
        :return: A tuple of routes, their total scores, and their individual scores.
        """
        scores_by_class = self.group_scores_by_class(query_results)
        if self.aggregation_method is None:
            raise ValueError("self.aggregation_method is not set.")

        # Calculate total score for each class
        total_scores = [
            (route, self.aggregation_method(scores), scores)
            for route, scores in scores_by_class.items()
        ]
        # order from highest to lowest score
        total_scores.sort(key=lambda x: x[1], reverse=True)
        return total_scores

    # TODO JB allow return of multiple routes
    @deprecated(
        "Direct use of `_semantic_classify` is deprecated. Use `__call__` or `acall` instead."
    )
    def _semantic_classify(self, query_results: List[Dict]) -> Tuple[str, List[float]]:
        """Classify the query results into a single class based on the highest total score.
        If no classification is found, return an empty string and an empty list.

        :param query_results: The query results to classify. Expected format is a list of
        dictionaries with "route" and "score" keys.
        :type query_results: List[Dict]
        :return: A tuple containing the top class and its associated scores.
        :rtype: Tuple[str, List[float]]
        """
        top_class, top_score, scores = self._score_routes(query_results)[0]

        # Return the top class and its associated scores
        if top_class is not None:
            return str(top_class), scores
        else:
            logger.warning("No classification found for semantic classifier.")
            return "", []

    def get(self, name: str) -> Optional[Route]:
        """Get a route by name.

        :param name: The name of the route to get.
        :type name: str
        :return: The route.
        :rtype: Optional[Route]
        """
        for route in self.routes:
            if route.name == name:
                return route
        logger.error(f"Route `{name}` not found")
        return None

    def group_scores_by_class(
        self, query_results: List[Dict]
    ) -> Dict[str, List[float]]:
        """Group the scores by class.

        :param query_results: The query results to group. Expected format is a list of
        dictionaries with "route" and "score" keys.
        :type query_results: List[Dict]
        :return: A dictionary of route names and their associated scores.
        :rtype: Dict[str, List[float]]
        """
        scores_by_class: Dict[str, List[float]] = {}
        for result in query_results:
            score = result["score"]
            route = result["route"]
            if route in scores_by_class:
                scores_by_class[route].append(score)
            else:
                scores_by_class[route] = [score]
        return scores_by_class

    def _update_thresholds(self, route_thresholds: Optional[Dict[str, float]] = None):
        """Update the score thresholds for each route using a dictionary of
        route names and thresholds.

        :param route_thresholds: A dictionary of route names and thresholds.
        :type route_thresholds: Dict[str, float] | None
        """
        if route_thresholds:
            for route, threshold in route_thresholds.items():
                self.set_threshold(
                    threshold=threshold,
                    route_name=route,
                )

    def set_threshold(self, threshold: float, route_name: str | None = None):
        """Set the score threshold for a specific route or all routes. A `threshold` of 0.0
        will mean that the route will be returned no matter how low it scores whereas
        a threshold of 1.0 will mean that a route must contain an exact utterance match
        to be returned.

        :param threshold: The threshold to set.
        :type threshold: float
        :param route_name: The name of the route to set the threshold for. If None, the
        threshold will be set for all routes.
        :type route_name: str | None
        """
        if route_name is None:
            for route in self.routes:
                route.score_threshold = threshold
            # set the router threshold too
            self.score_threshold = threshold
        else:
            route_get: Route | None = self.get(route_name)
            if route_get is not None:
                route_get.score_threshold = threshold
            else:
                logger.error(f"Route `{route_name}` not found")

    def to_config(self) -> RouterConfig:
        """Convert the router to a RouterConfig object.

        :return: The RouterConfig object.
        :rtype: RouterConfig
        """
        return RouterConfig(
            encoder_type=self.encoder.type,
            encoder_name=self.encoder.name,
            routes=self.routes,
        )

    def to_json(self, file_path: str):
        """Convert the router to a JSON file.

        :param file_path: The path to the JSON file.
        :type file_path: str
        """
        config = self.to_config()
        config.to_file(file_path)

    def to_yaml(self, file_path: str):
        """Convert the router to a YAML file.

        :param file_path: The path to the YAML file.
        :type file_path: str
        """
        config = self.to_config()
        config.to_file(file_path)

    def get_thresholds(self) -> Dict[str, float]:
        """Get the score thresholds for each route.

        :return: A dictionary of route names and their associated thresholds.
        :rtype: Dict[str, float]
        """
        thresholds = {
            route.name: route.score_threshold or self.score_threshold or 0.0
            for route in self.routes
        }
        return thresholds

    def fit(
        self,
        X: List[str],
        y: List[str],
        batch_size: int = 500,
        max_iter: int = 500,
        local_execution: bool = False,
    ):
        """Fit the router to the data. Works best with a large number of examples for each
        route and with many `None` utterances.

        :param X: The input data.
        :type X: List[str]
        :param y: The output data.
        :type y: List[str]
        :param batch_size: The batch size to use for fitting.
        :type batch_size: int
        :param max_iter: The maximum number of iterations to use for fitting.
        :type max_iter: int
        :param local_execution: Whether to execute the fitting locally.
        :type local_execution: bool
        """
        original_index = self.index
        if local_execution:
            # Switch to a local index for fitting
            from semantic_router.index.local import LocalIndex

            remote_utterances = self.index.get_utterances(include_metadata=True)
            # TODO Enhance by retrieving directly the vectors instead of embedding all utterances again
            routes = []
            utterances = []
            metadata = []
            for utterance in remote_utterances:
                routes.append(utterance.route)
                utterances.append(utterance.utterance)
                metadata.append(utterance.metadata)
            embeddings = self.encoder(utterances)
            self.index = LocalIndex()
            self.index.add(
                embeddings=embeddings,
                routes=routes,
                utterances=utterances,
                metadata_list=metadata,
            )

        # convert inputs into array
        Xq: List[List[float]] = []
        for i in tqdm(range(0, len(X), batch_size), desc="Generating embeddings"):
            emb = np.array(self.encoder(X[i : i + batch_size]))
            Xq.extend(emb)
        # initial eval (we will iterate from here)
        best_acc = self._vec_evaluate(Xq_d=np.array(Xq), y=y)
        best_thresholds = self.get_thresholds()
        # begin fit
        for _ in (pbar := tqdm(range(max_iter), desc="Training")):
            pbar.set_postfix({"acc": round(best_acc, 2)})
            # Find the best score threshold for each route
            thresholds = threshold_random_search(
                route_layer=self,
                search_range=0.8,
            )
            # update current route layer
            self._update_thresholds(route_thresholds=thresholds)
            # evaluate
            acc = self._vec_evaluate(Xq_d=Xq, y=y)
            # update best
            if acc > best_acc:
                best_acc = acc
                best_thresholds = thresholds
        # update route layer to best thresholds
        self._update_thresholds(route_thresholds=best_thresholds)

        if local_execution:
            # Switch back to the original index
            self.index = original_index

    def evaluate(self, X: List[str], y: List[str], batch_size: int = 500) -> float:
        """Evaluate the accuracy of the route selection.

        :param X: The input data.
        :type X: List[str]
        :param y: The output data.
        :type y: List[str]
        :param batch_size: The batch size to use for evaluation.
        :type batch_size: int
        :return: The accuracy of the route selection.
        :rtype: float
        """
        Xq: List[List[float]] = []
        for i in tqdm(range(0, len(X), batch_size), desc="Generating embeddings"):
            emb = np.array(self.encoder(X[i : i + batch_size]))
            Xq.extend(emb)

        accuracy = self._vec_evaluate(Xq_d=np.array(Xq), y=y)
        return accuracy

    def _vec_evaluate(
        self, Xq_d: Union[List[float], Any], y: List[str], **kwargs
    ) -> float:
        """Evaluate the accuracy of the route selection.

        :param Xq_d: The input data.
        :type Xq_d: Union[List[float], Any]
        :param y: The output data.
        :type y: List[str]
        :return: The accuracy of the route selection.
        :rtype: float
        """
        correct = 0
        for xq, target_route in zip(Xq_d, y):
            # We treate dynamic routes as static here, because when evaluating we use only vectors, and dynamic routes expect strings by default.
            route_choice = self(vector=xq, simulate_static=True)
            if isinstance(route_choice, list):
                route_name = route_choice[0].name
            else:
                route_name = route_choice.name
            if route_name == target_route:
                correct += 1
        accuracy = correct / len(Xq_d)
        return accuracy

    def _get_route_names(self) -> List[str]:
        """Get the names of the routes.

        :return: The names of the routes.
        :rtype: List[str]
        """
        return [route.name for route in self.routes]

    @deprecated("Use `__call__` or `acall` with `limit=None` instead.")
    def _semantic_classify_multiple_routes(
        self, query_results: list[dict]
    ) -> list[dict]:
        """Classify the query results into a list of routes.

        :param query_results: The query results to classify.
        :type query_results: List[Dict]
        :return: Most similar results with scores.
        :rtype list[dict]:
        """
        raise NotImplementedError(
            "This method has been deprecated. Use `__call__` or `acall` with "
            "`limit=None` instead."
        )


def threshold_random_search(
    route_layer: BaseRouter,
    search_range: Union[int, float],
) -> Dict[str, float]:
    """Performs a random search iteration given a route layer and a search range.

    :param route_layer: The route layer to search.
    :type route_layer: BaseRouter
    :param search_range: The search range to use.
    :type search_range: Union[int, float]
    :return: A dictionary of route names and their associated thresholds.
    :rtype: Dict[str, float]
    """
    # extract the route names
    routes = route_layer.get_thresholds()
    route_names = list(routes.keys())
    route_thresholds = list(routes.values())
    # generate search range for each
    score_threshold_values = []
    for threshold in route_thresholds:
        score_threshold_values.append(
            np.linspace(
                start=max(threshold - search_range, 0.0),
                stop=min(threshold + search_range, 1.0),
                num=100,
            )
        )
    # Generate a random threshold for each route
    score_thresholds = {
        route: random.choice(score_threshold_values[i])
        for i, route in enumerate(route_names)
    }
    return score_thresholds
