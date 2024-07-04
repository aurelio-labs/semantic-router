import importlib
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml  # type: ignore
from tqdm.auto import tqdm

from semantic_router.encoders import AutoEncoder, BaseEncoder, OpenAIEncoder
from semantic_router.index.base import BaseIndex
from semantic_router.index.local import LocalIndex
from semantic_router.llms import BaseLLM, OpenAILLM
from semantic_router.route import Route
from semantic_router.schema import EncoderType, RouteChoice
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


class LayerConfig:
    """
    Generates a LayerConfig object that can be used for initializing a
    RouteLayer.
    """

    routes: List[Route] = []

    def __init__(
        self,
        routes: List[Route] = [],
        encoder_type: str = "openai",
        encoder_name: Optional[str] = None,
    ):
        self.encoder_type = encoder_type
        if encoder_name is None:
            for encode_type in EncoderType:
                if encode_type.value == self.encoder_type:
                    if self.encoder_type == EncoderType.HUGGINGFACE.value:
                        raise NotImplementedError(
                            "HuggingFace encoder not supported by LayerConfig yet."
                        )
                    encoder_name = EncoderDefault[encode_type.name].value[
                        "embedding_model"
                    ]
                    break
            logger.info(f"Using default {encoder_type} encoder: {encoder_name}")
        self.encoder_name = encoder_name
        self.routes = routes

    @classmethod
    def from_file(cls, path: str) -> "LayerConfig":
        logger.info(f"Loading route config from {path}")
        _, ext = os.path.splitext(path)
        with open(path, "r") as f:
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "encoder_type": self.encoder_type,
            "encoder_name": self.encoder_name,
            "routes": [route.to_dict() for route in self.routes],
        }

    def to_file(self, path: str):
        """Save the routes to a file in JSON or YAML format"""
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

    def add(self, route: Route):
        self.routes.append(route)
        logger.info(f"Added route `{route.name}`")

    def get(self, name: str) -> Optional[Route]:
        for route in self.routes:
            if route.name == name:
                return route
        logger.error(f"Route `{name}` not found")
        return None

    def remove(self, name: str):
        if name not in [route.name for route in self.routes]:
            logger.error(f"Route `{name}` not found")
        else:
            self.routes = [route for route in self.routes if route.name != name]
            logger.info(f"Removed route `{name}`")


class RouteLayer:
    score_threshold: float
    encoder: BaseEncoder
    index: BaseIndex

    def __init__(
        self,
        encoder: Optional[BaseEncoder] = None,
        llm: Optional[BaseLLM] = None,
        routes: Optional[List[Route]] = None,
        index: Optional[BaseIndex] = None,  # type: ignore
        top_k: int = 5,
        aggregation: str = "sum",
    ):
        self.index: BaseIndex = index if index is not None else LocalIndex()
        if encoder is None:
            logger.warning(
                "No encoder provided. Using default OpenAIEncoder. Ensure "
                "that you have set OPENAI_API_KEY in your environment."
            )
            self.encoder = OpenAIEncoder()
        else:
            self.encoder = encoder
        self.llm = llm
        self.routes: List[Route] = routes if routes is not None else []
        if self.encoder.score_threshold is None:
            raise ValueError(
                "No score threshold provided for encoder. Please set the score threshold "
                "in the encoder config."
            )
        self.score_threshold = self.encoder.score_threshold
        self.top_k = top_k
        if self.top_k < 1:
            raise ValueError(f"top_k needs to be >= 1, but was: {self.top_k}.")
        self.aggregation = aggregation
        if self.aggregation not in ["sum", "mean", "max"]:
            raise ValueError(
                f"Unsupported aggregation method chosen: {aggregation}. Choose either 'SUM', 'MEAN', or 'MAX'."
            )
        self.aggregation_method = self._set_aggregation_method(self.aggregation)

        # set route score thresholds if not already set
        for route in self.routes:
            if route.score_threshold is None:
                route.score_threshold = self.score_threshold
        # if routes list has been passed, we initialize index now
        if len(self.routes) > 0:
            # initialize index now
            self._add_routes(routes=self.routes)

    def check_for_matching_routes(self, top_class: str) -> Optional[Route]:
        matching_routes = [route for route in self.routes if route.name == top_class]
        if not matching_routes:
            logger.error(
                f"No route found with name {top_class}. Check to see if any Routes "
                "have been defined."
            )
            return None
        return matching_routes[0]

    def __call__(
        self,
        text: Optional[str] = None,
        vector: Optional[List[float]] = None,
        simulate_static: bool = False,
        route_filter: Optional[List[str]] = None,
    ) -> RouteChoice:
        # if no vector provided, encode text to get vector
        if vector is None:
            if text is None:
                raise ValueError("Either text or vector must be provided")
            vector = self._encode(text=text)

        route, top_class_scores = self._retrieve_top_route(vector, route_filter)
        passed = self._check_threshold(top_class_scores, route)
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
            return route(text)
        elif passed and route is not None and simulate_static:
            return RouteChoice(
                name=route.name,
                function_call=None,
                similarity_score=None,
            )
        else:
            # if no route passes threshold, return empty route choice
            return RouteChoice()

    async def acall(
        self,
        text: Optional[str] = None,
        vector: Optional[List[float]] = None,
        simulate_static: bool = False,
        route_filter: Optional[List[str]] = None,
    ) -> RouteChoice:
        # if no vector provided, encode text to get vector
        if vector is None:
            if text is None:
                raise ValueError("Either text or vector must be provided")
            vector = await self._async_encode(text=text)

        route, top_class_scores = await self._async_retrieve_top_route(
            vector, route_filter
        )
        passed = self._check_threshold(top_class_scores, route)
        if passed and route is not None and not simulate_static:
            if route.function_schemas and text is None:
                raise ValueError(
                    "Route has a function schema, but no text was provided."
                )
            if route.function_schemas and not isinstance(route.llm, BaseLLM):
                raise NotImplementedError(
                    "Dynamic routes not yet supported for async calls."
                )
            return route(text)
        elif passed and route is not None and simulate_static:
            return RouteChoice(
                name=route.name,
                function_call=None,
                similarity_score=None,
            )
        else:
            # if no route passes threshold, return empty route choice
            return RouteChoice()

    def retrieve_multiple_routes(
        self,
        text: Optional[str] = None,
        vector: Optional[List[float]] = None,
    ) -> List[RouteChoice]:
        if vector is None:
            if text is None:
                raise ValueError("Either text or vector must be provided")
            vector_arr = self._encode(text=text)
        else:
            vector_arr = np.array(vector)
        # get relevant utterances
        results = self._retrieve(xq=vector_arr)

        # decide most relevant routes
        categories_with_scores = self._semantic_classify_multiple_routes(results)

        route_choices = []
        for category, score in categories_with_scores:
            route = self.check_for_matching_routes(category)
            if route:
                route_choice = RouteChoice(name=route.name, similarity_score=score)
                route_choices.append(route_choice)

        return route_choices

    def _retrieve_top_route(
        self, vector: List[float], route_filter: Optional[List[str]] = None
    ) -> Tuple[Optional[Route], List[float]]:
        """
        Retrieve the top matching route based on the given vector.
        Returns a tuple of the route (if any) and the scores of the top class.
        """
        # get relevant results (scores and routes)
        results = self._retrieve(
            xq=np.array(vector), top_k=self.top_k, route_filter=route_filter
        )
        # decide most relevant routes
        top_class, top_class_scores = self._semantic_classify(results)
        # TODO do we need this check?
        route = self.check_for_matching_routes(top_class)
        return route, top_class_scores

    async def _async_retrieve_top_route(
        self, vector: List[float], route_filter: Optional[List[str]] = None
    ) -> Tuple[Optional[Route], List[float]]:
        # get relevant results (scores and routes)
        results = await self._async_retrieve(
            xq=np.array(vector), top_k=self.top_k, route_filter=route_filter
        )
        # decide most relevant routes
        top_class, top_class_scores = await self._async_semantic_classify(results)
        # TODO do we need this check?
        route = self.check_for_matching_routes(top_class)
        return route, top_class_scores

    def _check_threshold(self, scores: List[float], route: Optional[Route]) -> bool:
        """
        Check if the route's score passes the specified threshold.
        """
        if route is None:
            return False
        threshold = (
            route.score_threshold
            if route.score_threshold is not None
            else self.score_threshold
        )
        return self._pass_threshold(scores, threshold)

    def __str__(self):
        return (
            f"RouteLayer(encoder={self.encoder}, "
            f"score_threshold={self.score_threshold}, "
            f"routes={self.routes})"
        )

    @classmethod
    def from_json(cls, file_path: str):
        config = LayerConfig.from_file(file_path)
        encoder = AutoEncoder(type=config.encoder_type, name=config.encoder_name).model
        return cls(encoder=encoder, routes=config.routes)

    @classmethod
    def from_yaml(cls, file_path: str):
        config = LayerConfig.from_file(file_path)
        encoder = AutoEncoder(type=config.encoder_type, name=config.encoder_name).model
        return cls(encoder=encoder, routes=config.routes)

    @classmethod
    def from_config(cls, config: LayerConfig, index: Optional[BaseIndex] = None):
        encoder = AutoEncoder(type=config.encoder_type, name=config.encoder_name).model
        return cls(encoder=encoder, routes=config.routes, index=index)

    def add(self, route: Route):
        logger.info(f"Adding `{route.name}` route")
        # create embeddings
        embeds = self.encoder(route.utterances)
        # if route has no score_threshold, use default
        if route.score_threshold is None:
            route.score_threshold = self.score_threshold

        # add routes to the index
        self.index.add(
            embeddings=embeds,
            routes=[route.name] * len(route.utterances),
            utterances=route.utterances,
        )
        self.routes.append(route)

    def list_route_names(self) -> List[str]:
        return [route.name for route in self.routes]

    def update(self, route_name: str, utterances: List[str]):
        raise NotImplementedError("This method has not yet been implemented.")

    def delete(self, route_name: str):
        """Deletes a route given a specific route name.

        :param route_name: the name of the route to be deleted
        :type str:
        """
        if route_name not in [route.name for route in self.routes]:
            err_msg = f"Route `{route_name}` not found in RouteLayer"
            logger.warning(err_msg)
            self.index.delete(route_name=route_name)
        else:
            self.routes = [route for route in self.routes if route.name != route_name]
            self.index.delete(route_name=route_name)

    def _refresh_routes(self):
        """Pulls out the latest routes from the index."""
        raise NotImplementedError("This method has not yet been implemented.")
        route_mapping = {route.name: route for route in self.routes}
        index_routes = self.index.get_routes()
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

    def _add_routes(self, routes: List[Route]):
        # create embeddings for all routes
        all_utterances = [
            utterance for route in routes for utterance in route.utterances
        ]
        embedded_utterances = self.encoder(all_utterances)
        # create route array
        route_names = [route.name for route in routes for _ in route.utterances]
        # add everything to the index
        self.index.add(
            embeddings=embedded_utterances,
            routes=route_names,
            utterances=all_utterances,
        )

    def _encode(self, text: str) -> Any:
        """Given some text, encode it."""
        # create query vector
        xq = np.array(self.encoder([text]))
        xq = np.squeeze(xq)  # Reduce to 1d array.
        return xq

    async def _async_encode(self, text: str) -> Any:
        """Given some text, encode it."""
        # create query vector
        xq = np.array(await self.encoder.acall(docs=[text]))
        xq = np.squeeze(xq)  # Reduce to 1d array.
        return xq

    def _retrieve(
        self, xq: Any, top_k: int = 5, route_filter: Optional[List[str]] = None
    ) -> List[Dict]:
        """Given a query vector, retrieve the top_k most similar records."""
        # get scores and routes
        scores, routes = self.index.query(
            vector=xq, top_k=top_k, route_filter=route_filter
        )
        return [{"route": d, "score": s.item()} for d, s in zip(routes, scores)]

    async def _async_retrieve(
        self, xq: Any, top_k: int = 5, route_filter: Optional[List[str]] = None
    ) -> List[Dict]:
        """Given a query vector, retrieve the top_k most similar records."""
        # get scores and routes
        scores, routes = await self.index.aquery(
            vector=xq, top_k=top_k, route_filter=route_filter
        )
        return [{"route": d, "score": s.item()} for d, s in zip(routes, scores)]

    def _set_aggregation_method(self, aggregation: str = "sum"):
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

    def _semantic_classify(self, query_results: List[Dict]) -> Tuple[str, List[float]]:
        scores_by_class = self.group_scores_by_class(query_results)

        # Calculate total score for each class
        total_scores = {
            route: self.aggregation_method(scores)
            for route, scores in scores_by_class.items()
        }
        top_class = max(total_scores, key=lambda x: total_scores[x], default=None)

        # Return the top class and its associated scores
        if top_class is not None:
            return str(top_class), scores_by_class.get(top_class, [])
        else:
            logger.warning("No classification found for semantic classifier.")
            return "", []

    async def _async_semantic_classify(
        self, query_results: List[Dict]
    ) -> Tuple[str, List[float]]:
        scores_by_class = await self.async_group_scores_by_class(query_results)

        # Calculate total score for each class
        total_scores = {
            route: self.aggregation_method(scores)
            for route, scores in scores_by_class.items()
        }
        top_class = max(total_scores, key=lambda x: total_scores[x], default=None)

        # Return the top class and its associated scores
        if top_class is not None:
            return str(top_class), scores_by_class.get(top_class, [])
        else:
            logger.warning("No classification found for semantic classifier.")
            return "", []

    def get(self, name: str) -> Optional[Route]:
        for route in self.routes:
            if route.name == name:
                return route
        logger.error(f"Route `{name}` not found")
        return None

    def _semantic_classify_multiple_routes(
        self, query_results: List[Dict]
    ) -> List[Tuple[str, float]]:
        scores_by_class = self.group_scores_by_class(query_results)

        # Filter classes based on threshold and find max score for each
        classes_above_threshold = []
        for route_name, scores in scores_by_class.items():
            # Use the get method to find the Route object by its name
            route_obj = self.get(route_name)
            if route_obj is not None:
                # Use the Route object's threshold if it exists, otherwise use the provided threshold
                _threshold = (
                    route_obj.score_threshold
                    if route_obj.score_threshold is not None
                    else self.score_threshold
                )
                if self._pass_threshold(scores, _threshold):
                    max_score = max(scores)
                    classes_above_threshold.append((route_name, max_score))

        return classes_above_threshold

    def group_scores_by_class(
        self, query_results: List[Dict]
    ) -> Dict[str, List[float]]:
        scores_by_class: Dict[str, List[float]] = {}
        for result in query_results:
            score = result["score"]
            route = result["route"]
            if route in scores_by_class:
                scores_by_class[route].append(score)
            else:
                scores_by_class[route] = [score]
        return scores_by_class

    async def async_group_scores_by_class(
        self, query_results: List[Dict]
    ) -> Dict[str, List[float]]:
        scores_by_class: Dict[str, List[float]] = {}
        for result in query_results:
            score = result["score"]
            route = result["route"]
            if route in scores_by_class:
                scores_by_class[route].append(score)
            else:
                scores_by_class[route] = [score]
        return scores_by_class

    def _pass_threshold(self, scores: List[float], threshold: float) -> bool:
        if scores:
            return max(scores) > threshold
        else:
            return False

    def _update_thresholds(self, score_thresholds: Optional[Dict[str, float]] = None):
        """
        Update the score thresholds for each route.
        """
        if score_thresholds:
            for route in self.routes:
                route.score_threshold = score_thresholds.get(
                    route.name, self.score_threshold
                )

    def to_config(self) -> LayerConfig:
        return LayerConfig(
            encoder_type=self.encoder.type,
            encoder_name=self.encoder.name,
            routes=self.routes,
        )

    def to_json(self, file_path: str):
        config = self.to_config()
        config.to_file(file_path)

    def to_yaml(self, file_path: str):
        config = self.to_config()
        config.to_file(file_path)

    def get_thresholds(self) -> Dict[str, float]:
        # TODO: float() below is hacky fix for lint, fix this with new type?
        thresholds = {
            route.name: float(route.score_threshold or self.score_threshold)
            for route in self.routes
        }
        return thresholds

    def fit(
        self,
        X: List[str],
        y: List[str],
        batch_size: int = 500,
        max_iter: int = 500,
    ):
        # convert inputs into array
        Xq: List[List[float]] = []
        for i in tqdm(range(0, len(X), batch_size), desc="Generating embeddings"):
            emb = np.array(self.encoder(X[i : i + batch_size]))
            Xq.extend(emb)
        # initial eval (we will iterate from here)
        best_acc = self._vec_evaluate(Xq=np.array(Xq), y=y)
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
            self._update_thresholds(score_thresholds=thresholds)
            # evaluate
            acc = self._vec_evaluate(Xq=Xq, y=y)
            # update best
            if acc > best_acc:
                best_acc = acc
                best_thresholds = thresholds
        # update route layer to best thresholds
        self._update_thresholds(score_thresholds=best_thresholds)

    def evaluate(self, X: List[str], y: List[str], batch_size: int = 500) -> float:
        """
        Evaluate the accuracy of the route selection.
        """
        Xq: List[List[float]] = []
        for i in tqdm(range(0, len(X), batch_size), desc="Generating embeddings"):
            emb = np.array(self.encoder(X[i : i + batch_size]))
            Xq.extend(emb)

        accuracy = self._vec_evaluate(Xq=np.array(Xq), y=y)
        return accuracy

    def _vec_evaluate(self, Xq: Union[List[float], Any], y: List[str]) -> float:
        """
        Evaluate the accuracy of the route selection.
        """
        correct = 0
        for xq, target_route in zip(Xq, y):
            # We treate dynamic routes as static here, because when evaluating we use only vectors, and dynamic routes expect strings by default.
            route_choice = self(vector=xq, simulate_static=True)
            if route_choice.name == target_route:
                correct += 1
        accuracy = correct / len(Xq)
        return accuracy

    def _get_route_names(self) -> List[str]:
        return [route.name for route in self.routes]


def threshold_random_search(
    route_layer: RouteLayer,
    search_range: Union[int, float],
) -> Dict[str, float]:
    """Performs a random search iteration given a route layer and a search range."""
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
