import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml
from tqdm.auto import tqdm

from semantic_router.encoders import BaseEncoder, OpenAIEncoder
from semantic_router.linear import similarity_matrix, top_scores
from semantic_router.llms import BaseLLM, OpenAILLM
from semantic_router.route import Route
from semantic_router.schema import Encoder, EncoderType, RouteChoice
from semantic_router.utils.logger import logger


def is_valid(layer_config: str) -> bool:
    """Make sure the given string is json format and contains the 3 keys: ["encoder_name", "encoder_type", "routes"]"""
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
            # if encoder_name is not provided, use the default encoder for type
            # TODO base these values on default values in encoders themselves..
            # TODO without initializing them (as this is just config)
            if encoder_type == EncoderType.OPENAI:
                encoder_name = "text-embedding-ada-002"
            elif encoder_type == EncoderType.COHERE:
                encoder_name = "embed-english-v3.0"
            elif encoder_type == EncoderType.FASTEMBED:
                encoder_name = "BAAI/bge-small-en-v1.5"
            elif encoder_type == EncoderType.HUGGINGFACE:
                raise NotImplementedError
            logger.info(f"Using default {encoder_type} encoder: {encoder_name}")
        self.encoder_name = encoder_name
        self.routes = routes

    @classmethod
    def from_file(cls, path: str) -> "LayerConfig":
        """Load the routes from a file in JSON or YAML format"""
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

            route_config_str = json.dumps(layer)
            if is_valid(route_config_str):
                encoder_type = layer["encoder_type"]
                encoder_name = layer["encoder_name"]
                routes = [Route.from_dict(route) for route in layer["routes"]]
                return cls(
                    encoder_type=encoder_type, encoder_name=encoder_name, routes=routes
                )
            else:
                raise Exception("Invalid config JSON or YAML")

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
    index: Optional[np.ndarray] = None
    categories: Optional[np.ndarray] = None
    score_threshold: float
    encoder: BaseEncoder

    def __init__(
        self,
        encoder: Optional[BaseEncoder] = None,
        llm: Optional[BaseLLM] = None,
        routes: Optional[List[Route]] = None,
    ):
        logger.info("Initializing RouteLayer")
        self.index = None
        self.categories = None
        if encoder is None:
            logger.warning(
                "No encoder provided. Using default OpenAIEncoder. Ensure "
                "that you have set OPENAI_API_KEY in your environment."
            )
            self.encoder = OpenAIEncoder()
        else:
            self.encoder = encoder
        self.llm = llm
        self.routes: list[Route] = routes if routes is not None else []
        self.score_threshold = self.encoder.score_threshold
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
                f"No route found with name {top_class}. Check to see if any Routes have been defined."
            )
            return None
        return matching_routes[0]

    def __call__(
        self,
        text: Optional[str] = None,
        vector: Optional[List[float]] = None,
    ) -> RouteChoice:
        # if no vector provided, encode text to get vector
        if vector is None:
            if text is None:
                raise ValueError("Either text or vector must be provided")
            vector_arr = self._encode(text=text)
        else:
            vector_arr = np.array(vector)
        # get relevant utterances
        results = self._retrieve(xq=vector_arr)
        # decide most relevant routes
        top_class, top_class_scores = self._semantic_classify(results)
        # TODO do we need this check?
        route = self.check_for_matching_routes(top_class)
        if route is None:
            return RouteChoice()
        threshold = (
            route.score_threshold
            if route.score_threshold is not None
            else self.score_threshold
        )
        passed = self._pass_threshold(top_class_scores, threshold)
        if passed:
            if route.function_schema and text is None:
                raise ValueError(
                    "Route has a function schema, but no text was provided."
                )
            if route.function_schema and not isinstance(route.llm, BaseLLM):
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
        else:
            # if no route passes threshold, return empty route choice
            return RouteChoice()

    def __str__(self):
        return (
            f"RouteLayer(encoder={self.encoder}, "
            f"score_threshold={self.score_threshold}, "
            f"routes={self.routes})"
        )

    @classmethod
    def from_json(cls, file_path: str):
        config = LayerConfig.from_file(file_path)
        encoder = Encoder(type=config.encoder_type, name=config.encoder_name).model
        return cls(encoder=encoder, routes=config.routes)

    @classmethod
    def from_yaml(cls, file_path: str):
        config = LayerConfig.from_file(file_path)
        encoder = Encoder(type=config.encoder_type, name=config.encoder_name).model
        return cls(encoder=encoder, routes=config.routes)

    @classmethod
    def from_config(cls, config: LayerConfig):
        encoder = Encoder(type=config.encoder_type, name=config.encoder_name).model
        return cls(encoder=encoder, routes=config.routes)

    def add(self, route: Route):
        logger.info(f"Adding `{route.name}` route")
        # create embeddings
        embeds = self.encoder(route.utterances)
        # if route has no score_threshold, use default
        if route.score_threshold is None:
            route.score_threshold = self.score_threshold

        # create route array
        if self.categories is None:
            self.categories = np.array([route.name] * len(embeds))
        else:
            str_arr = np.array([route.name] * len(embeds))
            self.categories = np.concatenate([self.categories, str_arr])
        # create utterance array (the index)
        if self.index is None:
            self.index = np.array(embeds)
        else:
            embed_arr = np.array(embeds)
            self.index = np.concatenate([self.index, embed_arr])
        # add route to routes list
        self.routes.append(route)

    def _add_routes(self, routes: List[Route]):
        # create embeddings for all routes
        all_utterances = [
            utterance for route in routes for utterance in route.utterances
        ]
        embedded_utterance = self.encoder(all_utterances)

        # create route array
        route_names = [route.name for route in routes for _ in route.utterances]
        route_array = np.array(route_names)
        self.categories = (
            np.concatenate([self.categories, route_array])
            if self.categories is not None
            else route_array
        )

        # create utterance array (the index)
        embed_utterance_arr = np.array(embedded_utterance)
        self.index = (
            np.concatenate([self.index, embed_utterance_arr])
            if self.index is not None
            else embed_utterance_arr
        )

    def _encode(self, text: str) -> Any:
        """Given some text, encode it."""
        # create query vector
        xq = np.array(self.encoder([text]))
        xq = np.squeeze(xq)  # Reduce to 1d array.
        return xq

    def _retrieve(self, xq: Any, top_k: int = 5) -> List[dict]:
        """Given a query vector, retrieve the top_k most similar records."""
        if self.index is not None:
            # calculate similarity matrix
            sim = similarity_matrix(xq, self.index)
            scores, idx = top_scores(sim, top_k)
            # get the utterance categories (route names)
            routes = self.categories[idx] if self.categories is not None else []
            return [{"route": d, "score": s.item()} for d, s in zip(routes, scores)]
        else:
            logger.warning("No index found for route layer.")
            return []

    def _semantic_classify(self, query_results: List[dict]) -> Tuple[str, List[float]]:
        scores_by_class: Dict[str, List[float]] = {}
        for result in query_results:
            score = result["score"]
            route = result["route"]
            if route in scores_by_class:
                scores_by_class[route].append(score)
            else:
                scores_by_class[route] = [score]

        # Calculate total score for each class
        total_scores = {route: sum(scores) for route, scores in scores_by_class.items()}
        top_class = max(total_scores, key=lambda x: total_scores[x], default=None)

        # Return the top class and its associated scores
        if top_class is not None:
            return str(top_class), scores_by_class.get(top_class, [])
        else:
            logger.warning("No classification found for semantic classifier.")
            return "", []

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
        max_iter: int = 500,
    ):
        # convert inputs into array
        Xq: Any = np.array(self.encoder(X))
        # initial eval (we will iterate from here)
        best_acc = self._vec_evaluate(Xq=Xq, y=y)
        best_thresholds = self.get_thresholds()
        # begin fit
        for _ in (pbar := tqdm(range(max_iter))):
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

    def evaluate(self, X: List[str], y: List[str]) -> float:
        """
        Evaluate the accuracy of the route selection.
        """
        Xq = np.array(self.encoder(X))
        accuracy = self._vec_evaluate(Xq=Xq, y=y)
        return accuracy

    def _vec_evaluate(self, Xq: Union[List[float], Any], y: List[str]) -> float:
        """
        Evaluate the accuracy of the route selection.
        """
        correct = 0
        for xq, target_route in zip(Xq, y):
            route_choice = self(vector=xq)
            if route_choice.name == target_route:
                correct += 1
        accuracy = correct / len(Xq)
        return accuracy


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
