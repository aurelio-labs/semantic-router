"""
This module provides the HFEndpointEncoder class to embeddings models using Huggingface's endpoint.

The HFEndpointEncoder class is a subclass of DenseEncoder and utilizes a specified Huggingface
endpoint to generate embeddings for given documents. It requires the URL of the Huggingface
API endpoint and an API key for authentication. The class supports customization of the score
threshold for filtering or processing the embeddings.

Example usage:

    from semantic_router.encoders.hfendpointencoder import HFEndpointEncoder

    encoder = HFEndpointEncoder(
        huggingface_url="https://api-inference.huggingface.co/models/BAAI/bge-large-en-v1.5",
        huggingface_api_key="your-hugging-face-api-key"
    )
    embeddings = encoder(["document1", "document2"])

Classes:
    HFEndpointEncoder: A class for generating embeddings using a Huggingface endpoint.
"""

import os
import time
from typing import Any, Dict, List, Optional

import requests
from pydantic import PrivateAttr

from semantic_router.encoders import DenseEncoder
from semantic_router.utils.logger import logger

# TODO: this should support local models, and we should have another class for remote
# inference endpoint models


class HuggingFaceEncoder(DenseEncoder):
    """HuggingFace encoder class for local embedding models. Models can be trained and
    loaded from private repositories, or from the Huggingface Hub. The class supports
    customization of the score threshold for filtering or processing the embeddings.

    Example usage:

    ```python
    from semantic_router.encoders import HuggingFaceEncoder

    encoder = HuggingFaceEncoder(
        name="sentence-transformers/all-MiniLM-L6-v2",
        device="cuda"
    )
    embeddings = encoder(["document1", "document2"])
    ```
    """

    name: str = "sentence-transformers/all-MiniLM-L6-v2"
    type: str = "huggingface"
    tokenizer_kwargs: Dict = {}
    model_kwargs: Dict = {}
    device: Optional[str] = None
    _tokenizer: Any = PrivateAttr()
    _model: Any = PrivateAttr()
    _torch: Any = PrivateAttr()

    def __init__(self, **data):
        if data.get("score_threshold") is None:
            data["score_threshold"] = 0.5
        super().__init__(**data)
        self._tokenizer, self._model = self._initialize_hf_model()

    def _initialize_hf_model(self):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except (ImportError, RuntimeError, ModuleNotFoundError):
            raise ImportError(
                "Please install transformers to use HuggingFaceEncoder. "
                "You can install it with: "
                "`pip install semantic-router[local]`"
            )

        self._torch = torch

        tokenizer = AutoTokenizer.from_pretrained(
            self.name,
            **self.tokenizer_kwargs,
        )

        model = AutoModel.from_pretrained(self.name, **self.model_kwargs)

        if self.device:
            model.to(self.device)

        else:
            device = "cuda" if self._torch.cuda.is_available() else "cpu"
            model.to(device)
            self.device = device

        return tokenizer, model

    def __call__(
        self,
        docs: List[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        pooling_strategy: str = "mean",
    ) -> List[List[float]]:
        """Encode a list of documents into embeddings using the local Hugging Face model.

        :param docs: A list of documents to encode.
        :type docs: List[str]
        :param batch_size: The batch size for encoding.
        """
        all_embeddings = []
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i : i + batch_size]

            encoded_input = self._tokenizer(
                batch_docs, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)

            with self._torch.no_grad():
                model_output = self._model(**encoded_input)

            if pooling_strategy == "mean":
                embeddings = self._mean_pooling(
                    model_output, encoded_input["attention_mask"]
                )
            elif pooling_strategy == "max":
                embeddings = self._max_pooling(
                    model_output, encoded_input["attention_mask"]
                )
            else:
                raise ValueError(
                    "Invalid pooling_strategy. Please use 'mean' or 'max'."
                )

            if normalize_embeddings:
                embeddings = self._torch.nn.functional.normalize(embeddings, p=2, dim=1)

            embeddings = embeddings.tolist()
            all_embeddings.extend(embeddings)
        return all_embeddings

    def _mean_pooling(self, model_output, attention_mask):
        """Perform mean pooling on the token embeddings.

        :param model_output: The output of the model.
        :type model_output: torch.Tensor
        :param attention_mask: The attention mask.
        """
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return self._torch.sum(
            token_embeddings * input_mask_expanded, 1
        ) / self._torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _max_pooling(self, model_output, attention_mask):
        """Perform max pooling on the token embeddings.

        :param model_output: The output of the model.
        :type model_output: torch.Tensor
        :param attention_mask: The attention mask.
        """
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        token_embeddings[input_mask_expanded == 0] = -1e9
        return self._torch.max(token_embeddings, 1)[0]


class HFEndpointEncoder(DenseEncoder):
    """HFEndpointEncoder class to embeddings models using Huggingface's inference endpoints.

    The HFEndpointEncoder class is a subclass of DenseEncoder and utilizes a specified
    Huggingface endpoint to generate embeddings for given documents. It requires the URL
    of the Huggingface API endpoint and an API key for authentication. The class supports
    customization of the score threshold for filtering or processing the embeddings.

    Example usage:

    ```python
    from semantic_router.encoders import HFEndpointEncoder

    encoder = HFEndpointEncoder(
        huggingface_url="https://api-inference.huggingface.co/models/BAAI/bge-large-en-v1.5",
        huggingface_api_key="your-hugging-face-api-key"
    )
    embeddings = encoder(["document1", "document2"])
    ```
    """

    name: str = "hugging_face_custom_endpoint"
    huggingface_url: Optional[str] = None
    huggingface_api_key: Optional[str] = None

    def __init__(
        self,
        name: Optional[str] = "hugging_face_custom_endpoint",
        huggingface_url: Optional[str] = None,
        huggingface_api_key: Optional[str] = None,
        score_threshold: float = 0.8,
    ):
        """Initializes the HFEndpointEncoder with the specified parameters.

        :param name: The name of the encoder.
        :type name: str
        :param huggingface_url: The URL of the Hugging Face API endpoint.
        :type huggingface_url: str
        :param huggingface_api_key: The API key for the Hugging Face API.
        :type huggingface_api_key: str
        :param score_threshold: A threshold for processing the embeddings.
        :type score_threshold: float
        :raise ValueError: If either `huggingface_url` or `huggingface_api_key` is None.
        """
        huggingface_url = huggingface_url or os.getenv("HF_API_URL")
        huggingface_api_key = huggingface_api_key or os.getenv("HF_API_KEY")
        if score_threshold is None:
            score_threshold = 0.8

        super().__init__(name=name, score_threshold=score_threshold)  # type: ignore

        if huggingface_url is None:
            raise ValueError("HuggingFace endpoint url cannot be 'None'.")
        if huggingface_api_key is None:
            raise ValueError("HuggingFace API key cannot be 'None'.")

        self.huggingface_url = huggingface_url or os.getenv("HF_API_URL")
        self.huggingface_api_key = huggingface_api_key or os.getenv("HF_API_KEY")

        try:
            self.query({"inputs": "Hello World!", "parameters": {}})
        except Exception as e:
            raise ValueError(
                f"HuggingFace endpoint client failed to initialize. Error: {e}"
            ) from e

    def __call__(self, docs: List[str]) -> List[List[float]]:
        """Encodes a list of documents into embeddings using the Hugging Face API.

        :param docs: A list of documents to encode.
        :type docs: List[str]
        :return: A list of embeddings for the given documents.
        :rtype: List[List[float]]
        :raise ValueError: If no embeddings are returned for a document.
        """
        embeddings = []
        for d in docs:
            try:
                output = self.query({"inputs": d, "parameters": {}})
                if not output or len(output) == 0:
                    raise ValueError("No embeddings returned from the query.")
                embeddings.append(output)

            except Exception as e:
                raise ValueError(
                    f"No embeddings returned for document. Error: {e}"
                ) from e
        return embeddings

    def query(self, payload, max_retries=3, retry_interval=5):
        """Sends a query to the Hugging Face API and returns the response.

        :param payload: The payload to send in the request.
        :type payload: dict
        :return: The response from the Hugging Face API.
        :rtype: dict
        :raise ValueError: If the query fails or the response status is not 200.
        """
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.huggingface_api_key}",
            "Content-Type": "application/json",
        }
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(
                    self.huggingface_url,
                    headers=headers,
                    json=payload,
                    # timeout=timeout_seconds,
                )
                if response.status_code == 503:
                    estimated_time = response.json().get("estimated_time", "")
                    if estimated_time:
                        logger.info(
                            f"Model Initializing wait for - {estimated_time:.2f}s "
                        )
                        time.sleep(estimated_time)
                        continue
                else:
                    response.raise_for_status()

            except requests.exceptions.RequestException:
                if attempt < max_retries - 1:
                    logger.info(f"Retrying attempt: {attempt} for payload: {payload} ")
                    time.sleep(retry_interval)
                    retry_interval += attempt
                else:
                    raise ValueError(
                        f"Query failed with status {response.status_code}: {response.text}"
                    )

        return response.json()
