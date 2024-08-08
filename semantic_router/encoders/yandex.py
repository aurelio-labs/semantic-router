"""
This module provides the YandexGPTEncoder class for generating embeddings using YandexGPT.

The YandexGPTEncoder class is a subclass of BaseEncoder and utilizes the Embeddings class from the
YandexGPT to generate embeddings for given documents. It requires a YandexGPT API key and a model URI.

Example usage:

    from semantic_router.encoders import YandexGPTEncoder

    encoder = YandexGPTEncoder(api_key="your-api-key", catalog_id="your-catalog-id")
    embeddings = encoder(["document1", "document2"])

Classes:
    YandexGPTEncoder: A class for generating embeddings using YandexGPT.
"""

import os
from time import sleep
from typing import Any, List, Optional

import requests
from semantic_router.encoders import BaseEncoder
from semantic_router.utils.defaults import EncoderDefault


class YandexGPTEncoder(BaseEncoder):
    """YandexGPTEncoder class for generating embeddings using YandexGPT.

        Attributes:
            client: An instance of the TextEmbeddingModel client.
            type: The type of the encoder, which is "yandexgpt".
        """
    client: Optional[Any] = None
    type: str = "yandexgpt"

    def __init__(
        self,
        name: Optional[str] = None,
        api_key: Optional[str] = None,
        catalog_id: Optional[str] = None,
        score_threshold: float = 0.75):
        """Initializes the YandexGPTEncoder.

            Args:
                name: The name of the pre-trained model to use for embedding.
                    If not provided, the default model specified in EncoderDefault will
                    be used.
                api_key: The YandexGPT API key.
                    If not provided, it will be retrieved from the YANDEX_GPT_KEY
                    environment variable.
                catalog_id: The catalog ID used to retrieve the model from.

            Raises:
                ValueError: If the YandexGPT API key or model URI is not provided.
                """
        if name is None:
            name = EncoderDefault.YANDEX.value["embedding_model"]

        super().__init__(name=name, score_threshold=score_threshold)

        self.client = self._initialize_client(api_key, catalog_id)

    def _initialize_client(self, api_key, catalog_id):
            """Initializes the YandexGPT client.

            Args:
                api_key: The YandexGPT API key.
                catalog_id: The URI of the YandexGPT model.

            Returns:
                An instance of the Embeddings client.

            Raises:
                ImportError: If the required YandexGPT library is not installed.
                ValueError: If the YandexGPT API key or model URI is not provided.
            """

            api_key = api_key or os.getenv("YANDEX_GPT_KEY")
            catalog_id = catalog_id or os.getenv("YANDEX_CATALOG_ID")
            if api_key is None:
                raise ValueError("YandexGPT API key cannot be 'None'.")
            if catalog_id is None:
                raise ValueError("YandexGPT catalog ID cannot be 'None'.")
            try:
                return {"api_key": api_key, "model_Uri": f"emb://{catalog_id}/text-search-doc/latest"}
            except Exception as e:
                raise ValueError(
                    f"Yandex API client failed to initialize. Error: {e}"
                ) from e

    def _get_headers(self):
        """Returns the headers for the YandexGPT API request."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {self.client['api_key']}",
            "x-data-logging-enabled": "false"
        }

    def __call__(self, docs: List[str]) -> List[List[float]]:
        """Generates embeddings for the given documents.

        Args:
            docs: A list of strings representing the documents to embed.

        Returns:
            A list of lists, where each inner list contains the embedding values for a
            document.

        Raises:
            ValueError: If the YandexGPT client is not initialized or if the
            API call fails.
        """
        if self.client is None:
            raise ValueError("YandexGPT client is not initialized.")

        url = "https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding"
        embeddings = []
        for doc in docs:
            data = {
                "modelUri": self.client["model_Uri"],
                "text": doc
            }

            try:
                sleep(0.2) # Ensure compliance with rate limits
                response = requests.post(url, json=data, headers=self._get_headers())
                if response.status_code == 200:
                    embeddings.append(response.json()["embedding"])
                else:
                    raise ValueError(f"Failed to get embedding for document: {doc}")
            except Exception as e:
                raise ValueError(f"YandexGPT API call failed. Error: {e}") from e

        return embeddings


