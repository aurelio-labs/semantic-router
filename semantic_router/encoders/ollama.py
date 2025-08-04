import os
from typing import Any, List, Optional

from semantic_router.encoders import DenseEncoder
from semantic_router.utils.defaults import EncoderDefault


class OllamaEncoder(DenseEncoder):
    """OllamaEncoder class for generating embeddings using OLLAMA.

    https://ollama.com/search?c=embedding

    Example usage:

    ```python
    from semantic_router.encoders.ollama import OllamaEncoder

    encoder = OllamaEncoder(base_url="http://localhost:11434")
    embeddings = encoder(["document1", "document2"])
    ```

    Attributes:
        client: An instance of the TextEmbeddingModel client.
        type: The type of the encoder, which is "ollama".
    """

    client: Optional[Any] = None
    type: str = "ollama"

    def __init__(
        self,
        name: Optional[str] = None,
        score_threshold: float = 0.5,
        base_url: str | None = None,
    ):
        """Initializes the OllamaEncoder.

        :param model_name: The name of the pre-trained model to use for embedding.
            If not provided, the default model specified in EncoderDefault will
            be used.
        :type model_name: str
        :param score_threshold: The threshold for similarity scores.
        :type score_threshold: float
        :param base_url: The API endpoint for OLLAMA.
            If not provided, it will be retrieved from the `OLLAMA_BASE_URL` environment variable.
        :type base_url: str

        :raise ValueError: If the hosted base url is not provided properly or if the ollama
            client fails to initialize.
        """
        if name is None:
            name = EncoderDefault.OLLAMA.value["embedding_model"]

        super().__init__(name=name, score_threshold=score_threshold)
        if base_url is None:
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.client = self._initialize_client(base_url=base_url)

    def _initialize_client(self, base_url: str):
        """Initializes the Google AI Platform client.

        :param base_url: hosted URL of ollama.
        :return: An instance of the TextEmbeddingModel client.
        :rtype: TextEmbeddingModel
        :raise ImportError: If the required ollama library is not installed.
        :raise ValueError: If the hosted base url is not provided properly or if the ollama
            client fails to initialize.
        """
        try:
            from ollama import Client
        except ImportError:
            raise ImportError(
                "The 'ollama' package is not installed. Install it with: pip install 'semantic-router[ollama]'"
            )

        client: Client = Client(host=base_url)
        return client

    def __call__(self, docs: List[str]) -> List[List[float]]:
        """Generates embeddings for the given documents.

        :param docs: A list of strings representing the documents to embed.
        :type docs: List[str]
        :return: A list of lists, where each inner list contains the embedding values for a
            document.
        :rtype: List[List[float]]
        :raise ValueError: If the Google AI Platform client is not initialized or if the
            API call fails.
        """
        if self.client is None:
            raise ValueError("OLLAMA Platform client is not initialized.")
        try:
            return self.client.embed(model=self.name, input=docs).embeddings
        except Exception as e:
            raise ValueError(f"OLLAMA API call failed. Error: {e}") from e
