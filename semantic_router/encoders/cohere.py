import os
from typing import Any, List, Optional

from cohere import EmbedInputType
from pydantic import PrivateAttr

from semantic_router.encoders import DenseEncoder
from semantic_router.encoders.encode_input_type import EncodeInputType
from semantic_router.utils.defaults import EncoderDefault


class CohereEncoder(DenseEncoder):
    """Dense encoder that uses Cohere API to embed documents. Supports text only. Requires
    a Cohere API key from https://dashboard.cohere.com/api-keys.
    """

    _client: Any = PrivateAttr()
    _embed_type: Any = PrivateAttr()
    type: str = "cohere"

    def __init__(
        self,
        name: Optional[str] = None,
        cohere_api_key: Optional[str] = None,
        score_threshold: float = 0.3,
    ):
        """Initialize the Cohere encoder.

        :param name: The name of the embedding model to use.
        :type name: str
        :param cohere_api_key: The API key for the Cohere client, can also
            be set via the COHERE_API_KEY environment variable.
        :type cohere_api_key: str
        :param score_threshold: The threshold for the score of the embedding.
        :type score_threshold: float
        :param input_type: The type of input to embed.
        :type input_type: str
        """
        if name is None:
            name = EncoderDefault.COHERE.value["embedding_model"]
        super().__init__(
            name=name,
            score_threshold=score_threshold,
        )
        self._client = self._initialize_client(cohere_api_key)

    def _initialize_client(self, cohere_api_key: Optional[str] = None):
        """Initializes the Cohere client.

        :param cohere_api_key: The API key for the Cohere client, can also
            be set via the COHERE_API_KEY environment variable.
        :type cohere_api_key: str
        :return: An instance of the Cohere client.
        :rtype: cohere.Client
        """
        try:
            import cohere
            from cohere.types.embed_response import EmbeddingsByTypeEmbedResponse

            self._embed_type = EmbeddingsByTypeEmbedResponse
        except ImportError:
            raise ImportError(
                "Please install Cohere to use CohereEncoder. "
                "You can install it with: "
                "`pip install 'semantic-router[cohere]'`"
            )
        cohere_api_key = cohere_api_key or os.getenv("COHERE_API_KEY")
        if cohere_api_key is None:
            raise ValueError("Cohere API key cannot be 'None'.")
        try:
            client = cohere.Client(cohere_api_key)
        except Exception as e:
            raise ValueError(
                f"Cohere API client failed to initialize. Error: {e}"
            ) from e
        return client

    def __call__(
        self, docs: List[str], input_type: EncodeInputType
    ) -> List[List[float]]:
        """Embed a list of documents. Supports text only.

        :param docs: The documents to embed.
        :type docs: List[str]
        :param input_type: Specify whether encoding 'queries' or 'documents', used in asymmetric retrieval
        :type input_type: semantic_router.encoders.encode_input_type.EncodeInputType
        :return: The vector embeddings of the documents.
        :rtype: List[List[float]]
        """
        if self._client is None:
            raise ValueError("Cohere client is not initialized.")

        cohere_input_type: EmbedInputType = None
        match input_type:
            case "queries":
                cohere_input_type = "search_query"
            case "documents":
                cohere_input_type = "search_document"
        try:
            embeds = self._client.embed(
                texts=docs, input_type=cohere_input_type, model=self.name
            )
            # Check for unsupported type.
            if isinstance(embeds, self._embed_type):
                raise NotImplementedError(
                    "Handling of EmbedByTypeResponseEmbeddings is not implemented."
                )
            else:
                return embeds.embeddings
        except Exception as e:
            raise ValueError(f"Cohere API call failed. Error: {e}") from e
