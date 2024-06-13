import os
from time import sleep
from typing import Any, List, Optional

from gigachat import GigaChat # Install the GigaChat API library via 'pip install gigachat'

from semantic_router.encoders import BaseEncoder
from semantic_router.utils.defaults import EncoderDefault

class GigaChatEncoder(BaseEncoder):
    """GigaChat encoder class for generating embeddings.

        Attributes:
            client (Optional[Any]): Instance of the GigaChat client.
            type (str): Type identifier for the encoder, which is "gigachat".
    """

    client: Optional[Any] = None
    type: str = "gigachat"

    def __init__(self,
                 name: Optional[str] = None,
                 auth_data: Optional[str] = None,
                 scope: Optional[str] = None,
                 score_threshold: float = 0.75
        ):
        """Initializes the GigaChatEncoder.

        Args:
            name (Optional[str]): Name of the encoder model.
            auth_data (Optional[str]): Authorization data for GigaChat.
            scope (Optional[str]): Scope of the GigaChat API usage.
            score_threshold (float): Threshold for scoring embeddings.

        Raises:
            ValueError: If auth_data or scope is None.
        """
        if name is None:
            name = EncoderDefault.GIGACHAT.value["embedding_model"]
        super().__init__(name=name, score_threshold=score_threshold)
        auth_data = auth_data or os.getenv("GIGACHAT_AUTH_DATA")
        if auth_data is None:
            raise ValueError("GigaChat authorization data cannot be 'None'.")
        if scope is None:
            raise ValueError("GigaChat scope cannot be 'None'. Set 'GIGACHAT_API_PERS' for personal use or 'GIGACHAT_API_CORP' for corporate use.")
        try:
            self.client = GigaChat(scope=scope, credentials=auth_data, verify_ssl_certs=False)
        except Exception as e:
            raise ValueError(
                f"GigaChat client failed to initialize. Error: {e}"
            ) from e

    def __call__(self, docs: List[str]) -> List[List[float]]:
        """Generates embeddings for a list of documents.

        Args:
            docs: List of documents to generate embeddings for.

        Returns:
            List: List of embeddings for each document.

        Raises:
            ValueError: If the client is not initialized or the GigaChat call fails.
        """
        if self.client is None:
            raise ValueError("GigaChat client is not initialized.")
        try:
            embeddings = self.client.embeddings(docs).data
            embeddings = [embeds_obj.embedding for embeds_obj in embeddings]
            return embeddings
        except Exception as e:
            raise ValueError(f"GigaChat call failed. Error: {e}") from e