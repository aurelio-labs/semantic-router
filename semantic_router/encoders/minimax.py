import os
from typing import Any, Dict, List, Optional

import aiohttp
import requests
from pydantic import PrivateAttr

from semantic_router.encoders import DenseEncoder
from semantic_router.utils.defaults import EncoderDefault
from semantic_router.utils.logger import logger


class MiniMaxEncoder(DenseEncoder):
    """MiniMax encoder class for generating embeddings using the MiniMax API.

    Uses MiniMax's native embedding endpoint at https://api.minimax.io/v1/embeddings.
    Requires a MiniMax API key from https://platform.minimaxi.com/
    """

    _api_key: str = PrivateAttr(default="")
    _base_url: str = PrivateAttr(default="https://api.minimax.io/v1")
    type: str = "minimax"

    def __init__(
        self,
        name: Optional[str] = None,
        minimax_api_key: Optional[str] = None,
        base_url: str = "https://api.minimax.io/v1",
        score_threshold: float = 0.3,
    ):
        """Initialize the MiniMaxEncoder.

        :param name: The name of the MiniMax embedding model to use.
        :type name: Optional[str]
        :param minimax_api_key: The MiniMax API key.
        :type minimax_api_key: Optional[str]
        :param base_url: The base URL for the MiniMax API.
        :type base_url: str
        :param score_threshold: The score threshold for the embeddings.
        :type score_threshold: float
        """
        if name is None:
            name = EncoderDefault.MINIMAX.value["embedding_model"]
        super().__init__(name=name, score_threshold=score_threshold)
        api_key = minimax_api_key or os.getenv("MINIMAX_API_KEY")
        if api_key is None:
            raise ValueError("MiniMax API key cannot be 'None'.")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    def _build_request(self, docs: List[str]) -> Dict[str, Any]:
        """Build the request payload for the MiniMax embedding API.

        :param docs: List of text documents to encode.
        :return: Request payload dictionary.
        """
        return {
            "model": self.name,
            "texts": docs,
            "type": "db",
        }

    def _build_headers(self) -> Dict[str, str]:
        """Build the request headers.

        :return: Headers dictionary.
        """
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _parse_response(data: Dict[str, Any]) -> List[List[float]]:
        """Parse embeddings from the API response.

        :param data: API response JSON.
        :return: List of embedding vectors.
        """
        if data.get("base_resp", {}).get("status_code", 0) != 0:
            status_msg = data.get("base_resp", {}).get("status_msg", "Unknown error")
            raise ValueError(f"MiniMax API error: {status_msg}")
        vectors = data.get("vectors")
        if not vectors:
            raise ValueError("No embeddings returned from MiniMax API.")
        return vectors

    def __call__(self, docs: List[str], **kwargs: Any) -> List[List[float]]:
        """Encode a list of text documents into embeddings using MiniMax API.

        :param docs: List of text documents to encode.
        :type docs: List[str]
        :return: List of embeddings for each document.
        :rtype: List[List[float]]
        """
        try:
            url = f"{self._base_url}/embeddings"
            response = requests.post(
                url,
                json=self._build_request(docs),
                headers=self._build_headers(),
                timeout=30,
            )
            response.raise_for_status()
            return self._parse_response(response.json())
        except Exception as e:
            logger.error(f"MiniMax API call failed. Error: {e}")
            raise ValueError(f"MiniMax API call failed. Error: {e}") from e

    async def acall(self, docs: List[str], **kwargs: Any) -> List[List[float]]:
        """Encode a list of text documents into embeddings using MiniMax API
        asynchronously.

        :param docs: List of text documents to encode.
        :type docs: List[str]
        :return: List of embeddings for each document.
        :rtype: List[List[float]]
        """
        try:
            url = f"{self._base_url}/embeddings"
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=self._build_request(docs),
                    headers=self._build_headers(),
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return self._parse_response(data)
        except Exception as e:
            logger.error(f"MiniMax API call failed. Error: {e}")
            raise ValueError(f"MiniMax API call failed. Error: {e}") from e
