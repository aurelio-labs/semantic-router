"""This file contains the MistralEncoder class which is used to encode text using MistralAI"""

import os
from typing import Any

from pydantic import PrivateAttr
from typing_extensions import deprecated

from semantic_router.encoders.litellm import LiteLLMEncoder
from semantic_router.utils.defaults import EncoderDefault


class MistralEncoder(LiteLLMEncoder):
    """Class to encode text using MistralAI. Requires a MistralAI API key from
    https://console.mistral.ai/api-keys/"""

    _client: Any = PrivateAttr()  # TODO: deprecated, to remove in v0.2.0
    _mistralai: Any = PrivateAttr()  # TODO: deprecated, to remove in v0.2.0
    type: str = "mistral"

    def __init__(
        self,
        name: str | None = None,
        mistralai_api_key: str | None = None,  # TODO: rename to api_key in v0.2.0
        score_threshold: float = 0.4,
    ):
        """Initialize the MistralEncoder.

        :param name: The name of the embedding model to use such as "mistral-embed".
        :type name: str
        :param mistralai_api_key: The MistralAI API key.
        :type mistralai_api_key: str
        :param score_threshold: The score threshold for the embeddings.
        """
        # get default model name if none provided and convert to litellm format
        if name is None:
            name = f"mistral/{EncoderDefault.MISTRAL.value['embedding_model']}"
        elif not name.startswith("mistral/"):
            name = f"mistral/{name}"
        if mistralai_api_key is None:
            mistralai_api_key = os.getenv("MISTRALAI_API_KEY")
        if mistralai_api_key is None:
            mistralai_api_key = os.getenv("MISTRAL_API_KEY")
        super().__init__(
            name=name,
            score_threshold=score_threshold,
            api_key=mistralai_api_key,
        )

    # TODO: deprecated, to remove in v0.2.0
    @deprecated("_initialize_client method no longer required")
    def _initialize_client(self, api_key):
        """Initialize the MistralAI client.

        :param api_key: The MistralAI API key.
        :type api_key: str
        :return: None
        :rtype: None
        """
        api_key = (
            api_key or os.getenv("MISTRALAI_API_KEY") or os.getenv("MISTRAL_API_KEY")
        )
        if api_key is None:
            raise ValueError("Mistral API key not provided")
        return None
