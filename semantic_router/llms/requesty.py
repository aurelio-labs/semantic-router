import os
from typing import Optional

from semantic_router.llms.openai_compatible import OpenAICompatibleLLM


class RequestyLLM(OpenAICompatibleLLM):
    """LLM for Requesty, a preset of OpenAICompatibleLLM pointing at the Requesty
    router (https://router.requesty.ai/v1) and reading the API key from
    REQUESTY_API_KEY. See https://docs.requesty.ai for more information and
    https://app.requesty.ai/api-keys to create a key."""

    def __init__(
        self,
        name: Optional[str] = None,
        requesty_api_key: Optional[str] = None,
        base_url: str = "https://router.requesty.ai/v1",
        temperature: float = 0.01,
        max_tokens: int = 200,
    ):
        """Initialize the RequestyLLM.

        :param name: The name of the Requesty model to use. Defaults to the
            REQUESTY_CHAT_MODEL_NAME environment variable or openai/gpt-4o-mini.
        :type name: Optional[str]
        :param requesty_api_key: The Requesty API key. Defaults to the
            REQUESTY_API_KEY environment variable.
        :type requesty_api_key: Optional[str]
        :param base_url: The base URL for the Requesty API.
        :type base_url: str
        :param temperature: The temperature of the LLM.
        :type temperature: float
        :param max_tokens: The maximum number of tokens to generate.
        :type max_tokens: int
        """
        if name is None:
            name = os.getenv("REQUESTY_CHAT_MODEL_NAME", "openai/gpt-4o-mini")
        super().__init__(
            name=name,
            base_url=base_url,
            api_key=requesty_api_key,
            api_key_var_name="REQUESTY_API_KEY",
            temperature=temperature,
            max_tokens=max_tokens,
        )
