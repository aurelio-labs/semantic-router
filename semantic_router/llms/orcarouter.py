import os
from typing import Optional

from semantic_router.llms.openai_compatible import OpenAICompatibleLLM


class OrcaRouterLLM(OpenAICompatibleLLM):
    """LLM for OrcaRouter, a preset of OpenAICompatibleLLM pointing at the
    OrcaRouter gateway (https://api.orcarouter.ai/v1) and reading the API key
    from ORCAROUTER_API_KEY. See https://www.orcarouter.ai for more information."""

    def __init__(
        self,
        name: Optional[str] = None,
        orcarouter_api_key: Optional[str] = None,
        base_url: str = "https://api.orcarouter.ai/v1",
        temperature: float = 0.01,
        max_tokens: int = 200,
    ):
        """Initialize the OrcaRouterLLM.

        :param name: The name of the OrcaRouter model to use. Defaults to the
            ORCAROUTER_CHAT_MODEL_NAME environment variable or orcarouter/auto.
        :type name: Optional[str]
        :param orcarouter_api_key: The OrcaRouter API key. Defaults to the
            ORCAROUTER_API_KEY environment variable.
        :type orcarouter_api_key: Optional[str]
        :param base_url: The base URL for the OrcaRouter API.
        :type base_url: str
        :param temperature: The temperature of the LLM.
        :type temperature: float
        :param max_tokens: The maximum number of tokens to generate.
        :type max_tokens: int
        """
        if name is None:
            name = os.getenv("ORCAROUTER_CHAT_MODEL_NAME", "orcarouter/auto")
        super().__init__(
            name=name,
            base_url=base_url,
            api_key=orcarouter_api_key,
            api_key_var_name="ORCAROUTER_API_KEY",
            temperature=temperature,
            max_tokens=max_tokens,
        )
