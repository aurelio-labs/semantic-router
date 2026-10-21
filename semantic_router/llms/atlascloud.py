import os
from typing import Optional

from semantic_router.llms.openai_compatible import OpenAICompatibleLLM


class AtlasCloudLLM(OpenAICompatibleLLM):
    """LLM for Atlas Cloud, a preset of OpenAICompatibleLLM pointing at the Atlas
    Cloud gateway (https://api.atlascloud.ai/v1) and reading the API key from
    ATLASCLOUD_API_KEY. See https://docs.atlascloud.ai/ for more information."""

    def __init__(
        self,
        name: Optional[str] = None,
        atlascloud_api_key: Optional[str] = None,
        base_url: str = "https://api.atlascloud.ai/v1",
        temperature: float = 0.01,
        max_tokens: int = 200,
    ):
        """Initialize the AtlasCloudLLM.

        :param name: The name of the Atlas Cloud model to use. Defaults to the
            ATLASCLOUD_CHAT_MODEL_NAME environment variable or
            Qwen/Qwen3-235B-A22B-Instruct-2507.
        :type name: Optional[str]
        :param atlascloud_api_key: The Atlas Cloud API key. Defaults to the
            ATLASCLOUD_API_KEY environment variable.
        :type atlascloud_api_key: Optional[str]
        :param base_url: The base URL for the Atlas Cloud API.
        :type base_url: str
        :param temperature: The temperature of the LLM.
        :type temperature: float
        :param max_tokens: The maximum number of tokens to generate.
        :type max_tokens: int
        """
        if name is None:
            name = os.getenv(
                "ATLASCLOUD_CHAT_MODEL_NAME", "Qwen/Qwen3-235B-A22B-Instruct-2507"
            )
        super().__init__(
            name=name,
            base_url=base_url,
            api_key=atlascloud_api_key,
            api_key_var_name="ATLASCLOUD_API_KEY",
            temperature=temperature,
            max_tokens=max_tokens,
        )
