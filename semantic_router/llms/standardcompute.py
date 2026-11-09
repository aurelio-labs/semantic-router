import os
from typing import Optional

from semantic_router.llms.openai_compatible import OpenAICompatibleLLM


class StandardComputeLLM(OpenAICompatibleLLM):
    """LLM for Standard Compute, a preset of OpenAICompatibleLLM pointing at the
    Standard Compute gateway (https://api.stdcmpt.com/v1) and reading the API key
    from STANDARDCOMPUTE_API_KEY. See https://standardcompute.com for more
    information.

    The default model, StandardCompute, is routed: each request goes to the
    cheapest enabled model that can handle it, with automatic fallback. Pass a
    pinned model id from GET https://api.stdcmpt.com/v1/models to use one model.

    Limits inherited from semantic-router 0.x:

    - Dynamic routes extract function inputs only when called synchronously;
      an async call returns the route with function_call=None.
    - A router config file saves only the class and model name. Reloading one
      uses the default base_url and STANDARDCOMPUTE_API_KEY, so a custom base_url
      or a key passed in code is lost."""

    def __init__(
        self,
        name: Optional[str] = None,
        standardcompute_api_key: Optional[str] = None,
        base_url: str = "https://api.stdcmpt.com/v1",
        temperature: float = 0.01,
        max_tokens: int = 200,
    ):
        """Initialize the StandardComputeLLM.

        :param name: The name of the Standard Compute model to use. Defaults to
            the STANDARDCOMPUTE_CHAT_MODEL_NAME environment variable or
            StandardCompute.
        :type name: Optional[str]
        :param standardcompute_api_key: The Standard Compute API key. Defaults to
            the STANDARDCOMPUTE_API_KEY environment variable.
        :type standardcompute_api_key: Optional[str]
        :param base_url: The base URL for the Standard Compute API.
        :type base_url: str
        :param temperature: The temperature of the LLM.
        :type temperature: float
        :param max_tokens: The maximum number of tokens to generate.
        :type max_tokens: int
        """
        if name is None:
            name = os.getenv("STANDARDCOMPUTE_CHAT_MODEL_NAME", "StandardCompute")
        super().__init__(
            name=name,
            base_url=base_url,
            api_key=standardcompute_api_key,
            api_key_var_name="STANDARDCOMPUTE_API_KEY",
            temperature=temperature,
            max_tokens=max_tokens,
        )
