import os
from asyncio import sleep as asleep
from time import sleep
from typing import Any, Callable, Dict, List, Optional, Union

import httpx
import openai
from openai import OpenAIError
from openai._types import NotGiven
from openai.types import CreateEmbeddingResponse

from semantic_router.encoders import DenseEncoder
from semantic_router.utils.defaults import EncoderDefault
from semantic_router.utils.logger import logger


class AzureOpenAIEncoder(DenseEncoder):
    """Encoder for Azure OpenAI API.

    This class provides functionality to encode text documents using the Azure OpenAI API.
    It supports customization of the score threshold for filtering or processing the embeddings.
    """

    client: Optional[openai.AzureOpenAI] = None
    async_client: Optional[openai.AsyncAzureOpenAI] = None
    dimensions: Union[int, NotGiven] = NotGiven()
    type: str = "azure"
    deployment_name: str | None = None
    max_retries: int = 3

    def __init__(
        self,
        name: Optional[str] = None,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        api_key: str | None = None,
        azure_ad_token: str | None = None,
        azure_ad_token_provider: Callable[[], str] | None = None,
        http_client_options: Optional[Dict[str, Any]] = None,
        deployment_name: str = EncoderDefault.AZURE.value["deployment_name"],
        score_threshold: float = 0.82,
        dimensions: Union[int, NotGiven] = NotGiven(),
        max_retries: int = 3,
    ):
        """Initialize the AzureOpenAIEncoder.

        :param azure_endpoint: The endpoint for the Azure OpenAI API.
            Example: `"https://accountname.openai.azure.com"`
        :type azure_endpoint: str, optional

        :param api_version: The version of the API to use.
            Example: `"2025-02-01-preview"`
        :type api_version: str, optional

        :param api_key: The API key for the Azure OpenAI API.
        :type api_key: str, optional

        :param azure_ad_token: The Azure AD/Entra ID token for authentication.
            https://www.microsoft.com/en-us/security/business/identity-access/microsoft-entra-id
        :type azure_ad_token: str, optional

        :param azure_ad_token_provider: A callable function that returns an Azure AD/Entra ID token.
        :type azure_ad_token_provider: Callable[[], str], optional

        :param http_client_options: Dictionary of options to configure httpx client
            Example:
            ```
                {
                    "proxies": "http://proxy.server:8080",
                    "timeout": 20.0,
                    "headers": {"Authorization": "Bearer xyz"}
                }
            ```
        :type http_client_options: Dict[str, Any], optional

        :param deployment_name: The name of the model deployment to use.
        :type deployment_name: str, optional

        :param score_threshold: The score threshold for filtering embeddings.
            Default is `0.82`.
        :type score_threshold: float, optional

        :param dimensions: The number of dimensions for the embeddings. If not given, it defaults to the model's default setting.
        :type dimensions: int, optional

        :param max_retries: The maximum number of retries for API calls in case of failures.
            Default is `3`.
        :type max_retries: int, optional
        """
        if name is None:
            name = deployment_name
            if name is None:
                name = EncoderDefault.AZURE.value["embedding_model"]
        super().__init__(name=name, score_threshold=score_threshold)

        azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        if not azure_endpoint:
            raise ValueError("No Azure OpenAI endpoint provided.")

        api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION")
        if not api_version:
            raise ValueError("No Azure OpenAI API version provided.")

        if not (
            azure_ad_token
            or azure_ad_token_provider
            or api_key
            or os.getenv("AZURE_OPENAI_API_KEY")
        ):
            raise ValueError(
                "No authentication method provided. Please provide either `azure_ad_token`, "
                "`azure_ad_token_provider`, or `api_key`."
            )

        # Only check API Key if no AD token or provider is used
        if not azure_ad_token and not azure_ad_token_provider:
            api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
            if not api_key:
                raise ValueError("No Azure OpenAI API key provided.")

        self.deployment_name = deployment_name

        # set dimensions to support openai embed 3 dimensions param
        self.dimensions = dimensions

        if max_retries is not None:
            self.max_retries = max_retries

        # Only create HTTP clients if options are provided
        sync_http_client = (
            httpx.Client(**http_client_options) if http_client_options else None
        )
        async_http_client = (
            httpx.AsyncClient(**http_client_options) if http_client_options else None
        )

        assert azure_endpoint is not None and self.deployment_name is not None

        try:
            self.client = openai.AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_version=api_version,
                api_key=api_key,
                azure_ad_token=azure_ad_token,
                azure_ad_token_provider=azure_ad_token_provider,
                http_client=sync_http_client,
            )
            self.async_client = openai.AsyncAzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_version=api_version,
                api_key=api_key,
                azure_ad_token=azure_ad_token,
                azure_ad_token_provider=azure_ad_token_provider,
                http_client=async_http_client,
            )

        except Exception as e:
            logger.error("OpenAI API client failed to initialize. Error: %s", e)
            raise ValueError(
                f"OpenAI API client failed to initialize. Error: {e}"
            ) from e

    def __call__(self, docs: List[str]) -> List[List[float]]:
        """Encode a list of documents into embeddings using the Azure OpenAI API.

        :param docs: The documents to encode.
        :type docs: List[str]
        :return: The embeddings for the documents.
        :rtype: List[List[float]]
        """
        if self.client is None:
            raise ValueError("Azure OpenAI client is not initialized.")
        embeds = None

        # Exponential backoff
        for j in range(self.max_retries + 1):
            try:
                embeds = self.client.embeddings.create(
                    input=docs,
                    model=str(self.deployment_name),
                    dimensions=self.dimensions,  # type: ignore[arg-type]  # NotGiven vs Omit - ignore type errors between openai SDK <2.0.0 and >=2.0.0
                )
                if embeds.data:
                    break
            except OpenAIError as e:
                logger.error("Exception occurred", exc_info=True)
                if self.max_retries != 0 and j < self.max_retries:
                    sleep(2**j)
                    logger.warning(
                        "Retrying in %d seconds due to OpenAIError: %s", 2**j, e
                    )
                else:
                    raise
            except Exception as e:
                logger.error("Azure OpenAI API call failed. Error: %s", e)
                raise ValueError(f"Azure OpenAI API call failed. Error: {e}") from e

        if (
            not embeds
            or not isinstance(embeds, CreateEmbeddingResponse)
            or not embeds.data
        ):
            raise ValueError("No embeddings returned.")

        embeddings = [embeds_obj.embedding for embeds_obj in embeds.data]
        return embeddings

    async def acall(self, docs: List[str]) -> List[List[float]]:
        """Encode a list of documents into embeddings using the Azure OpenAI API asynchronously.

        :param docs: The documents to encode.
        :type docs: List[str]
        :return: The embeddings for the documents.
        :rtype: List[List[float]]
        """
        if self.async_client is None:
            raise ValueError("Azure OpenAI async client is not initialized.")
        embeds = None
        # Exponential backoff
        for j in range(self.max_retries + 1):
            try:
                embeds = await self.async_client.embeddings.create(
                    input=docs,
                    model=str(self.deployment_name),
                    dimensions=self.dimensions,  # type: ignore[arg-type]  # NotGiven vs Omit - ignore type errors between openai SDK <2.0.0 and >=2.0.0
                )
                if embeds.data:
                    break
            except OpenAIError as e:
                logger.error("Exception occurred", exc_info=True)
                if self.max_retries != 0 and j < self.max_retries:
                    await asleep(2**j)
                    logger.warning(
                        "Retrying in %d seconds due to OpenAIError: %s", 2**j, e
                    )
                else:
                    raise
            except Exception as e:
                logger.error("Azure OpenAI API call failed. Error: %s", e)
                raise ValueError(f"Azure OpenAI API call failed. Error: {e}") from e

        if (
            not embeds
            or not isinstance(embeds, CreateEmbeddingResponse)
            or not embeds.data
        ):
            raise ValueError("No embeddings returned.")

        embeddings = [embeds_obj.embedding for embeds_obj in embeds.data]
        return embeddings
