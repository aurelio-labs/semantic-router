import os
from time import sleep
from typing import List, Optional

import openai
from openai import OpenAIError
from openai.types import CreateEmbeddingResponse

from semantic_router.encoders import BaseEncoder
from semantic_router.utils.defaults import EncoderDefault
from semantic_router.utils.logger import logger


class AzureOpenAIEncoder(BaseEncoder):
    client: Optional[openai.AzureOpenAI] = None
    type: str = "azure"
    api_key: Optional[str] = None
    deployment_name: Optional[str] = None
    azure_endpoint: Optional[str] = None
    api_version: Optional[str] = None
    model: Optional[str] = None

    def __init__(
        self,
        api_key: Optional[str] = None,
        deployment_name: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        model: Optional[str] = None,
        score_threshold: float = 0.82,
    ):
        name = deployment_name
        if name is None:
            name = EncoderDefault.AZURE.value["embedding_model"]
        super().__init__(name=name, score_threshold=score_threshold)
        self.api_key = api_key
        self.deployment_name = deployment_name
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version
        self.model = model
        if self.api_key is None:
            self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
            if self.api_key is None:
                raise ValueError("No Azure OpenAI API key provided.")
        if self.deployment_name is None:
            self.deployment_name = EncoderDefault.AZURE.value["deployment_name"]
        # deployment_name may still be None, but it is optional in the API
        if self.azure_endpoint is None:
            self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            if self.azure_endpoint is None:
                raise ValueError("No Azure OpenAI endpoint provided.")
        if self.api_version is None:
            self.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
            if self.api_version is None:
                raise ValueError("No Azure OpenAI API version provided.")
        if self.model is None:
            self.model = os.getenv("AZURE_OPENAI_MODEL")
            if self.model is None:
                raise ValueError("No Azure OpenAI model provided.")
        assert (
            self.api_key is not None
            and self.azure_endpoint is not None
            and self.api_version is not None
            and self.model is not None
        )

        try:
            self.client = openai.AzureOpenAI(
                azure_deployment=(
                    str(self.deployment_name) if self.deployment_name else None
                ),
                api_key=str(self.api_key),
                azure_endpoint=str(self.azure_endpoint),
                api_version=str(self.api_version),
                # _strict_response_validation=True,
            )
        except Exception as e:
            raise ValueError(
                f"OpenAI API client failed to initialize. Error: {e}"
            ) from e

    def __call__(self, docs: List[str]) -> List[List[float]]:
        if self.client is None:
            raise ValueError("OpenAI client is not initialized.")
        embeds = None
        error_message = ""

        # Exponential backoff
        for j in range(3):
            try:
                embeds = self.client.embeddings.create(
                    input=docs, model=str(self.model)
                )
                if embeds.data:
                    break
            except OpenAIError as e:
                # print full traceback
                import traceback

                traceback.print_exc()
                sleep(2**j)
                error_message = str(e)
                logger.warning(f"Retrying in {2**j} seconds...")
            except Exception as e:
                logger.error(f"Azure OpenAI API call failed. Error: {error_message}")
                raise ValueError(f"Azure OpenAI API call failed. Error: {e}") from e

        if (
            not embeds
            or not isinstance(embeds, CreateEmbeddingResponse)
            or not embeds.data
        ):
            raise ValueError(f"No embeddings returned. Error: {error_message}")

        embeddings = [embeds_obj.embedding for embeds_obj in embeds.data]
        return embeddings
