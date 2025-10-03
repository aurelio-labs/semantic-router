"""
This module provides the BedrockEncoder class for generating embeddings using Amazon's Bedrock Platform.

The BedrockEncoder class is a subclass of DenseEncoder and utilizes the TextEmbeddingModel from the
Amazon's Bedrock Platform to generate embeddings for given documents. It requires an AWS Access Key ID
and AWS Secret Access Key and supports customization of the pre-trained model, score threshold, and region.

Example usage:

    from semantic_router.encoders.bedrock_encoder import BedrockEncoder

    encoder = BedrockEncoder(access_key_id="your-access-key-id", secret_access_key="your-secret-key", region="your-region")
    embeddings = encoder(["document1", "document2"])

Classes:
    BedrockEncoder: A class for generating embeddings using the Bedrock Platform.
"""

import json
import os
from time import sleep
from typing import Any, Dict, List, Optional, Union

import tiktoken

from semantic_router.encoders import DenseEncoder
from semantic_router.utils.defaults import EncoderDefault
from semantic_router.utils.logger import logger


class BedrockEncoder(DenseEncoder):
    """Dense encoder using Amazon Bedrock embedding API. Requires an AWS Access Key ID
    and AWS Secret Access Key.

    The BedrockEncoder class is a subclass of DenseEncoder and utilizes the
    TextEmbeddingModel from the Amazon's Bedrock Platform to generate embeddings for
    given documents. It supports customization of the pre-trained model, score
    threshold, and region.

    Example usage:

    ```python
    from semantic_router.encoders.bedrock_encoder import BedrockEncoder

    encoder = BedrockEncoder(
        access_key_id="your-access-key-id",
        secret_access_key="your-secret-key",
        region="your-region"
    )
    embeddings = encoder(["document1", "document2"])
    ```
    """

    client: Any = None
    type: str = "bedrock"
    input_type: Optional[str] = "search_query"
    name: str
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    session_token: Optional[str] = None
    region: Optional[str] = None

    def __init__(
        self,
        name: str = EncoderDefault.BEDROCK.value["embedding_model"],
        input_type: Optional[str] = "search_query",
        score_threshold: float = 0.3,
        client: Optional[Any] = None,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        session_token: Optional[str] = None,
        region: Optional[str] = None,
    ):
        """Initializes the BedrockEncoder.

        :param name: The name of the pre-trained model to use for embedding.
            If not provided, the default model specified in EncoderDefault will
            be used.
        :type name: str
        :param input_type: The type of input to use for the embedding.
            If not provided, the default input type specified in EncoderDefault will
            be used.
        :type input_type: str
        :param score_threshold: The threshold for similarity scores.
        :type score_threshold: float
        :param access_key_id: The AWS access key id for an IAM principle.
            If not provided, it will be retrieved from the access_key_id
            environment variable.
        :type access_key_id: str
        :param secret_access_key: The secret access key for an IAM principle.
            If not provided, it will be retrieved from the AWS_SECRET_KEY
            environment variable.
        :type secret_access_key: str
        :param session_token: The session token for an IAM principle.
            If not provided, it will be retrieved from the AWS_SESSION_TOKEN
            environment variable.
        :param region: The location of the Bedrock resources.
            If not provided, it will be retrieved from the AWS_REGION
            environment variable, defaulting to "us-west-1"
        :type region: str
        :raises ValueError: If the Bedrock Platform client fails to initialize.
        """
        super().__init__(name=name, score_threshold=score_threshold)
        self.input_type = input_type
        if client:
            self.client = client
        else:
            self.access_key_id = self.get_env_variable("AWS_ACCESS_KEY_ID", access_key_id)
            self.secret_access_key = self.get_env_variable(
                "AWS_SECRET_ACCESS_KEY", secret_access_key
            )
            self.session_token = self.get_env_variable("AWS_SESSION_TOKEN", session_token)
            self.region = self.get_env_variable(
                "AWS_DEFAULT_REGION", region, default="us-west-1"
            )
            try:
                self.client = self._initialize_client(
                    self.access_key_id,
                    self.secret_access_key,
                    self.session_token,
                    self.region,
                )
            except Exception as e:
                raise ValueError(f"Bedrock client failed to initialise. Error: {e}") from e

    def _initialize_client(
        self, access_key_id, secret_access_key, session_token, region
    ):
        """Initializes the Bedrock client.

        :param access_key_id: The Amazon access key ID.
        :type access_key_id: str
        :param secret_access_key: The Amazon secret key.
        :type secret_access_key: str
        :param region: The location of the AI Platform resources.
        :type region: str
        :returns: An instance of the TextEmbeddingModel client.
        :rtype: Any
        :raises ImportError: If the required Bedrock libraries are not
            installed.
            ValueError: If the Bedrock client fails to initialize.
        """
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "Please install Amazon's Boto3 client library to use the BedrockEncoder. "
                "You can install them with: "
                "`pip install boto3`"
            )
        access_key_id = access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
        region = region or os.getenv("AWS_DEFAULT_REGION", "us-west-2")
        if access_key_id is None:
            raise ValueError("AWS access key ID cannot be 'None'.")
        if aws_secret_key is None:
            raise ValueError("AWS secret access key cannot be 'None'.")
        session = boto3.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            aws_session_token=session_token,
        )
        try:
            bedrock_client = session.client(
                "bedrock-runtime",
                region_name=region,
            )
        except Exception as err:
            raise ValueError(
                f"The Bedrock client failed to initialize. Error: {err}"
            ) from err
        return bedrock_client

    def __call__(
        self, docs: List[Union[str, Dict]], model_kwargs: Optional[Dict] = None
    ) -> List[List[float]]:
        """Generates embeddings for the given documents.

        :param docs: A list of strings representing the documents to embed.
        :type docs: list[str]
        :param model_kwargs: A dictionary of model-specific inference parameters.
        :type model_kwargs: dict
        :returns: A list of lists, where each inner list contains the embedding values for a
            document.
        :rtype: list[list[float]]
        :raises ValueError: If the Bedrock Platform client is not initialized or if the
            API call fails.
        """
        try:
            from botocore.exceptions import ClientError
        except ImportError:
            raise ImportError(
                "Please install Amazon's Botocore client library to use the BedrockEncoder. "
                "You can install them with: "
                "`pip install botocore`"
            )
        if self.client is None:
            raise ValueError("Bedrock client is not initialised.")
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                embeddings = []
                if self.name and "amazon" in self.name:
                    for doc in docs:
                        embedding_body = {}

                        if isinstance(doc, dict):
                            embedding_body["inputText"] = doc.get("text")
                            embedding_body["inputImage"] = doc.get(
                                "image"
                            )  # expects a base64-encoded image
                        else:
                            embedding_body["inputText"] = doc

                        # Add model-specific inference parameters
                        if model_kwargs:
                            embedding_body = embedding_body | model_kwargs

                        # Clean up null values
                        embedding_body = {k: v for k, v in embedding_body.items() if v}

                        # Format payload
                        embedding_body_payload: str = json.dumps(embedding_body)

                        response = self.client.invoke_model(
                            body=embedding_body_payload,
                            modelId=self.name,
                            accept="application/json",
                            contentType="application/json",
                        )
                        response_body = json.loads(response.get("body").read())
                        embeddings.append(response_body.get("embedding"))
                elif self.name and "cohere" in self.name:
                    chunked_docs = self.chunk_strings(docs)
                    for chunk in chunked_docs:
                        chunk = {"texts": chunk, "input_type": self.input_type}

                        # Add model-specific inference parameters
                        # Note: if specified, input_type will be overwritten by model_kwargs
                        if model_kwargs:
                            chunk = chunk | model_kwargs

                        # Format payload
                        chunk = json.dumps(chunk)

                        response = self.client.invoke_model(
                            body=chunk,
                            modelId=self.name,
                            accept="*/*",
                            contentType="application/json",
                        )
                        response_body = json.loads(response.get("body").read())
                        chunk_embeddings = response_body.get("embeddings")
                        embeddings.extend(chunk_embeddings)
                else:
                    raise ValueError("Unknown model name")
                return embeddings
            except ClientError as error:
                if attempt < max_attempts - 1:
                    if error.response["Error"]["Code"] == "ExpiredTokenException":
                        logger.warning(
                            "Session token has expired. Retrying initialisation."
                        )
                        try:
                            self.session_token = os.getenv("AWS_SESSION_TOKEN")
                            self.client = self._initialize_client(
                                self.access_key_id,
                                self.secret_access_key,
                                self.session_token,
                                self.region,
                            )
                        except Exception as e:
                            raise ValueError(
                                f"Bedrock client failed to reinitialise. Error: {e}"
                            ) from e
                    sleep(2**attempt)
                    logger.warning(f"Retrying in {2**attempt} seconds...")
                raise ValueError(
                    f"Retries exhausted, Bedrock call failed. Error: {error}"
                ) from error
            except Exception as e:
                raise ValueError(f"Bedrock call failed. Error: {e}") from e
        raise ValueError("Bedrock call failed to return embeddings.")

    def chunk_strings(self, strings, MAX_WORDS=20):
        """Breaks up a list of strings into smaller chunks.

        :param strings: A list of strings to be chunked.
        :type strings: list
        :param max_chunk_size: The maximum size of each chunk. Default is 20.
        :type max_chunk_size: int
        :returns: A list of lists, where each inner list contains a chunk of strings.
        :rtype: list[list[str]]
        """
        encoding = tiktoken.get_encoding("cl100k_base")
        chunked_strings = []
        for text in strings:
            encoded_text = encoding.encode(text)
            chunks = [
                encoding.decode(encoded_text[i : i + MAX_WORDS])
                for i in range(0, len(encoded_text), MAX_WORDS)
            ]
            chunked_strings.append(chunks)
        return chunked_strings

    @staticmethod
    def get_env_variable(var_name, provided_value, default=None):
        """Retrieves environment variable or uses a provided value.

        :param var_name: The name of the environment variable.
        :type var_name: str
        :param provided_value: The provided value to use if not None.
        :type provided_value: Optional[str]
        :param default: The default value if the environment variable is not set.
        :type default: Optional[str]
        :returns: The value of the environment variable or the provided/default value.
        :rtype: str
        :raises ValueError: If no value is provided and the environment variable is not set.
        """
        if provided_value is not None:
            return provided_value
        value = os.getenv(var_name, default)
        if value is None:
            if var_name == "AWS_SESSION_TOKEN":
                return None
            raise ValueError(f"No {var_name} provided")
        return value
