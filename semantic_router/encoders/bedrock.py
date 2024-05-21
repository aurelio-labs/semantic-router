"""
This module provides the BedrockEncoder class for generating embeddings using Amazon's Bedrock Platform.

The BedrockEncoder class is a subclass of BaseEncoder and utilizes the TextEmbeddingModel from the
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
from typing import List, Optional, Any
import os
from time import sleep
import tiktoken
from semantic_router.encoders import BaseEncoder
from semantic_router.utils.defaults import EncoderDefault
from semantic_router.utils.logger import logger


class BedrockEncoder(BaseEncoder):
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
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        session_token: Optional[str] = None,
        region: Optional[str] = None,
    ):
        """Initializes the BedrockEncoder.

        Args:
            name: The name of the pre-trained model to use for embedding.
                If not provided, the default model specified in EncoderDefault will
                be used.
            score_threshold: The threshold for similarity scores.
            access_key_id: The AWS access key id for an IAM principle.
                If not provided, it will be retrieved from the access_key_id
                environment variable.
            secret_access_key: The secret access key for an IAM principle.
                If not provided, it will be retrieved from the AWS_SECRET_KEY
                environment variable.
            session_token: The session token for an IAM principle.
                If not provided, it will be retrieved from the AWS_SESSION_TOKEN
                environment variable.
            region: The location of the Bedrock resources.
                If not provided, it will be retrieved from the AWS_REGION
                environment variable, defaulting to "us-west-1"

        Raises:
            ValueError: If the Bedrock Platform client fails to initialize.
        """
        super().__init__(name=name, score_threshold=score_threshold)
        self.access_key_id = self.get_env_variable("AWS_ACCESS_KEY_ID", access_key_id)
        self.secret_access_key = self.get_env_variable(
            "AWS_SECRET_ACCESS_KEY", secret_access_key
        )
        self.session_token = self.get_env_variable("AWS_SESSION_TOKEN", session_token)
        self.region = self.get_env_variable(
            "AWS_DEFAULT_REGION", region, default="us-west-1"
        )
        self.input_type = input_type
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

        Args:
            access_key_id: The Amazon access key ID.
            secret_access_key: The Amazon secret key.
            region: The location of the AI Platform resources.

        Returns:
            An instance of the TextEmbeddingModel client.

        Raises:
            ImportError: If the required Bedrock libraries are not
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

    def __call__(self, docs: List[str]) -> List[List[float]]:
        """Generates embeddings for the given documents.

        Args:
            docs: A list of strings representing the documents to embed.

        Returns:
            A list of lists, where each inner list contains the embedding values for a
            document.

        Raises:
            ValueError: If the Bedrock Platform client is not initialized or if the
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
                        embedding_body = json.dumps(
                            {
                                "inputText": doc,
                            }
                        )
                        response = self.client.invoke_model(
                            body=embedding_body,
                            modelId=self.name,
                            accept="application/json",
                            contentType="application/json",
                        )
                        response_body = json.loads(response.get("body").read())
                        embeddings.append(response_body.get("embedding"))
                elif self.name and "cohere" in self.name:
                    chunked_docs = self.chunk_strings(docs)
                    for chunk in chunked_docs:
                        chunk = json.dumps(
                            {"texts": chunk, "input_type": self.input_type}
                        )
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
        """
        Breaks up a list of strings into smaller chunks.

        Args:
            strings (list): A list of strings to be chunked.
            max_chunk_size (int): The maximum size of each chunk. Default is 20.

        Returns:
            list: A list of lists, where each inner list contains a chunk of strings.
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

        Args:
            var_name (str): The name of the environment variable.
            provided_value (Optional[str]): The provided value to use if not None.
            default (Optional[str]): The default value if the environment variable is not set.

        Returns:
            str: The value of the environment variable or the provided/default value.
            None: Where AWS_SESSION_TOKEN is not set or provided

        Raises:
            ValueError: If no value is provided and the environment variable is not set.
        """
        if provided_value is not None:
            return provided_value
        value = os.getenv(var_name, default)
        if value is None:
            if var_name == "AWS_SESSION_TOKEN":
                return None
            raise ValueError(f"No {var_name} provided")
        return value
