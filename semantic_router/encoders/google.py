"""
This module provides the GoogleEncoder class for generating embeddings using Google's AI Platform.

The GoogleEncoder class is a subclass of BaseEncoder and utilizes the TextEmbeddingModel from the
Google AI Platform to generate embeddings for given documents. It requires a Google Cloud project ID
and supports customization of the pre-trained model, score threshold, location, and API endpoint.

Example usage:

    from semantic_router.encoders.google_encoder import GoogleEncoder

    encoder = GoogleEncoder(project_id="your-project-id")
    embeddings = encoder(["document1", "document2"])

Classes:
    GoogleEncoder: A class for generating embeddings using Google's AI Platform.
"""

import os
from typing import List, Optional

from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel

from semantic_router.encoders import BaseEncoder
from semantic_router.utils.defaults import EncoderDefault


class GoogleEncoder(BaseEncoder):
    """GoogleEncoder class for generating embeddings using Google's AI Platform.

    Attributes:
        client: An instance of the TextEmbeddingModel client.
        type: The type of the encoder, which is "google".
    """

    client: Optional[TextEmbeddingModel] = None
    type: str = "google"

    def __init__(
        self,
        name: Optional[str] = None,
        score_threshold: float = 0.75,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        api_endpoint: Optional[str] = None,
    ):
        """Initializes the GoogleEncoder.

        Args:
            model_name: The name of the pre-trained model to use for embedding.
                If not provided, the default model specified in EncoderDefault will be used.
            score_threshold: The threshold for similarity scores. Default is 0.3.
            project_id: The Google Cloud project ID.
                If not provided, it will be retrieved from the GOOGLE_PROJECT_ID environment variable.
            location: The location of the AI Platform resources.
                If not provided, it will be retrieved from the GOOGLE_LOCATION environment variable,
                defaulting to "us-central1".
            api_endpoint: The API endpoint for the AI Platform.
                If not provided, it will be retrieved from the GOOGLE_API_ENDPOINT environment variable.

        Raises:
            ValueError: If the Google Project ID is not provided or if the AI Platform client fails to initialize.
        """
        if name is None:
            name = EncoderDefault.GOOGLE.value["embedding_model"]

        super().__init__(name=name, score_threshold=score_threshold)

        project_id = project_id or os.getenv("GOOGLE_PROJECT_ID")
        location = location or os.getenv("GOOGLE_LOCATION", "us-central1")
        api_endpoint = api_endpoint or os.getenv("GOOGLE_API_ENDPOINT")

        if project_id is None:
            raise ValueError("Google Project ID cannot be 'None'.")
        try:
            aiplatform.init(
                project=project_id, location=location, api_endpoint=api_endpoint
            )
            self.client = TextEmbeddingModel.from_pretrained(self.name)
        except Exception as e:
            raise ValueError(
                f"Google AI Platform client failed to initialize. Error: {e}"
            ) from e

    def __call__(self, docs: List[str]) -> List[List[float]]:
        """Generates embeddings for the given documents.

        Args:
            docs: A list of strings representing the documents to embed.

        Returns:
            A list of lists, where each inner list contains the embedding values for a document.

        Raises:
            ValueError: If the Google AI Platform client is not initialized or if the API call fails.
        """
        if self.client is None:
            raise ValueError("Google AI Platform client is not initialized.")
        try:
            embeddings = self.client.get_embeddings(docs)
            return [embedding.values for embedding in embeddings]
        except Exception as e:
            raise ValueError(f"Google AI Platform API call failed. Error: {e}") from e
