import os
from typing import Any, List, Optional

import numpy as np

from semantic_router.encoders import DenseEncoder
from semantic_router.utils.defaults import EncoderDefault


class TritonEncoder(DenseEncoder):
    """TritonEncoder class for generating embeddings using TritonInference server.

    https://triton-inference-server.github.io/pytriton/latest/

    Example usage:

    ```python
    from semantic_router.encoders.triton import TritonEncoder

    # Assumes Triton is running with model "your_model_name"
    # accessible via gRPC on port 8001
    encoder = TritonEncoder(
        name="your_model_name",
        base_url="grpc://localhost:8001"
    )
    embeddings = encoder(["document1", "document2"])
    ```

    Attributes:
        client: An instance of the PyTriton ModelClient.
        type: The type of the encoder, which is "triton".
        input_name: The name of the input tensor expected by the Triton model.
        output_name: The name of the output tensor produced by the Triton model.
    """

    client: Optional[Any] = None
    type: str = "triton"
    input_name: str
    output_name: str

    def __init__(
        self,
        name: Optional[str] = None,
        score_threshold: float = 0.5,
        base_url: str | None = None,
        input_name: str = "text_snippet",
        output_name: str = "embedding",
    ):
        """Initializes the TritonEncoder.

        :param name: The name of the pre-trained model to use for embedding,
            as registered in Triton. If not provided, the default model
            specified in EncoderDefault will be used.
        :type name: str, optional
        :param score_threshold: The threshold for similarity scores.
        :type score_threshold: float
        :param base_url: The API endpoint for your TritonServer (e.g.,
            "http://localhost:8000" or "grpc://localhost:8001").
            If not provided, it will be retrieved from the `TRITON_BASE_URL`
            environment variable, defaulting to "http://localhost:8000".
        :type base_url: str, optional
        :param input_name: The name of the input tensor in your Triton model's
            configuration (e.g., "text_snippet", "text", "INPUT").
            Defaults to "text_snippet".
        :type input_name: str
        :param output_name: The name of the output tensor in your Triton model's
            configuration (e.g., "embedding", "OUTPUT").
            Defaults to "embedding".
        :type output_name: str

        :raise ValueError: If the hosted base url is not provided properly or
            if the Triton client fails to initialize.
        """
        if name is None:
            name = EncoderDefault.TRITON.value["embedding_model"]

        super().__init__(
            name=name,
            score_threshold=score_threshold,
            input_name=input_name,
            output_name=output_name,
        )
        if base_url is None:
            base_url = os.getenv("TRITON_BASE_URL", "http://localhost:8000")

        self.input_name = input_name
        self.output_name = output_name
        self.client = self._initialize_client(base_url=base_url)

    def _initialize_client(self, base_url: str):
        """Initializes the TRITON client.

        :param base_url: Hosted URL of Triton (e.g., "http://..." or "grpc://...").
        :return: An instance of the ModelClient.
        :rtype: ModelClient
        :raise ImportError: If the required 'pytriton-client' library is not installed.
        :raise ValueError: If the Triton client fails to initialize or connect
            to the model.
        """
        try:
            from pytriton.client import ModelClient
        except ImportError:
            raise ImportError(
                "The 'pytriton-client' package is not installed. "
                "Install it with: pip install 'semantic-router[triton]'"
                " or 'pip install pytriton-client'"
            )

        try:
            # We use the synchronous ModelClient to match the base class interface
            client = ModelClient(url=base_url, model_name=self.name)
            # Wait for the model to be ready to avoid errors on the first call
            client.wait_for_model(timeout_s=10.0)
            return client
        except Exception as e:
            raise ValueError(
                f"Failed to initialize Triton client or connect to model "
                f"'{self.name}' at '{base_url}'. Error: {e}"
            ) from e

    def __call__(self, docs: List[str]) -> List[List[float]]:
        """Generates embeddings for the given documents.

        :param docs: A list of strings representing the documents to embed.
        :type docs: List[str]
        :return: A list of lists, where each inner list contains the embedding
            values for a document.
        :rtype: List[List[float]]
        :raise ValueError: If the Triton client is not initialized or if the
            API call fails.
        """
        if self.client is None:
            raise ValueError("Triton client is not initialized.")
        self.client
        try:
            docs_np = np.array([doc.encode("utf-8") for doc in docs], dtype=np.bytes_)

            if self.client.is_batching_supported:
                inputs = {self.input_name: docs_np}

                results_dict = self.client.infer_batch(**inputs)

                embeddings_np = results_dict[self.output_name]
            else:
                for doc in docs:
                    doc_np = np.array(doc.encode("utf-8"), dtype=np.bytes_)
                    input = {self.input_name: doc_np}
                    result = self.client.infer_sample(**input)
                    embeddings_np = result[self.output_name]
            return embeddings_np.tolist()
        except Exception as e:
            raise ValueError(f"Triton API call failed. Error: {e}") from e
