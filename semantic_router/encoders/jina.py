"""This file contains the JinaEncoder class which is used to encode text using Jina"""

from semantic_router.encoders.litellm import LiteLLMEncoder
from semantic_router.utils.defaults import EncoderDefault


class JinaEncoder(LiteLLMEncoder):
    """Class to encode text using Jina. Requires a Jina API key from
    https://jina.ai/api-keys/"""

    type: str = "jina"

    def __init__(
        self,
        name: str | None = None,
        api_key: str | None = None,
        score_threshold: float = 0.4,
    ):
        """Initialize the JinaEncoder.

        :param name: The name of the embedding model to use such as "jina-embeddings-v3".
        :param jina_api_key: The Jina API key.
        :type jina_api_key: str
        """

        if name is None:
            name = f"jina_ai/{EncoderDefault.JINA.value['embedding_model']}"
        elif not name.startswith("jina_ai/"):
            name = f"jina_ai/{name}"
        super().__init__(
            name=name,
            score_threshold=score_threshold,
            api_key=api_key,
        )
