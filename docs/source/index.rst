Semantic Router documentation
=============================

Semantic Router is a superfast decision-making layer for your LLMs and agents. Rather than waiting for slow LLM generations to make tool-use decisions, we use the magic of semantic vector space to make those decisions — *routing* our requests using *semantic* meaning.

Integrations
------------

The *encoders* of semantic router include easy-to-use integrations with `Cohere <https://github.com/aurelio-labs/semantic-router/blob/main/semantic_router/encoders/cohere.py>`_, `OpenAI <https://github.com/aurelio-labs/semantic-router/blob/main/docs/encoders/openai-embed-3.ipynb>`_, `Hugging Face <https://github.com/aurelio-labs/semantic-router/blob/main/docs/encoders/huggingface.ipynb>`_, `FastEmbed <https://github.com/aurelio-labs/semantic-router/blob/main/docs/encoders/fastembed.ipynb>`_, and `more <https://github.com/aurelio-labs/semantic-router/tree/main/semantic_router/encoders>`_ — we even support `multi-modality <https://github.com/aurelio-labs/semantic-router/blob/main/docs/07-multi-modal.ipynb>`_!.

Our utterance vector space also integrates with `Pinecone <https://github.com/aurelio-labs/semantic-router/blob/main/docs/indexes/pinecone.ipynb>`_ and `Qdrant <https://github.com/aurelio-labs/semantic-router/blob/main/docs/indexes/qdrant.ipynb>`_!

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   api
