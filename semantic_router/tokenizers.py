import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np

# Tokenizers imports moved to be optional


class BaseTokenizer:
    """Abstract Tokenizer class"""

    @property
    def vocab_size(self) -> int:
        """Returns the vocabulary size of the tokenizer

        :return: Vocabulary size of tokenizer
        :rtype: int
        """
        raise NotImplementedError

    @property
    def config(self) -> dict:
        """The tokenizer config

        :return: dictionary of tokenizer config
        :rtype: dict
        """
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        """Saves the configuration of the tokenizer

        Saves these files:
        - tokenizer.json: saved configuration of the tokenizer

        :param path: Path to save the tokenizer to
        :type path: str, :class:`pathlib.Path`
        """
        if isinstance(path, str):
            path = Path(path)

        with open(path, "w") as fp:
            json.dump(self.config, fp)

    @classmethod
    def load(cls, path: str | Path) -> "BaseTokenizer":
        """Returns a :class:`bm25_engine.tokenizer.BaseTokenizer` object from saved configuration

        Requires these files:
        - tokenizer.json: saved configuration of the tokenizer

        :param path: Path to load the tokenizer from
        :type path: str, :class:`pathlib.Path`
        :returns: Configured BaseTokenizer
        :rtype: BaseTokenizer
        """
        if isinstance(path, str):
            path = Path(path)

        with open(path, encoding="utf-8") as fp:
            config = json.load(fp)
        return cls(**config)

    def tokenize(self, texts: str | list[str], pad: bool = True) -> np.ndarray:
        raise NotImplementedError


class HashTokenizer(BaseTokenizer):
    def __init__(self) -> None:
        super().__init__()


class PretrainedTokenizer(BaseTokenizer):
    """Wrapper for HuggingFace tokenizers, representing a pretrained tokenizer (i.e. bert-base-uncased).
    Extends the :class:`semantic_router.tokenizers.BaseTokenizer` class.

    :param tokenizer: Binding for HuggingFace Rust tokenizers
    :type tokenizer: class:`tokenizers.Tokenizer`
    :param add_special_tokens: Whether to accept special tokens from the tokenizer (i.e. `[PAD]`)
    :type add_special_tokens: bool
    :param pad: Whether to pad the input to a consistent length (using `[PAD]` tokens)
    :type pad: bool
    :param model_ident: HuggingFace ID of the model (i.e. `bert-base-uncased`)
    :type model_ident: str
    """

    add_special_tokens: bool
    pad: bool
    model_ident: str

    def __init__(
        self,
        model_ident: str,
        custom_normalizer: Any = None,
        add_special_tokens: bool = False,
        pad: bool = True,
    ) -> None:
        """Constructor method"""
        # Check if tokenizers is available
        if importlib.util.find_spec("tokenizers") is None:
            raise ImportError(
                "The 'tokenizers' package is required for PretrainedTokenizer but not installed. "
                "Please install it with `pip install tokenizers`."
            )

        # Import tokenizers only when needed
        from tokenizers import Tokenizer

        super().__init__()
        self.add_special_tokens = add_special_tokens
        self.model_ident = model_ident
        self.tokenizer = Tokenizer.from_pretrained(model_ident)
        self.pad = pad
        if custom_normalizer:
            self.tokenizer.normalizer = custom_normalizer  # type: ignore
        if pad:
            self.tokenizer.enable_padding(direction="right", pad_id=0)

    @property
    def vocab_size(self):
        """Returns the vocabulary size of the tokenizer

        :return: Vocabulary size of tokenizer
        :rtype: int
        """
        return self.tokenizer.get_vocab_size()

    @property
    def config(self) -> dict:
        """The tokenizer config

        :return: dictionary of tokenizer config
        :rtype: dict
        """
        return {
            "model_ident": self.model_ident,
            "add_special_tokens": self.add_special_tokens,
            "pad": self.pad,
        }

    def tokenize(self, texts: str | list[str], pad: bool = True) -> np.ndarray:
        """Tokenizes a string or list of strings into a 2D :class:`numpy.ndarray` of token ids

        :param texts: Texts to be tokenized
        :type texts: str, list
        :param pad: unused here (configured in the constructor)
        :type pad: bool
        :return: 2D numpy array representing token ids
        :rtype: class:`numpy.ndarray`
        """
        if isinstance(texts, str):
            texts = [texts]

        encodings = self.tokenizer.encode_batch_fast(
            texts, add_special_tokens=self.add_special_tokens
        )

        return np.array([e.ids for e in encodings])


class TokenizerFactory:
    """Tokenizer factory class"""

    @staticmethod
    def get(type_: str, **tokenizer_kwargs) -> BaseTokenizer:
        r"""Get a configured :class:`bm25_engine.tokenizer.BaseTokenizer`

        :param type_: Tokenizer type to instantiate
        :type type_: str
        :param \**kwargs: kwargs to be passed to Tokenizer constructor
        :return: Tokenizer
        :rtype: `bm25_engine.tokenizer.BaseTokenizer`
        """
        match type_:
            case "pretrained":
                return PretrainedTokenizer(**tokenizer_kwargs)
            case _:
                return PretrainedTokenizer(**tokenizer_kwargs)
