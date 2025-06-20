import json
import tempfile

import numpy as np
import pytest

from semantic_router.tokenizers import (
    BaseTokenizer,
    PretrainedTokenizer,
)


class TestBaseTokenizer:
    def test_abstract_methods(self):
        class ConcreteTokenizer(BaseTokenizer):
            pass

        tokenizer = ConcreteTokenizer()
        with pytest.raises(NotImplementedError):
            _ = tokenizer.vocab_size
        with pytest.raises(NotImplementedError):
            _ = tokenizer.config
        with pytest.raises(NotImplementedError):
            tokenizer.tokenize("test")

    def test_save_load(self):
        class ConcreteTokenizer(BaseTokenizer):
            def __init__(self, test_param) -> None:
                self.test_param = test_param
                super().__init__()

            @property
            def vocab_size(self):
                return 100

            @property
            def config(self):
                return {"test_param": self.test_param}

            def tokenize(self, texts, pad=True):
                pass

        with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
            tokenizer = ConcreteTokenizer(test_param="value")
            tokenizer.save(tmp.name)

            loaded = ConcreteTokenizer.load(tmp.name)
            assert isinstance(loaded, ConcreteTokenizer)
            with open(tmp.name) as f:
                saved_config = json.load(f)
            assert saved_config == {"test_param": "value"}


class TestPretrainedTokenizer:
    @pytest.fixture
    def tokenizer(self):
        return PretrainedTokenizer("google-bert/bert-base-uncased")

    def test_initialization(self, tokenizer):
        assert tokenizer.model_ident == "google-bert/bert-base-uncased"
        assert tokenizer.add_special_tokens is False
        assert tokenizer.pad is True

    def test_vocab_size(self, tokenizer):
        assert isinstance(tokenizer.vocab_size, int)
        assert tokenizer.vocab_size > 0

    def test_config(self, tokenizer):
        config = tokenizer.config
        assert isinstance(config, dict)
        assert "model_ident" in config
        assert "add_special_tokens" in config
        assert "pad" in config

    def test_tokenize_single_text(self, tokenizer):
        text = "Hello world"
        tokens = tokenizer.tokenize(text)
        assert isinstance(tokens, np.ndarray)
        assert tokens.ndim == 2
        assert tokens.shape[0] == 1  # One sequence
        assert tokens.shape[1] > 0  # At least one token

    def test_tokenize_multiple_texts(self, tokenizer):
        texts = ["Hello world", "Testing tokenization"]
        tokens = tokenizer.tokenize(texts)
        assert isinstance(tokens, np.ndarray)
        assert tokens.ndim == 2
        assert tokens.shape[0] == 2  # Two sequences

    def test_save_load_cycle(self, tokenizer):
        with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
            tokenizer.save(tmp.name)
            loaded = PretrainedTokenizer.load(tmp.name)

            assert isinstance(loaded, PretrainedTokenizer)
            assert loaded.model_ident == tokenizer.model_ident
            assert loaded.add_special_tokens == tokenizer.add_special_tokens
            assert loaded.pad == tokenizer.pad
