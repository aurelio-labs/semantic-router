"""Tests for litellm being an optional dependency.

This module must NOT ``importorskip`` litellm: its whole purpose is to cover the
behaviour when litellm is absent.
"""

import sys

import pytest


@pytest.fixture
def no_litellm(monkeypatch):
    """Make ``import litellm`` fail, as it would without the extra installed.

    A ``None`` entry in ``sys.modules`` makes the import statement raise ImportError,
    which simulates the missing package without touching the environment.
    """
    monkeypatch.setitem(sys.modules, "litellm", None)


class TestImportsWithoutLitellm:
    def test_semantic_router_imports(self):
        """The package must import without the litellm extra."""
        import semantic_router

        assert semantic_router is not None

    def test_encoder_class_is_importable(self):
        """LiteLLMEncoder stays exported, so `__all__` and AutoEncoder keep working."""
        from semantic_router.encoders import LiteLLMEncoder

        assert LiteLLMEncoder is not None

    def test_encoder_module_has_no_module_level_litellm_import(self, no_litellm):
        """Re-importing the encoder module must not need litellm."""
        monkey = sys.modules.pop("semantic_router.encoders.litellm", None)
        try:
            import importlib

            module = importlib.import_module("semantic_router.encoders.litellm")
            assert module.LiteLLMEncoder is not None
        finally:
            if monkey is not None:
                sys.modules["semantic_router.encoders.litellm"] = monkey

    def test_provider_encoders_are_importable(self):
        """The provider encoders must not depend on the litellm extra."""
        from semantic_router.encoders import (
            CohereEncoder,
            JinaEncoder,
            MistralEncoder,
            NimEncoder,
            VoyageEncoder,
        )

        assert all(
            enc is not None
            for enc in (
                CohereEncoder,
                JinaEncoder,
                MistralEncoder,
                NimEncoder,
                VoyageEncoder,
            )
        )


class TestHelpfulImportError:
    def test_init_raises_import_error(self, no_litellm, monkeypatch):
        from semantic_router.encoders import LiteLLMEncoder

        monkeypatch.setenv("OPENAI_API_KEY", "test_api_key")
        with pytest.raises(ImportError, match=r"semantic-router\[litellm\]"):
            LiteLLMEncoder("openai/text-embedding-3-small")

    def test_error_names_the_encoder(self, no_litellm, monkeypatch):
        from semantic_router.encoders import LiteLLMEncoder

        monkeypatch.setenv("OPENAI_API_KEY", "test_api_key")
        with pytest.raises(ImportError, match="LiteLLMEncoder"):
            LiteLLMEncoder("openai/text-embedding-3-small")

    def test_litellm_to_list_raises_import_error(self, no_litellm):
        from semantic_router.encoders.litellm import litellm_to_list

        with pytest.raises(ImportError, match=r"semantic-router\[litellm\]"):
            litellm_to_list(None)  # type: ignore[arg-type]

    def test_auto_encoder_raises_import_error(self, no_litellm, monkeypatch):
        """AutoEncoder(type="litellm") must guide the user, not fail obscurely."""
        from semantic_router.encoders import AutoEncoder

        monkeypatch.setenv("OPENAI_API_KEY", "test_api_key")
        with pytest.raises(ImportError, match=r"semantic-router\[litellm\]"):
            AutoEncoder(type="litellm", name="openai/text-embedding-3-small")
