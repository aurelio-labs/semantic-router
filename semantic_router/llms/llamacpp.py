from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

from pydantic.v1 import PrivateAttr

from semantic_router.llms.base import BaseLLM
from semantic_router.schema import Message
from semantic_router.utils.logger import logger


class LlamaCppLLM(BaseLLM):
    llm: Any
    temperature: float
    max_tokens: Optional[int] = 200
    grammar: Optional[Any] = None
    _llama_cpp: Any = PrivateAttr()

    def __init__(
        self,
        llm: Any,
        name: str = "llama.cpp",
        temperature: float = 0.2,
        max_tokens: Optional[int] = 200,
        grammar: Optional[Any] = None,
    ):
        super().__init__(
            name=name,
            llm=llm,
            temperature=temperature,
            max_tokens=max_tokens,
            grammar=grammar,
        )

        try:
            import llama_cpp
        except ImportError:
            raise ImportError(
                "Please install LlamaCPP to use Llama CPP llm. "
                "You can install it with: "
                "`pip install 'semantic-router[local]'`"
            )
        self._llama_cpp = llama_cpp
        self.llm = llm
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.grammar = grammar

    def __call__(
        self,
        messages: list[Message],
    ) -> str:
        try:
            completion = self.llm.create_chat_completion(
                messages=[m.to_llamacpp() for m in messages],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                grammar=self.grammar,
                stream=False,
            )
            assert isinstance(completion, dict)  # keep mypy happy
            output = completion["choices"][0]["message"]["content"]

            if not output:
                raise Exception("No output generated")
            return output
        except Exception as e:
            logger.error(f"LLM error: {e}")
            raise

    @contextmanager
    def _grammar(self):
        grammar_path = Path(__file__).parent.joinpath("grammars", "json.gbnf")
        assert grammar_path.exists(), f"{grammar_path}\ndoes not exist"
        try:
            self.grammar = self._llama_cpp.LlamaGrammar.from_file(grammar_path)
            yield
        finally:
            self.grammar = None

    def extract_function_inputs(
        self, query: str, function_schema: dict[str, Any]
    ) -> dict:
        with self._grammar():
            return super().extract_function_inputs(
                query=query, function_schema=function_schema
            )
