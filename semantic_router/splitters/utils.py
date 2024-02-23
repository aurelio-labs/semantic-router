from typing import List

import regex
import tiktoken
from colorama import Fore, Style

from semantic_router.schema import DocumentSplit


def split_to_sentences(text: str) -> list[str]:
    """
    Enhanced regex pattern to split a given text into sentences more accurately.

    The enhanced regex pattern includes handling for:
    - Direct speech and quotations.
    - Abbreviations, initials, and acronyms.
    - Decimal numbers and dates.
    - Ellipses and other punctuation marks used in informal text.
    - Removing control characters and format characters.

    Args:
        text (str): The text to split into sentences.

    Returns:
        list: A list of sentences extracted from the text.
    """
    regex_pattern = r"""
        # Negative lookbehind for word boundary, word char, dot, word char
        (?<!\b\w\.\w.)
        # Negative lookbehind for single uppercase initials like "A."
        (?<!\b[A-Z][a-z]\.)
        # Negative lookbehind for abbreviations like "U.S."
        (?<!\b[A-Z]\.)
        # Negative lookbehind for abbreviations with uppercase letters and dots
        (?<!\b\p{Lu}\.\p{Lu}.)
        # Negative lookbehind for numbers, to avoid splitting decimals
        (?<!\b\p{N}\.)
        # Positive lookbehind for punctuation followed by whitespace
        (?<=\.|\?|!|:|\.\.\.)\s+
        # Positive lookahead for uppercase letter or opening quote at word boundary
        (?="?(?=[A-Z])|"\b)
        # OR
        |
        # Splits after punctuation that follows closing punctuation, followed by
        # whitespace
        (?<=[\"\'\]\)\}][\.!?])\s+(?=[\"\'\(A-Z])
        # OR
        |
        # Splits after punctuation if not preceded by a period
        (?<=[^\.][\.!?])\s+(?=[A-Z])
        # OR
        |
        # Handles splitting after ellipses
        (?<=\.\.\.)\s+(?=[A-Z])
        # OR
        |
        # Matches and removes control characters and format characters
        [\p{Cc}\p{Cf}]+
    """
    sentences = regex.split(regex_pattern, text, flags=regex.VERBOSE)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return sentences


def tiktoken_length(text: str) -> int:
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)


def plot_splits(document_splits: List[DocumentSplit]) -> None:
    colors = [Fore.RED, Fore.GREEN, Fore.BLUE, Fore.MAGENTA]
    for i, split in enumerate(document_splits):
        color = colors[i % len(colors)]
        colored_content = f"{color}{split.content}{Style.RESET_ALL}"
        if split.is_triggered:
            triggered = f"{split.triggered_score:.2f}"
        elif i == len(document_splits) - 1:
            triggered = "final split"
        else:
            triggered = "token limit"
        print(
            f"Split {i + 1}, "
            f"tokens {split.token_count}, "
            f"triggered by: {triggered}"
        )
        print(colored_content)
        print("-" * 88)
        print("\n")
