import re
from typing import Any

from colorama import Fore, Style

from semantic_router.encoders import BaseEncoder
from semantic_router.splitters import RollingWindowSplitter


class UnstructuredSemanticSplitter:
    def __init__(
        self,
        encoder: BaseEncoder,
        window_size: int,
        min_split_tokens: int,
        max_split_tokens: int,
    ):
        self.splitter = RollingWindowSplitter(
            encoder=encoder,
            window_size=window_size,
            min_split_tokens=min_split_tokens,
            max_split_tokens=max_split_tokens,
        )

    def is_valid_title(self, title: str) -> bool:
        # Rule 1: Title starts with a lowercase letter
        if re.match(r"^[a-z]", title):
            return False
        # Rule 2: Title has a special character (excluding :, -, and .)
        if re.search(r"[^\w\s:\-\.]", title):
            return False
        # Rule 3: Title ends with a dot
        if title.endswith("."):
            return False
        return True

    def _group_elements_by_title(self, elements: list[dict[str, Any]]) -> dict:
        grouped_elements = {}
        current_title = "Untitled"  # Default title for initial text without a title

        for element in elements:
            if element.get("type") == "Title":
                potential_title = element.get("text", "Untitled")
                if self.is_valid_title(potential_title):
                    print(f"{Fore.GREEN}{potential_title}: True{Style.RESET_ALL}")
                    current_title = potential_title
                else:
                    print(f"{Fore.RED}{potential_title}: False{Style.RESET_ALL}")
                    continue
            else:
                if current_title not in grouped_elements:
                    grouped_elements[current_title] = []
                else:
                    grouped_elements[current_title].append(element)
        return grouped_elements

    async def split_grouped_elements(
        self, elements: list[dict[str, Any]], splitter: RollingWindowSplitter
    ) -> list[dict[str, Any]]:
        grouped_elements = self._group_elements_by_title(elements)
        chunks_with_title = []

        def _append_chunks(*, title: str, content: str, index: int, metadata: dict):
            chunks_with_title.append(
                {
                    "title": title,
                    "content": content,
                    "chunk_index": index,
                    "metadata": metadata,
                }
            )

        for index, (title, elements) in enumerate(grouped_elements.items()):
            if not elements:
                continue
            section_metadata = elements[0].get(
                "metadata", {}
            )  # Took first element's data
            accumulated_element_texts: list[str] = []
            chunks: list[dict[str, Any]] = []

            for element in elements:
                if not element.get("text"):
                    continue
                if element.get("type") == "Table":
                    # Process accumulated text before the table
                    if accumulated_element_texts:
                        splits = splitter(accumulated_element_texts)
                        for split in splits:
                            _append_chunks(
                                title=title,
                                content=split.content,
                                index=index,
                                metadata=section_metadata,
                            )
                        # TODO: reset after PageBreak also
                        accumulated_element_texts = (
                            []
                        )  # Start new accumulation after table

                    # Add table as a separate chunk
                    _append_chunks(
                        title=title,
                        content=element.get("metadata", {}).get(
                            "text_as_html", "No text"
                        ),
                        index=index,
                        metadata=element.get("metadata", {}),
                    )
                else:
                    accumulated_element_texts.append(element.get("text", "No text"))

            # Process any remaining accumulated text after the last table
            # or if no table was encountered

            if accumulated_element_texts:
                splits = splitter(accumulated_element_texts)
                for split in splits:
                    _append_chunks(
                        title=title,
                        content=split.content,
                        index=index,
                        metadata=section_metadata,
                    )
            if chunks:
                chunks_with_title.extend(chunks)
        return chunks_with_title

    async def __call__(self, elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return await self.split_grouped_elements(elements, self.splitter)
