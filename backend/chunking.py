from __future__ import annotations

import uuid
from typing import Protocol

from .models import Chunk
from .text_utils import split_paragraphs, split_sentences


class Chunker(Protocol):
    def chunk(
        self, book_id: str, markdown_text: str, window_sizes_sentences: list[int]
    ) -> list[Chunk]: ...


class ParagraphSentenceChunker:
    """Slides fixed-size (in sentences) windows over the book's flat sentence
    stream at 50% stride, never cutting mid-sentence. Windows may span
    paragraph breaks (novel paragraphs vary too much in length to restrict a
    window to one); a blank line is reinserted wherever a window crosses one.
    """

    def chunk(
        self, book_id: str, markdown_text: str, window_sizes_sentences: list[int]
    ) -> list[Chunk]:
        paragraphs = split_paragraphs(markdown_text)

        stream: list[tuple[int, str]] = [
            (para_idx, sentence)
            for para_idx, para in enumerate(paragraphs)
            for sentence in split_sentences(para)
        ]

        chunks: list[Chunk] = []
        seen_text: set[str] = set()

        for window_size in window_sizes_sentences:
            stride = max(1, window_size // 2)
            min_len = max(1, window_size // 2)
            i = 0
            while i < len(stream):
                window = stream[i : i + window_size]
                if len(window) >= min_len:
                    text = self._reconstruct(window)
                    if text not in seen_text:
                        seen_text.add(text)
                        chunks.append(
                            Chunk(
                                id=str(uuid.uuid4()),
                                book_id=book_id,
                                text=text,
                                start_sentence_idx=i,
                                end_sentence_idx=i + len(window) - 1,
                                window_label=f"{window_size}s",
                            )
                        )
                i += stride

        return chunks

    @staticmethod
    def _reconstruct(window: list[tuple[int, str]]) -> str:
        pieces: list[str] = []
        prev_para: int | None = None
        for para_idx, sentence in window:
            if prev_para is None:
                pieces.append(sentence)
            elif para_idx != prev_para:
                pieces.append("\n\n" + sentence)
            else:
                pieces.append(" " + sentence)
            prev_para = para_idx
        return "".join(pieces)
