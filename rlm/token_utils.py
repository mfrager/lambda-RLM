#!/usr/bin/env python3

from typing import List, Tuple
import tiktoken
import nltk

nltk.download("punkt", quiet=True)
from nltk.tokenize import sent_tokenize

class TokenChunker:
    def __init__(self, model: str = "gpt-4o"):
        self.encoding = tiktoken.encoding_for_model(model)

    def _token_len(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def _encode(self, text: str):
        return self.encoding.encode(text)

    def _decode(self, tokens):
        return self.encoding.decode(tokens)

    def split_sentences(self, text: str) -> List[str]:
        return sent_tokenize(text)

    def chunk(
        self,
        text: str,
        max_tokens: int,
        max_chunks: int
    ) -> List[str]:

        sentences = self.split_sentences(text)
        token_sizes = [self._token_len(s) for s in sentences]

        chunks = []
        current_chunk = []
        current_size = 0

        i = 0
        n = len(sentences)

        while i < n:

            # If we've reached max_chunks - 1, dump everything remaining
            if len(chunks) == max_chunks - 1:
                remaining = " ".join(sentences[i:])
                chunks.append(remaining)
                return chunks

            sent = sentences[i]
            size = token_sizes[i]

            # Oversized sentence handling
            if size > max_tokens:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_size = 0

                tokens = self._encode(sent)

                for j in range(0, len(tokens), max_tokens):
                    piece = self._decode(tokens[j:j + max_tokens])

                    # respect max_chunks cap
                    if len(chunks) == max_chunks - 1:
                        chunks.append(piece + " " + " ".join(sentences[i+1:]))
                        return chunks

                    chunks.append(piece)

                i += 1
                continue

            # normal packing
            if current_size + size <= max_tokens:
                current_chunk.append(sent)
                current_size += size
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sent]
                current_size = size

            i += 1

        # flush remaining
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

_chunker = TokenChunker()

def count_tokens(text: str) -> int:
    return _chunker._token_len(text)

