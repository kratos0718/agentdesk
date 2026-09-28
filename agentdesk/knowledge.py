"""Semantic search over the policy documents.

Documents are split into paragraph chunks, embedded as TF-IDF vectors (word and
character n-grams) and searched by cosine similarity. It needs no model download
and no external service. The `VectorIndex` interface is small on purpose so it can
be swapped for Chroma, FAISS or pgvector with dense embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import settings
from .guardrails import detect_injection


@dataclass
class Chunk:
    source: str
    text: str


@dataclass
class Hit:
    source: str
    text: str
    score: float


def load_chunks(folder: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(folder.glob("*.md")):
        for para in path.read_text(encoding="utf-8").split("\n\n"):
            para = para.strip()
            if para and not para.startswith("#"):
                chunks.append(Chunk(source=path.name, text=para))
    return chunks


class VectorIndex:
    def __init__(self, chunks: list[Chunk]):
        # Chunks that carry instructions aimed at the model are quarantined at
        # indexing time, so a poisoned document can never reach the prompt.
        self.quarantined = [c for c in chunks if detect_injection(c.text)]
        self.chunks = [c for c in chunks if not detect_injection(c.text)]
        texts = [c.text for c in self.chunks]
        self.word_vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.matrix = self._normalise(hstack([self.word_vec.fit_transform(texts),
                                              self.char_vec.fit_transform(texts)]).tocsr())

    @staticmethod
    def _normalise(m):
        norms = np.sqrt(m.multiply(m).sum(axis=1)).A1
        norms[norms == 0] = 1.0
        return m.multiply(1 / norms[:, None]).tocsr()

    def search(self, query: str, k: int = 3, min_score: float = 0.08) -> list[Hit]:
        q = self._normalise(hstack([self.word_vec.transform([query]), self.char_vec.transform([query])]).tocsr())
        scores = (self.matrix @ q.T).toarray().ravel()
        order = np.argsort(-scores)[:k]
        return [Hit(self.chunks[i].source, self.chunks[i].text, round(float(scores[i]), 3))
                for i in order if scores[i] >= min_score]


_index: VectorIndex | None = None


def get_index() -> VectorIndex:
    global _index
    if _index is None:
        _index = VectorIndex(load_chunks(settings.knowledge_dir))
    return _index
