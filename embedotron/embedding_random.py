import hashlib
from typing import List, Tuple

import numpy as np

from embedotron import embedding_abstract
from embedotron import embedding_vecmath


# Deterministic random embedder for tests and offline dev: no http, no key, no cost. The vector for a
# given text is a function of its content hash, so the same text always embeds to the same unit vector
# -- cosine(same_text, same_text) == 1.0. Unrelated texts land roughly orthogonal, like real
# embeddings, but carry no semantic meaning.

VECTOR_DIMENSION = 1536


class EmbeddingRandom(embedding_abstract.EmbeddingAbstract):
    def __init__(self, *, dimensions: int = VECTOR_DIMENSION):
        self.D = dimensions
        self._max_tokens = 8191

    async def ask_for_embedding(
        self,
        texts: List[str],
    ) -> Tuple[List[np.ndarray], List[int]]:
        vectors = []
        for t in texts:
            seed = int.from_bytes(hashlib.sha256(t.encode("utf-8")).digest()[:8], "big")
            v = np.random.default_rng(seed).standard_normal(self.D).astype(np.float32)
            vectors.append(embedding_vecmath.normalize(v))
        return vectors, [0] * len(texts)

    def estimate_tokens(self, s: str) -> int:
        return len(s) // 4 + 1

    def max_tokens(self) -> int:
        return self._max_tokens

    def throwaway_score(self) -> float:
        return 0.0
