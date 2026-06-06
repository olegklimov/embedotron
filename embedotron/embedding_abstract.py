import numpy as np
from typing import List


class EmbeddingAbstract:
    def __init__(self):
        self.D = -1                # vector length, set by the subclass

    # Returns one np.float32 vector per input text, in input order.
    async def ask_for_embedding(
        self,
        texts: List[str],
    ) -> List[np.ndarray]:
        raise NotImplementedError()

    def estimate_tokens(self, s: str) -> int:
        raise NotImplementedError()

    def max_tokens(self) -> int:
        raise NotImplementedError()

    def throwaway_score(self) -> float:
        raise NotImplementedError()
