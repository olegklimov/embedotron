import numpy as np
from typing import List, Tuple


class EmbeddingAbstract:
    def __init__(self):
        self.D = -1                # vector length, set by the subclass

    # Returns one np.float32 vector per input text (in input order), plus one cost in integer coins
    # per input (1 coin = $1e-6). Costs are parallel to vectors so a caller metering per item can just
    # zip them; sum() the coins list for the call total.
    async def ask_for_embedding(
        self,
        texts: List[str],
    ) -> Tuple[List[np.ndarray], List[int]]:
        raise NotImplementedError()

    def estimate_tokens(self, s: str) -> int:
        raise NotImplementedError()

    def max_tokens(self) -> int:
        raise NotImplementedError()

    def throwaway_score(self) -> float:
        raise NotImplementedError()
