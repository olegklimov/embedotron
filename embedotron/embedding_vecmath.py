import numpy as np
from typing import List


# Small helpers around the np.float32 vectors the embedders return. Every caller of an embedding model
# ends up writing the same three things: L2-normalize, cosine, and serialize-to-pgvector. The last was
# open-coded as '"[" + ",".join(...) + "]"' all over the place; having it here kills that copy-paste.


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def to_pgvector(v: np.ndarray) -> str:
    # pgvector wants a literal like "[0.1,0.2,0.3]"; pass the result to a $N::vector cast in SQL.
    # str() on a float32 scalar gives the shortest decimal that round-trips back to the same float32,
    # so the literal stays compact (0.1, not 0.10000000149...) without losing precision.
    v = np.asarray(v, dtype=np.float32)
    return "[" + ",".join(str(x) for x in v) + "]"


def from_pgvector(s: str) -> np.ndarray:
    return np.array([float(x) for x in s.strip().lstrip("[").rstrip("]").split(",") if x], dtype=np.float32)
