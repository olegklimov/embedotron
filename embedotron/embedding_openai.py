import asyncio
import logging
from typing import List, Optional

import httpx
import numpy as np

from embedotron import embedding_abstract
from embedotron import embedding_vecmath


logger = logging.getLogger("embedding")


# OpenAI /embeddings adapter. The same wire shape is spoken by a large openai-compatible ecosystem --
# text-embedding-inference (TEI), Ollama, vLLM, Jina, xAI -- so this one class covers all of them; they
# differ only in endpoint and model name. Point base_url at the server, leave api_key empty if it needs
# none.
#
# OpenAI caps a request at ~300k tokens across all inputs. We greedily pack inputs into batches under
# 95% of that, one POST per batch, and stitch results back in input order.

MAX_TOKENS_PER_REQUEST = 300000
RETRY_DELAYS = [1.0, 2.0, 4.0, 8.0]

openai_embedding_models = {
    "text-embedding-3-small": {"dimensions": 1536, "max_tokens": 8191},
    "text-embedding-3-large": {"dimensions": 3072, "max_tokens": 8191},
    "text-embedding-ada-002": {"dimensions": 1536, "max_tokens": 8191},
}


class EmbeddingOpenAI(embedding_abstract.EmbeddingAbstract):
    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        dimensions: Optional[int] = None,     # ask openai-3 models to shorten output vectors
        normalize: bool = False,
        # For openai-compatible servers not in the table above, pass these instead of relying on the model name.
        model_dimensions: Optional[int] = None,
        model_max_tokens: Optional[int] = None,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.want_dimensions = dimensions
        self.want_normalize = normalize
        known = openai_embedding_models.get(model)
        if known:
            self.D = dimensions or known["dimensions"]
            self._max_tokens = known["max_tokens"]
        else:
            if model_dimensions is None:
                raise ValueError("unknown model %r, pass model_dimensions/model_max_tokens for a custom server" % model)
            self.D = dimensions or model_dimensions
            self._max_tokens = model_max_tokens or 8191

    async def ask_for_embedding(
        self,
        texts: List[str],
    ) -> List[np.ndarray]:
        if not texts:
            return []
        texts = [_truncate_to_tokens(t, self._max_tokens) for t in texts]

        vectors: List[Optional[np.ndarray]] = [None] * len(texts)
        base = 0
        async with httpx.AsyncClient(timeout=60.0) as client:
            for batch in self._split_into_batches(texts):
                body = {"input": batch, "model": self.model}
                if self.want_dimensions is not None:
                    body["dimensions"] = self.want_dimensions
                data = await self._post_with_retry(client, body)
                for item in data["data"]:
                    v = np.array(item["embedding"], dtype=np.float32)
                    vectors[base + item["index"]] = embedding_vecmath.normalize(v) if self.want_normalize else v
                base += len(batch)

        missing = [i for i, v in enumerate(vectors) if v is None]
        if missing:
            raise RuntimeError("openai embeddings returned no vector for inputs %r" % missing)
        return vectors

    def _split_into_batches(self, texts):
        batches, cur, cur_tokens = [], [], 0
        for t in texts:
            tk = self.estimate_tokens(t)
            if cur and cur_tokens + tk > MAX_TOKENS_PER_REQUEST * 0.95:
                batches.append(cur)
                cur, cur_tokens = [t], tk
            else:
                cur.append(t)
                cur_tokens += tk
        if cur:
            batches.append(cur)
        return batches

    async def _post_with_retry(self, client, body):
        url = self.base_url.rstrip("/") + "/embeddings"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        attempt = 0
        while True:
            try:
                response = await client.post(url, headers=headers, json=body)
            except httpx.TransportError as e:
                if attempt < len(RETRY_DELAYS):
                    d = RETRY_DELAYS[attempt]
                    logger.warning("embeddings transport error %s, retry %d after %.1fs", type(e).__name__, attempt + 1, d)
                    await asyncio.sleep(d)
                    attempt += 1
                    continue
                raise
            if response.status_code == 200:
                return response.json()
            # 429 and 5xx are transient; honor Retry-After on 429 when present.
            retryable = response.status_code == 429 or response.status_code >= 500
            if retryable and attempt < len(RETRY_DELAYS):
                d = RETRY_DELAYS[attempt]
                ra = response.headers.get("retry-after")
                if ra:
                    try:
                        d = float(ra)
                    except ValueError:
                        pass
                logger.warning("embeddings api %d, retry %d after %.1fs", response.status_code, attempt + 1, d)
                await asyncio.sleep(d)
                attempt += 1
                continue
            raise RuntimeError("openai embeddings api error %d: %s" % (response.status_code, response.text[:500]))

    def estimate_tokens(self, s: str) -> int:
        return _estimate_tokens(s)

    def max_tokens(self) -> int:
        return self._max_tokens

    def throwaway_score(self) -> float:
        return 0.700


# --- token counting: tiktoken when installed (embedotron[tiktoken]), else a cheap chars/4 estimate.
# The per-request token cap is the real guardrail at the API; the estimate only decides batch splits
# and truncation points, so a rough number is fine when tiktoken is absent.

_cl100k = None
_warned_no_tiktoken = False


def _have_tiktoken():
    try:
        import tiktoken  # noqa: F401
        return True
    except ImportError:
        return False


def _encoder():
    global _cl100k
    if _cl100k is None:
        import tiktoken
        _cl100k = tiktoken.get_encoding("cl100k_base")
    return _cl100k


def _estimate_tokens(s: str) -> int:
    global _warned_no_tiktoken
    if not s:
        return 0
    if _have_tiktoken():
        # disallowed_special=() so literal <|...|> strings in user text don't raise.
        return len(_encoder().encode(s, disallowed_special=()))
    if not _warned_no_tiktoken:
        logger.warning("tiktoken not installed, using chars/4 token estimate (pip install embedotron[tiktoken] for exact counts)")
        _warned_no_tiktoken = True
    return len(s) // 4 + 1


def _truncate_to_tokens(s: str, max_tokens: int) -> str:
    if not s:
        return s
    if _have_tiktoken():
        enc = _encoder()
        ids = enc.encode(s, disallowed_special=())
        if len(ids) <= max_tokens:
            return s
        logger.warning("clamping a %d-token input to %d tokens", len(ids), max_tokens)
        return enc.decode(ids[:max_tokens])
    if len(s) // 4 + 1 <= max_tokens:
        return s
    logger.warning("clamping a ~%d-char input to ~%d tokens", len(s), max_tokens)
    return s[: max_tokens * 4]
