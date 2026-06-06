# Embedotron

Provider-agnostic text embedding layer, a counterpart to [hallucitron](https://github.com/olegklimov/hallucitron) which does chat/completions.

Here is essentially the whole idea:

```python
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

```

Dependencies are `httpx`, `pyyaml`, and `numpy`. `tiktoken` is optional and only sharpens token
counts — without it, batching/truncation fall back to a cheap chars/4 estimate.


## Providers

* `EmbeddingOpenAI` — the OpenAI `/embeddings` API, which also covers the openai-compatible ecosystem (text-embedding-inference, Ollama, vLLM, Jina, xAI); point `base_url` at the server
* `EmbeddingRandom` — deterministic content-hash vectors for tests and offline dev (no network, no key)


## Example

```python
import asyncio
import embedotron as e

async def main():
    cfg = e.load_default_config(use_env_keys=True)        # keys from OPENAI_API_KEY etc.
    emb = cfg.embedder_for("text-embedding-3-large", normalize=False)
    vecs = await emb.ask_for_embedding([
        "Turning coffee into deliverables since 8:03 AM",
        "午前 8 時 3 分からコーヒーを成果物に変える",       # the same joke in Japanese
        "Reducing risk by increasing paperwork",
    ])
    print("EN coffee <-> JA coffee   ", e.cosine(vecs[0], vecs[1]))
    print("EN coffee <-> EN paperwork", e.cosine(vecs[0], vecs[2]))
    # store in postgres (column type `vector`):
    #   await conn.execute("INSERT INTO t (emb) VALUES ($1::vector)", e.to_pgvector(vecs[0]))

asyncio.run(main())
```


## Testing

`embedding_test.py` hits the real OpenAI API (a fraction of a cent) and is skipped when no key is
present:

```sh
cp .test_api_keys.example .test_api_keys   # add your key for the live test
PYTHONPATH=. python embedotron/embedding_test.py
```

`.test_api_keys` holds live secrets and is git-ignored. Never commit it. If one leaks, rotate it.


## License

MIT — see [LICENSE](LICENSE).
