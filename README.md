# Embedotron

A small, provider-agnostic text **embedding** layer — the embeddings counterpart to
[hallucitron](https://github.com/olegklimov/hallucitron) (which does chat/completions).

An abstract `EmbeddingAbstract` with one method that matters, `ask_for_embedding`, and a couple of
provider subclasses behind it. On top of the plain wrapper it adds the things every real caller ends
up writing by hand:

* Token-aware **batching** — pack many inputs under the provider's per-request token cap
* Per-input **truncation** to the model's token limit, so one over-long input doesn't fail the call
* **Retries with backoff** on 429 / 5xx / transport errors (honors `Retry-After`)
* Per-input **cost** in integer "coins" (1 coin = $1e-6), returned alongside the vectors
* numpy-native vectors plus **pgvector** serialization helpers
* **Per-tenant configs** for SaaS setups — each tenant brings its own providers and keys

Providers:

* `EmbeddingOpenAI` — the OpenAI `/embeddings` API, which also covers the openai-compatible ecosystem
  (text-embedding-inference, Ollama, vLLM, Jina, xAI); point `base_url` at the server
* `EmbeddingRandom` — deterministic content-hash vectors for tests and offline dev (no network, no key)

Dependencies are just `httpx`, `pyyaml`, and `numpy`. `tiktoken` is optional and only sharpens token
counts — without it, batching/truncation fall back to a cheap chars/4 estimate.


## Example

Construct an embedder directly:

```python
import asyncio
import embedotron as e

async def main():
    emb = e.EmbeddingOpenAI(model="text-embedding-3-small", api_key="sk-...", normalize=True)
    vecs, coins = await emb.ask_for_embedding([
        "The quick brown fox jumps over the lazy dog",
        "A fast auburn fox leaps above a sleepy hound",
        "Quantum chromodynamics describes the strong interaction",
    ])
    print("D =", emb.D, "coins:", sum(coins))
    print("fox vs fox    ", e.cosine(vecs[0], vecs[1]))
    print("fox vs physics", e.cosine(vecs[0], vecs[2]))
    # store the first vector in postgres (column type `vector`):
    #   await conn.execute("INSERT INTO t (emb) VALUES ($1::vector)", e.to_pgvector(vecs[0]))

asyncio.run(main())
```

Or via the config, which handles multi-tenant key injection and picks the right embedder:

```python
cfg = e.load_default_config(use_env_keys=True)        # keys from OPENAI_API_KEY etc.
emb = cfg.embedder_for("text-embedding-3-small", normalize=True)
vecs, coins = await emb.ask_for_embedding([...])
```

No keys, no network (deterministic vectors — same text always embeds the same):

```python
emb = e.EmbeddingRandom()
vecs, _ = await emb.ask_for_embedding(["hello", "hello", "goodbye"])
# e.cosine(vecs[0], vecs[1]) == 1.0
```


## Configuration

`providers_default.yaml` is the built-in config: a map of providers (each with a `kind`, endpoint,
key, and the model names it owns). A real multi-tenant deployment supplies its own config with
per-tenant keys; pass `use_env_keys=True` to fill keys from the environment, or `use_test_keys=True`
to read them from `.test_api_keys` for local runs.

To add a local embedding server, set a provider's `endpoint` to it, list the models you serve, and
give each a `modelcap_dimensions` (the embedder can't guess the vector size). TEI, Ollama, and vLLM
all speak the OpenAI `/embeddings` wire, so `kind: openai` just works.


## Testing

The pgvector test runs offline. The OpenAI tests hit the real API (a fraction of a cent) and are
skipped when no key is present.

```sh
cp .test_api_keys.example .test_api_keys   # add your keys (optional, for the live tests)

PYTHONPATH=. python embedotron/embedding_test.py            # all; live providers skipped if no key
PYTHONPATH=. python embedotron/embedding_test.py openai     # a subset by name
```

`.test_api_keys` holds live secrets and is git-ignored. Never commit it. If one leaks, rotate it.


## License

MIT — see [LICENSE](LICENSE).
