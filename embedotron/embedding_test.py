import asyncio
import sys

import numpy as np

import embedotron as e


# The pgvector test runs offline. The OpenAI tests hit the real API (a fraction of a cent) and are
# skipped when no key is in .test_api_keys.
#
#   python embedotron/embedding_test.py                 # all; live providers skipped if no key
#   python embedotron/embedding_test.py openai          # a subset by name


async def test_pgvector_roundtrip():
    v = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    s = e.to_pgvector(v)
    back = e.from_pgvector(s)
    assert np.allclose(v, back, atol=1e-6), (v, back)
    print("pgvector_roundtrip: %s" % s)


def _openai_embedder():
    cfg = e.load_default_config(use_test_keys=True)
    if not cfg.providers["openai_global"].get("api_key"):
        return None
    return cfg.embedder_for("text-embedding-3-small", normalize=True)


async def test_openai():
    emb = _openai_embedder()
    if emb is None:
        print("openai: SKIP (no OPENAI_API_KEY in .test_api_keys)")
        return
    vecs, coins = await emb.ask_for_embedding(["The quick brown fox", "A fast auburn fox", "Quantum chromodynamics"])
    assert emb.D == 1536, emb.D
    near = e.cosine(vecs[0], vecs[1])
    far = e.cosine(vecs[0], vecs[2])
    assert near > far, (near, far)
    print("openai: near=%.4f far=%.4f coins=%d" % (near, far, sum(coins)))


async def test_openai_multilingual():
    emb = _openai_embedder()
    if emb is None:
        print("openai_multilingual: SKIP (no OPENAI_API_KEY in .test_api_keys)")
        return
    a = "Turning coffee into deliverables since 8:03 AM"
    b = "午前 8 時 3 分からコーヒーを成果物に変える"   # the same joke in Japanese
    c = "Reducing risk by increasing paperwork"
    vecs, coins = await emb.ask_for_embedding([a, b, c])
    va, vb, vc = vecs
    print("openai_multilingual:")
    print("  EN coffee  <-> JA coffee    cos=%.4f" % e.cosine(va, vb))
    print("  EN coffee  <-> EN paperwork cos=%.4f" % e.cosine(va, vc))
    print("  JA coffee  <-> EN paperwork cos=%.4f" % e.cosine(vb, vc))
    print("  coins=%d" % sum(coins))


ALL_TESTS = {
    "pgvector_roundtrip": test_pgvector_roundtrip,
    "openai": test_openai,
    "openai_multilingual": test_openai_multilingual,
}


async def main():
    for name in sys.argv[1:] or list(ALL_TESTS.keys()):
        fn = ALL_TESTS.get(name)
        if fn is None:
            print("unknown test %r, available: %s" % (name, ", ".join(ALL_TESTS)))
            continue
        await fn()


if __name__ == "__main__":
    asyncio.run(main())
