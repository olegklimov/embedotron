import asyncio
import sys

import numpy as np

import embedotron as e


def _openai_embedder():
    cfg = e.load_default_config(use_test_keys=True)
    if not cfg.providers["openai_global"].get("api_key"):
        return None
    return cfg.embedder_for("text-embedding-3-large", normalize=False)


async def test_openai():
    emb = _openai_embedder()
    if emb is None:
        print("openai_multilingual: SKIP (no OPENAI_API_KEY in .test_api_keys)")
        return
    a = "Turning coffee into deliverables since 8:03 AM"
    b = "午前 8 時 3 分からコーヒーを成果物に変える"   # the same joke in Japanese
    c = "Reducing risk by increasing paperwork"
    vecs = await emb.ask_for_embedding([a, b, c])
    va, vb, vc = vecs
    for phrase, v in zip([a, b, c], vecs):
        print("%r\n\tD=%d norm=%.4f  %s ..." % (phrase, len(v), np.linalg.norm(v), np.array2string(v[:6], precision=3)))
    print("Cosine similariy, closer to 1 the more similar it is:")
    print("  EN coffee  <-> JA coffee    cos=%.4f" % e.cosine(va, vb))
    print("  EN coffee  <-> EN paperwork cos=%.4f" % e.cosine(va, vc))
    print("  JA coffee  <-> EN paperwork cos=%.4f" % e.cosine(vb, vc))
    print("throwaway_score=%.4f" % emb.throwaway_score())


ALL_TESTS = {
    "openai": test_openai,
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
