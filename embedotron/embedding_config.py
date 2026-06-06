import os

import yaml

from embedotron import embedding_openai
from embedotron import embedding_random


# A config is a set of providers, each owning a list of models, plus optional per-model overrides for
# openai-compatible servers that aren't in embedding_openai's built-in table. In a multi-tenant system
# each tenant supplies its own config with its own providers and keys; load_default_config() loads the
# built-in providers_default.yaml, with an explicit choice of where api keys come from.
#
# This layer is thin on purpose: embedder_for(model) finds the owning provider and constructs the right
# embedder subclass. The real work lives in the embedder classes, not here.


_DEFAULT_CONFIG = "providers_default.yaml"


class EmbeddingConfig:
    def __init__(self, providers, models):
        self.providers = providers
        self.models = models
        self._owner = {}
        for pid, prov in providers.items():
            for m in prov.get("models") or []:
                self._owner[m] = pid

    def embedder_for(self, model, *, dimensions=None, normalize=False):
        pid = self._owner.get(model)
        if pid is None:
            raise RuntimeError("model %r not owned by any provider" % model)
        prov = self.providers[pid]
        kind = prov["kind"]
        m = self.models.get(model) or {}
        if kind == "random":
            return embedding_random.EmbeddingRandom(dimensions=int(m.get("modelcap_dimensions", embedding_random.VECTOR_DIMENSION)))
        if kind in ("openai", "xai"):
            api_key = prov.get("api_key", "")
            if not api_key and prov.get("endpoint", "").startswith("https://api."):
                raise RuntimeError("no api_key for provider of model %r (set it, or load with use_env_keys=True)" % model)
            # Built-in openai models need no overrides; custom servers carry modelcap_* in the config.
            return embedding_openai.EmbeddingOpenAI(
                model=model,
                api_key=api_key,
                base_url=prov["endpoint"],
                dimensions=dimensions,
                normalize=normalize,
                model_dimensions=m.get("modelcap_dimensions"),
                model_max_tokens=m.get("modelcap_max_tokens"),
                pp1000t_prompt=int(m.get("pp1000t_prompt", 0)),
            )
        raise RuntimeError("unknown provider kind %r for model %r" % (kind, model))


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_config_path():
    return os.path.join(_repo_root(), _DEFAULT_CONFIG)


def _read_test_keys():
    keys = {}
    try:
        with open(os.path.join(_repo_root(), ".test_api_keys")) as f:
            lines = f.readlines()
    except FileNotFoundError:
        return keys
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        name, sep, value = line.partition("=")
        if sep:
            keys[name.strip()] = value.strip().strip('"').strip("'")
    return keys


def parse_config(data, use_env_keys=False, use_test_keys=False):
    providers = dict(data.get("providers") or {})
    models = dict(data.get("models") or {})
    if use_env_keys or use_test_keys:
        src = _read_test_keys() if use_test_keys else os.environ
        providers = {pid: dict(prov) for pid, prov in providers.items()}
        for prov in providers.values():
            env_name = prov.get("api_key_env")
            if env_name:
                prov["api_key"] = src.get(env_name, prov.get("api_key", ""))
    return EmbeddingConfig(providers, models)


def load_config(path, use_env_keys=False, use_test_keys=False):
    with open(path) as f:
        data = yaml.safe_load(f)
    return parse_config(data, use_env_keys=use_env_keys, use_test_keys=use_test_keys)


def load_default_config(use_env_keys=False, use_test_keys=False):
    return load_config(default_config_path(), use_env_keys=use_env_keys, use_test_keys=use_test_keys)
