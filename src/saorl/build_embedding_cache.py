"""Build a FROZEN embedding cache for the score-ablation's embedding condition.

Embeds every rule text, paraphrase, and candidate gloss the ablation scores
(text-embedding-3-small, via the OpenAI REST endpoint) ONCE and writes
results/conformal/embedding_cache.json = {text: [float,...]}. The ablation
reads only this cache, so the embedding condition is frozen and reproducible
(no API at ablation time), exactly like the 7-persona ensemble scores.

Run:  python3 -m saorl.build_embedding_cache      (needs $OPENAI_API_KEY)
"""
import json
import os
import urllib.request
from pathlib import Path

from .ablate_score_components import _rule_sets_lookup, _paraphrase_texts

_OUT = Path(__file__).parent.parent.parent / "results/conformal" / "embedding_cache.json"
_MODEL = "text-embedding-3-small"


def _collect_texts():
    texts = set()
    for meta in _rule_sets_lookup().values():
        texts.add(meta["rule_text"])
        texts.update(meta["glosses"].values())
    texts.update(_paraphrase_texts())
    return sorted(t for t in texts if t and t.strip())


def _embed(texts, key):
    # one batched request (a few dozen short strings)
    req = urllib.request.Request(
        "https://api.openai.com/v1/embeddings",
        data=json.dumps({"model": _MODEL, "input": texts}).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    resp = json.load(urllib.request.urlopen(req, timeout=120))
    return [d["embedding"] for d in resp["data"]], resp["usage"]["total_tokens"]


def main():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY not set")
    texts = _collect_texts()
    vecs, toks = _embed(texts, key)
    cache = {t: v for t, v in zip(texts, vecs)}
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(
        {"model": _MODEL, "n_texts": len(texts), "dim": len(vecs[0]),
         "total_tokens": toks, "vectors": cache}))
    print(f"cached {len(texts)} embeddings (dim {len(vecs[0])}, {toks} tokens) -> {_OUT}")


if __name__ == "__main__":
    main()
