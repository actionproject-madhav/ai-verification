"""Does matching tokens by geometry find a copy whose tokenizer file was relabelled? Real models.

A tokenizer file comes from the operator. If it names the wrong rows, tables built by token name compare the wrong
tokens and a copy looks unrelated. Here each copy is read through a wrapper that shuffles its token rows and keeps
the old names. The light is asked for twice: by name, and with tokens matched by geometry. Models that are not
copies of a listed model go through the same second pass, to see that it flags none of them.

Usage: python exp_relabelled.py <out.json> [nonce]
"""
import json, os, sys, time
import numpy as np
import torch
from lineage_check import light as LT, remote as R, universal as U

torch.set_grad_enabled(False)
HERE = os.path.dirname(os.path.abspath(__file__))
WATCH = os.path.join(HERE, "..", "watchlists", "government-named.json")
KNOWN = os.path.join(HERE, "..", "watchlists", "known-demo.json")
CASES = [  # model, relabel it?, what it is
    ("nvidia/GLM-5.2-NVFP4", True, "4-bit release of a listed model"),
    ("dealignai/GLM-5.3-CYBERSECURITY-FP8", True, "8-bit release of a listed model, renamed"),
    ("nvidia/Kimi-K3-NVFP4", True, "4-bit release of a listed model"),
    ("moonshotai/Kimi-K3.1-Pro", True, "a later version of a listed model"),
    ("zai-org/GLM-5.3-Flash", False, "a different model that shares a listed name"),
    ("deepseek-ai/DeepSeek-V4-Flash", False, "a model on neither list"),
    ("Qwen/Qwen2.5-7B-Instruct", False, "the instruct version of a known model"),
    ("NousResearch/Hermes-3-Llama-3.1-8B", False, "a fine-tune of a known model"),
    ("zai-org/GLM-5.3-Flash", True, "a different model, relabelled"),
]


class Relabelled:
    """A model with its token rows shuffled, behind a tokenizer file that still names the old rows."""

    def __init__(self, src, seed=0):
        self.src, self.index, self.config = src, src.index, src.config
        self.emb = U.layout(src)[0]
        self.perm = np.random.default_rng(seed).permutation(U.shape_of(src, self.emb)[0])
        self.table = None

    @property
    def vocab(self):
        return self.src.vocab

    def shape(self, name):
        return U.shape_of(self.src, name)

    def get(self, name):
        return self.src.get(name)

    def rows(self, name, lo, hi):
        if name != self.emb:
            return self.src.rows(name, lo, hi)
        if self.table is None:
            V = len(self.perm)
            self.table = np.concatenate([self.src.rows(name, a, min(a + 16384, V)) for a in range(0, V, 16384)])
        return self.table[self.perm[lo:hi]]


def main(out, nonce="relabelled-2026-11"):
    watch, known, cache, rows = LT.sources(WATCH), LT.sources(KNOWN), {}, []
    for repo, relabel, what in CASES:
        t0 = time.time()
        row = {"model": repo, "relabelled": relabel, "what": what}
        try:
            src = R.Remote(repo)
            src = Relabelled(src) if relabel else src
            for key, geometry in (("by_name", False), ("with_geometry", True)):
                r = LT.light(src, watch, known, nonce, cache=cache, geometry=geometry)
                row[key] = {"light": r["light"], "matched": r["matched"],
                            "distances": {k: None if v is None else [round(v[0], 4), round(v[1], 4)] for k, v in r["distances"].items()}}
            for k in [k for k in cache if k[0] == id(src)]:        # drop this model's tables before the next one
                del cache[k]
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {str(e)[:160]}"
        row["seconds"] = round(time.time() - t0)
        rows.append(row)
        print(row, flush=True)
        json.dump({"nonce": nonce, "rows": rows}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:3])
