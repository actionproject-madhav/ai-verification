"""The design-free fingerprint on real 7-9B models: true derivatives against independently trained models.

Each pair is compared on 256 tokens drawn from the anchor's vocabulary and two layers, with no knowledge of
either architecture. Reports the distance (best-matching third of the tables), the mean over every table, and
the distance embeddings alone would give.

Usage: python exp_universal.py <out.json>
"""
import json, sys
import torch
from lineage_check import fingerprint as fp4, remote as R, universal as U

torch.set_grad_enabled(False)
NONCE = "universal-7b"
LLAMA, QWEN = "unsloth/Llama-3.1-8B", "Qwen/Qwen2.5-7B"
PAIRS = [  # label, kind, anchor, anchor layers, query, query layers
    ("Llama-3.1-8B-Instruct", "derivative", LLAMA, [12, 20], "unsloth/Llama-3.1-8B-Instruct", [12, 20]),
    ("Hermes-3 (fine-tune of Llama-3.1-8B)", "derivative", LLAMA, [12, 20], "NousResearch/Hermes-3-Llama-3.1-8B", [12, 20]),
    ("Llama-3.1-8B-Instruct abliterated", "derivative", LLAMA, [12, 20], "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated", [12, 20]),
    ("Qwen2.5-7B-Instruct", "derivative", QWEN, [10, 17], "Qwen/Qwen2.5-7B-Instruct", [10, 17]),
    ("Marin-8B (same design and tokenizer as Llama-3.1)", "independent", LLAMA, [12, 20], "marin-community/marin-8b-base", [12, 20]),
    ("Qwen3-8B-Base (same tokenizer as Qwen2.5)", "independent", QWEN, [10, 17], "Qwen/Qwen3-8B-Base", [13, 22]),
    ("Mistral-7B-v0.3 against Llama-3.1-8B", "independent", LLAMA, [12, 20], "mistralai/Mistral-7B-v0.3", [12, 20]),
    ("Qwen2.5-7B against Llama-3.1-8B", "independent", LLAMA, [12, 20], QWEN, [10, 17]),
]
_src, _fp = {}, {}


def source(repo):
    if repo not in _src:
        _src[repo] = R.Remote(repo)
        _src[repo].index.pop("lm_head.weight", None)
    return _src[repo]


def fingerprint(repo, layers, anchor):
    key = (repo, tuple(layers), anchor)
    if key not in _fp:
        strings = fp4.select_strings(source(anchor).vocab, NONCE, 256)
        _fp[key] = U.fingerprint(source(repo), source(repo).vocab, strings, layers=layers)
    return _fp[key]


def main(out):
    res = []
    for label, kind, a, la, q, lq in PAIRS:
        r = U.compare(fingerprint(a, la, a), fingerprint(q, lq, a))
        res.append({"pair": label, "kind": kind, "anchor": a, "query": q, **r})
        print(f"{kind:12s} {label:52s} d {r['d']:.3f}  all tables {r['d_all']:.3f}  embeddings alone {r['embedding_only']:.3f}  "
              f"({r['tokens']} tokens, {r['matches']} matches)", flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
