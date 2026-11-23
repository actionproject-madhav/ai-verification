"""False positives at frontier scale: every pair among nine independently trained 7-9B model families.

Different families use different tokenizers, so the audit tokens are plain English words mapped to token ids by
each model's own tokenizer (single-token words only). Layers are sampled at the same relative depths and matched
by the order-preserving alignment. Includes Marin-8B, trained from scratch by Stanford with Llama-3.1's exact
architecture and tokenizer: the hardest independent pair available at this scale.

Usage: python exp_crossfamily.py <out.json>
"""
import gc, json, os, sys, time
import numpy as np
import torch
from transformers import AutoTokenizer
from lineage_check import fingerprint as fp4
from lineage_check import remote as R
from lineage_check import paths

torch.set_grad_enabled(False)
K = 4
MODELS = ["unsloth/Llama-3.1-8B", "marin-community/marin-8b-base", "Qwen/Qwen2.5-7B", "Qwen/Qwen3-8B-Base",
          "mistralai/Mistral-7B-v0.3", "allenai/OLMo-7B-0724-hf", "01-ai/Yi-1.5-9B", "LLM360/Amber",
          "ibm-granite/granite-3.0-8b-base"]
FP_DIR = os.path.join(paths.ZOO, "fps_cross")
os.makedirs(FP_DIR, exist_ok=True)


def word_list(n=1500):
    """Common English words: word-like tokens from a byte-level vocabulary (lowercase, 3+ letters)."""
    v = R.Remote("Qwen/Qwen2.5-7B").vocab
    inv = sorted((i, s) for s, i in v.items())
    out = []
    for i, s in inv:
        if s.startswith("Ġ") and s[1:].isalpha() and s[1:].islower() and len(s) >= 4:
            out.append(s[1:])
        if len(out) >= n:
            break
    return out


def ids_for(tok, words, lo, hi):
    ids = []
    for w in words:
        got = None
        for t in (" " + w, w):
            e = tok(t, add_special_tokens=False)["input_ids"]
            if len(e) == 1 and lo <= e[0] < hi and tok.decode(e).strip() == w:
                got = e[0]
                break
        ids.append(got)
    return ids


def fingerprint(repo, words):
    path = os.path.join(FP_DIR, repo.replace("/", "__") + ".pt")
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    src = R.Remote(repo)
    try:
        tok = AutoTokenizer.from_pretrained(repo)
    except ImportError:   # the slow tokenizer needs protobuf/sentencepiece: read the fast tokenizer.json directly
        from huggingface_hub import hf_hub_download
        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast(tokenizer_file=hf_hub_download(repo, "tokenizer.json"))
    meta = R.build(src, layers=[])
    L = meta["L"]
    layers = sorted({int(round((j + 0.5) * L / K - 0.5)) for j in range(K)})
    ids = ids_for(tok, words, meta["lo"], meta["lo"] + meta["E"].shape[0])
    present = [i is not None for i in ids]
    rows = [i - meta["lo"] for i in ids if i is not None]
    fp = fp4.fingerprint(meta, rows, layers=[])
    E = torch.from_numpy(np.ascontiguousarray(meta["E"]))[torch.as_tensor(rows)]
    for i in layers:
        mi = R.build(src, layers=[i])
        fp["layers"][i] = fp4.layer_comps(mi["layers"][i], E, mi)
        del mi
        gc.collect()
    fp.update(repo=repo, present=present, L=L, d=meta["d"], kind=meta["kind"])
    torch.save(fp, path)
    return fp


def compare(fa, fb):
    common = [t for t in range(len(fa["present"])) if fa["present"][t] and fb["present"][t]]
    def sub(fp):
        pos, k = {}, 0
        for t, p in enumerate(fp["present"]):
            if p:
                pos[t] = k
                k += 1
        idx = torch.as_tensor([pos[t] for t in common])
        s = lambda M: M[idx][:, idx]
        return {"EE": s(fp["EE"]), "EU": s(fp["EU"]), "layers": {i: {c: s(M) for c, M in L.items()} for i, L in fp["layers"].items()}}
    r = fp4.compare(sub(fa), sub(fb), align=True)
    return {"d": r["d"], "attn": float(np.mean([r["qk"], r["ov"]])), "common_words": len(common)}


def main(out):
    words = word_list()
    fps = {}
    for repo in MODELS:
        t0 = time.time()
        try:
            fps[repo] = fingerprint(repo, words)
            print(f"{repo:40s} fingerprinted: {sum(fps[repo]['present'])} of {len(words)} words, L={fps[repo]['L']}, d={fps[repo]['d']} ({time.time() - t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"{repo:40s} ERROR {type(e).__name__}: {str(e)[:160]}", flush=True)
    res = []
    names = [m for m in MODELS if m in fps]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = compare(fps[names[i]], fps[names[j]])
            res.append({"a": names[i], "b": names[j], **r})
            print(f"  {names[i].split('/')[1]:24s} vs {names[j].split('/')[1]:24s} attn {r['attn']:.3f}  all {r['d']:.3f}  ({r['common_words']} words)", flush=True)
    json.dump(res, open(out, "w"), indent=1)
    a = [r["attn"] for r in res]
    print(f"\n{len(res)} pairs: attention-circuit distance min {min(a):.3f}, median {np.median(a):.3f}, max {max(a):.3f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
