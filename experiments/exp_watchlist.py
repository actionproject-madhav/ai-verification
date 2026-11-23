"""The models government evaluators have named, and the copies of them already on the Hub.

For each listed model: the most-downloaded repos that declare it as their base or carry its name, plus other large
open-weight models as controls. Every repo is fingerprinted with the design-free fingerprint (two layers, 256
tokens from the listed model's vocabulary) and compared with every listed model.

Usage: python exp_watchlist.py <watchlist.json> <out.json> [per_model]
"""
import collections, json, os, shutil, sys, time
import torch
from huggingface_hub import HfApi
from lineage_check import fingerprint as fp4, remote as R, universal as U

torch.set_grad_enabled(False)
NONCE = os.environ.get("WATCH_NONCE", "watchlist-2026-11")
CONTROLS = ["deepseek-ai/DeepSeek-V4-Flash", "deepseek-ai/DeepSeek-V4.1-Flash", "zai-org/GLM-5.3-Flash",
            "moonshotai/Kimi-K3.1-Pro", "Qwen/Qwen3.8-27B", "Qwen/Qwen3.6-35B-A3B"]


def vocab_of(src):
    try:
        return src.vocab, "tokenizer"
    except Exception:
        emb = U.layout(src)[0]
        return {str(i): i for i in range(U.shape_of(src, emb)[0])}, "token ids (no tokenizer file)"


def candidates(api, listed, per):
    out, seen = [], {m["repo"] for m in listed}
    for m in listed:
        got = []
        for rel in ("finetune", "quantized", "merge"):
            for x in api.list_models(filter=f"base_model:{rel}:{m['repo']}", sort="downloads", limit=60, expand=["safetensors", "downloads"]):
                got.append((x, rel))
        name = m["repo"].split("/")[1]
        for x in api.list_models(search=name, sort="downloads", limit=60, expand=["safetensors", "downloads"]):
            got.append((x, "name"))
        n = 0
        for x, rel in sorted(got, key=lambda t: -(t[0].downloads or 0)):
            if x.id in seen:
                continue
            seen.add(x.id)
            if not getattr(x, "safetensors", None):
                out.append({"repo": x.id, "listed": m["id"], "relation": rel, "downloads": x.downloads, "skip": "no safetensors (GGUF or other)"})
                continue
            if n < per:
                out.append({"repo": x.id, "listed": m["id"], "relation": rel, "downloads": x.downloads})
                n += 1
    return out + [{"repo": r, "listed": None, "relation": "control", "downloads": None} for r in CONTROLS if r not in seen]


def main(wl_path, out, per=10):
    wl = json.load(open(wl_path))
    api = HfApi()
    anchors = {}
    for m in wl["models"]:
        t0 = time.time()
        src = R.Remote(m["repo"])
        vocab, how = vocab_of(src)
        strings = fp4.select_strings(vocab, NONCE, 256) if how == "tokenizer" else [str(i) for i in torch.randperm(19000, generator=torch.Generator().manual_seed(7))[:256].add(500).tolist()]
        f = U.fingerprint(src, vocab, strings)
        anchors[m["id"]] = (f, strings, how)
        print(f"listed {m['id']:10s} d={f['d']} layers {f['layer_ids']} tables {sum(len(x) for x in f['layers'])} tokens by {how} ({time.time() - t0:.0f}s)", flush=True)
        print("   matrices:", sorted({U.LAYER.search(n).group(2) for n in f["used"]})[:20], flush=True)
        shutil.rmtree(os.path.join(R.CACHE, m["repo"].replace("/", "--")), ignore_errors=True)
    rows = []
    pairs = [(a, b) for a in anchors for b in anchors if a < b]
    for a, b in pairs:                                       # are the listed models related to each other?
        fb = None
        try:
            src = R.Remote(next(m["repo"] for m in wl["models"] if m["id"] == b))
            fb = U.fingerprint(src, vocab_of(src)[0], anchors[a][1], layers=None)
            r = U.compare(anchors[a][0], fb)
            print(f"listed pair {a} vs {b}: {r}", flush=True)
            rows.append({"repo": b, "against": a, "relation": "listed pair", **(r or {"d": None})})
        except Exception as e:
            print(f"listed pair {a} vs {b}: {type(e).__name__}: {str(e)[:120]}", flush=True)
    for c in candidates(api, wl["models"], per):
        if c.get("skip"):
            rows.append(c)
            continue
        t0 = time.time()
        try:
            src = R.Remote(c["repo"])
            dist = {}
            for mid, (fa, strings, how) in anchors.items():
                vocab = vocab_of(src)[0] if how == "tokenizer" else {s: int(s) for s in strings}
                fq = U.fingerprint(src, vocab, strings)
                r = U.compare(fa, fq)
                dist[mid] = None if r is None else round(r["d"], 4)
                c.setdefault("tables", sum(len(x) for x in fq["layers"]))
            ok = {k: v for k, v in dist.items() if v is not None}
            best = min(ok, key=ok.get) if ok else None
            c.update(distances=dist, matched=best if best and ok[best] < 0.5 else None, seconds=round(time.time() - t0))
        except Exception as e:
            c.update(error=f"{type(e).__name__}: {str(e)[:160]}", seconds=round(time.time() - t0))
        rows.append(c)
        print(f"{c['repo'][:58]:58s} {str(c['listed']):9s} {c['relation']:9s} -> {str(c.get('matched')):9s} {c.get('distances', c.get('error', ''))}", flush=True)
        json.dump({"watchlist": wl, "rows": rows}, open(out, "w"), indent=1)
        shutil.rmtree(os.path.join(R.CACHE, c["repo"].replace("/", "--")), ignore_errors=True)
    n = collections.Counter((r.get("listed") is not None, bool(r.get("matched"))) for r in rows if "distances" in r)
    print("declared or named copies matched:", n[(True, True)], "of", n[(True, True)] + n[(True, False)],
          "| controls matched:", n[(False, True)], "of", n[(False, True)] + n[(False, False)])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *[int(x) for x in sys.argv[3:4]])
