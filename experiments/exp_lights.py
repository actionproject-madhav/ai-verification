"""The traffic light on real models: copies of the government-named models, close copies of known models, a model
with its refusals removed, and models on neither list.

Usage: python exp_lights.py <watchlist.json> <known.json> <out.json>
"""
import json, os, shutil, sys, time
import torch
from lineage_check import light as LT, remote as R

torch.set_grad_enabled(False)
NONCE = os.environ.get("LIGHT_NONCE", "lights-2026-11")
CASES = [
    ("nvidia/GLM-5.2-NVFP4", "4-bit release of a listed model"),
    ("dealignai/GLM-5.3-CYBERSECURITY-FP8", "8-bit release of a listed model, renamed"),
    ("nvidia/Kimi-K3-NVFP4", "4-bit release of a listed model"),
    ("moonshotai/Kimi-K3.1-Pro", "a later version of a listed model"),
    ("zai-org/GLM-5.3-Flash", "a different model that shares a listed name"),
    ("tiny-random/glm-5.2", "random weights under a listed name"),
    ("Qwen/Qwen2.5-7B-Instruct", "the instruct version of a known model"),
    ("NousResearch/Hermes-3-Llama-3.1-8B", "a fine-tune of a known model"),
    ("mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated", "a known model with its refusals removed"),
    ("deepseek-ai/DeepSeek-V4-Flash", "a model on neither list"),
]


def main(watch_path, known_path, out):
    watch, known, cache, rows = LT.sources(watch_path), LT.sources(known_path), {}, []
    for repo, what in CASES:
        t0 = time.time()
        row = {"repo": repo, "what": what}
        try:
            src = R.Remote(repo)
            r = LT.light(src, watch, known, NONCE, ablation=LT.ablation_signature(src, NONCE), cache=cache)
            row.update(revision=src.sha, light=r["light"], matched=r["matched"], reason=r["reason"],
                       distances={k: None if v is None else [round(v[0], 4), round(v[1], 4)] for k, v in r["distances"].items()})
            for k in [k for k in cache if k[0] == id(src)]:      # keep the list entries' fingerprints, drop this model's
                del cache[k]
        except Exception as e:
            row.update(error=f"{type(e).__name__}: {str(e)[:160]}")
        row["seconds"] = round(time.time() - t0)
        rows.append(row)
        print(f"{repo[:52]:52s} {row.get('light', 'ERROR'):6s} {str(row.get('matched')):14s} {row.get('reason', row.get('error', ''))}  [{row['seconds']}s]", flush=True)
        json.dump({"nonce": NONCE, "rows": rows}, open(out, "w"), indent=1)
        shutil.rmtree(os.path.join(R.CACHE, repo.replace("/", "--")), ignore_errors=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
