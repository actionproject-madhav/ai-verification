"""The traffic light on real models: copies of the government-named models, close copies of known models, models
with their refusals removed, models on neither list, and GGUF files of each kind.

Usage: python exp_lights.py <watchlist.json> <known.json> <out.json>          (LIGHT_NONCE sets the nonce)
       python exp_lights.py stability <out.json> <run.json> ...              (several nonces, side by side)
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
    ("huihui-ai/Qwen2.5-7B-Instruct-abliterated-v2", "a known model with its refusals removed"),
    ("DreamFast/qwen3-8b-heretic", "a known model with its refusals removed, by a newer tool"),
    ("deepseek-ai/DeepSeek-V4-Flash", "a model on neither list"),
    ("unsloth/GLM-5.3-GGUF", "GGUF release of a listed model, 4 bits"),
    ("unsloth/Kimi-K3-GGUF", "GGUF release of a listed model, 4 bits"),
    ("huihui-ai/Huihui-GLM-5.3-abliterated-GGUF", "GGUF of a listed model with its refusals removed, 1 bit"),
    ("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "GGUF release of a known model, 4 bits"),
    ("mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated-GGUF", "GGUF of a known model with its refusals removed, 4 bits"),
]


def main(watch_path, known_path, out):
    watch, known, cache, rows = LT.sources(watch_path), LT.sources(known_path), {}, []
    for repo, what in CASES:
        t0 = time.time()
        row = {"repo": repo, "what": what}
        try:
            src = R.Remote(repo)
            signature = LT.ablation_signature(src, NONCE)
            r = LT.light(src, watch, known, NONCE, ablation=signature, cache=cache)
            row.update(revision=src.sha, file=src.gguf[0] if src.gguf else None, light=r["light"], matched=r["matched"], finding=r["finding"], reason=r["reason"], signature=signature,
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


def stability(out, *runs):
    """One file from several runs under different nonces: per model, the light, the matched entry and the rule that
    decided, run by run."""
    runs = [json.load(open(r)) for r in runs]
    table = {}
    for run in runs:
        for r in run["rows"]:
            table.setdefault(r["repo"], []).append((r.get("light", "error"), r.get("matched"), r.get("finding")))
    rows = [{"repo": k, "lights": [v[0] for v in vs], "matched": [v[1] for v in vs], "findings": [v[2] for v in vs]} for k, vs in table.items()]
    json.dump({"nonces": [r["nonce"] for r in runs], "rows": rows}, open(out, "w"), indent=1)
    for r in rows:
        print(f"{r['repo'][:52]:52s} {'same light' if len(set(r['lights'])) == 1 else 'LIGHT CHANGES'}  {sorted(set(map(str, r['findings'])))}")


if __name__ == "__main__":
    if sys.argv[1] == "stability":
        stability(sys.argv[2], *sys.argv[3:])
    else:
        main(*sys.argv[1:4])
