"""Do exact file hashes link popular models to the base they come from?

For every scanned model that the fingerprint traces to a registry base (distance under 0.5), fetch the SHA-256 of
each weight file from the Hub's metadata and ask whether the model shares any weight file with that base. A shared
file is the most an exact hash can see. No weights are read.

Usage: python exp_scan_hashes.py <scan dir with compare.json> <out.json>
"""
import collections, json, os, sys, time
from huggingface_hub import HfApi

api = HfApi()
WEIGHTS = (".safetensors", ".bin", ".gguf", ".pt", ".pth")


def weight_hashes(repo, cache={}):
    if repo not in cache:
        out = None
        for attempt in range(6):
            try:
                out = set()
                for s in api.model_info(repo, files_metadata=True).siblings:
                    lfs = getattr(s, "lfs", None)
                    sha = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
                    if sha and s.rfilename.endswith(WEIGHTS):
                        out.add(sha)
                break
            except Exception as e:
                out = None
                if "429" not in str(e):
                    break
                time.sleep(60 * (attempt + 1))                  # rate limited: wait and retry
        cache[repo] = out
        time.sleep(0.4)
    return cache[repo]


def main(scan_dir, out):
    rows = json.load(open(os.path.join(scan_dir, "compare.json")))["rows"]
    traced = [r for r in rows if r.get("nearest") and r.get("distance", 2) < 0.5]
    done = {r["repo"]: r for r in json.load(open(out))["rows"]} if os.path.exists(out) else {}
    res = []
    for i, r in enumerate(traced):
        if r["repo"] in done:
            res.append(done[r["repo"]])
            continue
        a, q = weight_hashes(r["nearest"]), weight_hashes(r["repo"])
        res.append({"repo": r["repo"], "base": r["nearest"], "distance": r["distance"], "light": r["light"],
                    "weight_files": None if q is None else len(q), "shared_files": None if a is None or q is None else len(a & q)})
        if i % 50 == 0 or i == len(traced) - 1:
            ok = [x for x in res if x["shared_files"] is not None]
            summary = {"traced_to_a_base": len(traced), "checked": len(ok), "share_a_weight_file": sum(x["shared_files"] > 0 for x in ok),
                       "by_light": {k: [sum(x["shared_files"] > 0 for x in ok if x["light"] == k), sum(x["light"] == k for x in ok)]
                                    for k in ("close copy", "modified copy")}}
            json.dump({"summary": summary, "rows": res}, open(out, "w"), indent=1)
            print(i + 1, "of", len(traced), summary, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
