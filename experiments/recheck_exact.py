"""Re-check the census's exact files with the tool's own check (lineage_check.derive.check), whose rule is stricter:
the file's architecture is the known model's, it holds every tensor of the known checkpoint and no other, each in a
format the box can read, and the tokens match.

Usage: python recheck_exact.py <census rows ...> --out <out.jsonl>     (LINEAGE_GGML: the builds to try; DERIVE_SHARD=i/n)
"""
import json, os, sys, time
from huggingface_hub import hf_hub_download
from lineage_check import derive as D

args = sys.argv[1:]
out = args[args.index("--out") + 1]
files = [a for a in args if a != "--out" and a != out]
shard = tuple(int(v) for v in os.environ.get("DERIVE_SHARD", "0/1").split("/"))
rows = {}
for f in files:
    for r in map(json.loads, open(f)):
        if r.get("derived"):
            base = r.get("base") if "declared" not in r else r["base"]          # sibling rows carry the sibling as their base
            rows[(r["repo"], r["file"])] = {"repo": r["repo"], "file": r["file"], "base": base}
todo = sorted(rows.values(), key=lambda r: (r["base"], r["repo"], r["file"]))
done = {(r["repo"], r["file"]) for r in map(json.loads, open(out))} if os.path.exists(out) else set()
encoders, known, base = D.builds(), None, None
for i, r in enumerate(todo):
    if i % shard[1] != shard[0] or (r["repo"], r["file"]) in done:
        continue
    t0 = time.time()
    try:
        if r["base"] != base:
            base, known = r["base"], D.Known.open(r["base"])
        c = D.check(hf_hub_download(r["repo"], r["file"]), known, encoders)
        row = {**r, **{k: c.get(k) for k in ("exact", "reason", "architecture", "tensors", "identical", "missing", "extra", "formats",
                                             "unservable", "bits_per_weight", "builds", "vocab", "template", "sha256")}}
    except Exception as err:
        row = {**r, "error": f"{type(err).__name__}: {str(err)[:200]}"}
    row["seconds"] = round(time.time() - t0, 1)
    with open(out, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(r["repo"], r["file"], row.get("exact"), row.get("reason") or row.get("error", ""), flush=True)
