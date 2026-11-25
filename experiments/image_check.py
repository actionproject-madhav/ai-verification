"""Inside the enclave image: do its own builds of the encoder give the census's verdict on a sample of files?

Run in the image, with the census rows mounted (results/derive/image_check.txt holds the run of 24 Nov 2026):
  docker run --rm --entrypoint python3 -e HF_HUB_OFFLINE=0 -v <census.jsonl>:/rows.jsonl:ro -v <this file>:/check.py:ro lineage-check /check.py /rows.jsonl
"""
import json, os, random, sys, time
from huggingface_hub import hf_hub_download
from lineage_check import derive as D

rows = [json.loads(l) for l in open(sys.argv[1])]
rng = random.Random(0)
small = lambda r: r["base"].split("/")[1] in ("SmolLM-135M", "SmolLM2-135M", "SmolLM-360M", "SmolLM2-360M", "SmolLM2-135M-Instruct", "SmolLM2-360M-Instruct", "Qwen2.5-0.5B", "Qwen2.5-0.5B-Instruct")
groups = {}
for r in rows:
    if "error" in r or not small(r):
        continue
    key = ("+".join(r.get("builds") or []) or "none") if r["derived"] else "not derived"
    groups.setdefault(key, []).append(r)
pick = []
for key, rs in groups.items():
    pick += rng.sample(rs, min(len(rs), 8 if key != "not derived" else 10))
encoders = D.builds()
print("image builds:", [e.name for e in encoders], flush=True)
known, agree = {}, 0
for r in pick:
    t0 = time.time()
    k = known.setdefault(r["base"], D.Known.open(r["base"]))
    c = D.check(hf_hub_download(r["repo"], r["file"]), k, encoders)
    want = r["derived"] and r.get("vocab", {}).get("same", 1) == r.get("vocab", {}).get("compared", 1)
    build_ok = (not r.get("builds")) or (set(r["builds"]) & set(c["builds"])) or not c["exact"]
    same = c["exact"] == want and bool(build_ok)
    agree += same
    print(f"{'SAME' if same else 'DIFF'}  census {('exact ' + '+'.join(r.get('builds') or ['-'])) if want else 'not exact':22s} image {('exact ' + '+'.join(c['builds'] or ['-'])) if c['exact'] else 'not exact':28s} {r['repo']}/{r['file']}  {time.time() - t0:.1f} s", flush=True)
print(f"{agree} of {len(pick)} agree")
