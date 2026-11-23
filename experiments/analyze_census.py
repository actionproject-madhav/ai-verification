"""Summarise the Hub census: declared lineage verified, contradicted or unverified; undeclared lineage found.

Usage: python analyze_census.py <out_dir>   (reads hub_population.json, hub_census.jsonl; writes hub_census_summary.json)
"""
import collections, json, os, sys

O = sys.argv[1]
pop = json.load(open(os.path.join(O, "hub_population.json")))
recs = {}
for l in open(os.path.join(O, "hub_census.jsonl")):
    if l.strip():
        r = json.loads(l)
        recs[r["repo"]] = r                                     # a rerun's record replaces an earlier one
ANCH = {a["id"]: a for a in pop["anchors"]}
OFFICIAL = {a["id"]: a["base"].split("/")[0] for a in pop["anchors"]}
out = {"population": len(pop["population"]), "skipped_format": len(pop["skipped"]), "scanned": len(recs)}
rows = []
for r in recs.values():
    if "error" in r:
        status = "unreadable"
    elif r["group"] == "declared":
        status = "verified" if r["matched"] == r["declared"] else ("contradicted" if r["matched"] else "unverified")
    else:
        status = "lineage found" if r["matched"] else "no listed lineage"
    rows.append({**{k: r.get(k) for k in ("repo", "group", "declared", "relation", "shape_of", "matched", "downloads",
                                           "ablation_signature", "error", "sha")},
                 "status": status, "official": r["repo"].split("/")[0] == OFFICIAL.get(r.get("declared") or r.get("shape_of")),
                 "best": (min(((k, v) for k, v in (r.get("distances") or {}).items() if v is not None), key=lambda kv: kv[1], default=(None, None)))})
# Byte identity: a model within 0.002 of its anchor whose weight files hash the same as the anchor's is a re-upload.
from huggingface_hub import HfApi
api = HfApi()
_wh = {}
def weight_hashes(repo, rev=None):
    if (repo, rev) not in _wh:
        try:
            info = api.model_info(repo, revision=rev, files_metadata=True)
            _wh[(repo, rev)] = sorted(f.lfs.sha256 for f in info.siblings
                                      if f.rfilename.endswith(".safetensors") and "/" not in f.rfilename and f.lfs)
        except Exception:
            _wh[(repo, rev)] = None
    return _wh[(repo, rev)]
for x in rows:
    x["identical_files"] = None
    b = x["best"]
    if x["matched"] and b[1] is not None and b[1] < 0.002:
        mine, theirs = weight_hashes(x["repo"], x["sha"]), weight_hashes(ANCH[x["matched"]]["repo"])
        x["identical_files"] = bool(mine and theirs and mine == theirs)
# Facts for the write-up, read from the repos themselves: a merge's own recipe, and whether a card's text names its base.
import re
from huggingface_hub import hf_hub_download
FAMILY = {"qwen2.5-0.5b-instruct": r"qwen\s?2\.5", "qwen2.5-1.5b-instruct": r"qwen\s?2\.5", "llama-3.2-1b-instruct": r"llama[\s-]?3\.2",
          "smollm2-1.7b-instruct": r"smollm\s?2", "qwen3-1.7b": r"qwen\s?3"}
def repo_text(repo, rev, name):
    try:
        return open(hf_hub_download(repo, name, revision=rev)).read()
    except Exception:
        return None
for x in rows:
    x["merge_models"], x["card_names_base"] = None, None
    if x["status"] in ("unverified", "contradicted") and x["relation"] == "merge":
        y = repo_text(x["repo"], x["sha"], "mergekit_config.yml")
        if y:
            x["merge_models"] = sorted(set(re.findall(r"model:\s*([\w.\-]+/[\w.\-]+)", y)))
    if x["status"] == "lineage found":
        card = repo_text(x["repo"], x["sha"], "README.md")
        x["card_names_base"] = bool(card and re.search(FAMILY[x["matched"]], card, re.I))
c = collections.Counter((x["group"], x["status"]) for x in rows)
out["counts"] = {f"{g}/{s}": n for (g, s), n in sorted(c.items())}
out["by_relation"] = {rel: dict(collections.Counter(x["status"] for x in rows if x["relation"] == rel))
                      for rel in ("finetune", "merge", "quantized")}
out["ablation_flags"] = [x["repo"] for x in rows if x["ablation_signature"]]
out["identical_files"] = [x["repo"] for x in rows if x["identical_files"]]
lf = [x for x in rows if x["status"] == "lineage found" and not x["official"]]
out["found_third_party"], out["found_card_names_base"] = len(lf), sum(bool(x["card_names_base"]) for x in lf)
out["near_zero"] = sum(1 for x in rows if x["best"][1] is not None and x["matched"] and x["best"][1] < 0.002)
out["rows"] = sorted(rows, key=lambda x: (x["group"], x["status"], x["repo"]))
json.dump(out, open(os.path.join(O, "hub_census_summary.json"), "w"), indent=1)
print(f"population {out['population']} (+{out['skipped_format']} skipped for format), scanned {out['scanned']}")
for k, v in out["counts"].items():
    print(f"  {k:32s} {v}")
print("by relation:", out["by_relation"])
print("ablation flags:", out["ablation_flags"])
print(f"within 0.002 of the anchor: {out['near_zero']}; byte-identical weight files: {len(out['identical_files'])}", out["identical_files"][:12])
for x in out["rows"]:
    if x["status"] in ("contradicted", "unverified", "lineage found", "no listed lineage", "unreadable"):
        b = x["best"]
        print(f"  {x['status']:17s} {x['repo'][:60]:60s} declared={x['declared'] or x['shape_of']} matched={x['matched']} "
              f"best={b[0]}:{b[1]:.3f}" if b[1] is not None else f"  {x['status']:17s} {x['repo'][:60]:60s} {x['error'] or ''}"[:200])
