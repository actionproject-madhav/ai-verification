"""Recompute the README's headline numbers from the files in results/. Exits non-zero if one no longer holds.

Usage: python headline.py [results dir]
"""
import collections, json, os, sys

RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
load = lambda f: json.load(open(os.path.join(RES, f)))
bad = []


def claim(text, holds, measured):
    bad.append(text) if not holds else None
    print(("ok    " if holds else "FAILS ") + text + f"   [{measured}]")


def run(f, mode):
    """One mode of a stress-test file, after its last restart."""
    rows = [r for r in map(json.loads, filter(str.strip, open(os.path.join(RES, f)))) if r.get("mode") == mode]
    return rows[max([i for i in range(1, len(rows)) if rows[i]["step"] < rows[i - 1]["step"]], default=0):]


# hashes against the fingerprint
h = load("hash_baseline.json")
copies = [r for r in h if not r["kind"].startswith("unrelated")]
others = [r for r in h if r["kind"].startswith("unrelated")]
n = lambda rows, f: sum(map(f, rows))
claim("file hash 2, tensor hash 7, fingerprint 13 of 13 public copies; none flags the 3 unrelated models",
      (len(copies), n(copies, lambda r: r["files_identical"] > 0), n(copies, lambda r: r["tensors_identical"] > 0),
       n(copies, lambda r: r["closest_part"] < 0.5), n(others, lambda r: r["closest_part"] < 0.5)) == (13, 2, 7, 13, 0),
      f"{len(copies)} copies, {len(others)} unrelated")
s = load("hub_scan/scan_hashes.json")["summary"]
claim("file hashes link 74 of 1,718 copies in the Hub scan", (s["share_a_weight_file"], s["traced_to_a_base"]) == (74, 1718),
      f"{s['share_a_weight_file']} of {s['traced_to_a_base']}")

# copies against unrelated models
watch = load("watchlist.json")["rows"]
best = lambda r: min(v for v in r["distances"].values() if v is not None)
matched = [r for r in watch if r.get("matched")]
unmatched = [r for r in watch if "distances" in r and not r.get("matched") and r.get("relation") != "listed pair"
             and any(v is not None for v in r["distances"].values())]
pairs = load("universal_pairs.json")["rows"]
close = ([max(best(r), 0) for r in matched] + [r["d"] for r in load("universal.json") if r["kind"] == "derivative"]
         + [r["closest_part"] for r in pairs if r["kind"] != "retrained at length"])
far = [r["closest_part"] for r in load("universal_cross.json")["rows"]] + [best(r) for r in unmatched]
long = [r["closest_part"] for r in pairs if r["kind"] == "retrained at length"]
claim("27 modified copies all under 0.07", len(close) == 27 and max(close) < 0.07, f"{len(close)} copies, furthest {max(close):.3f}")
claim("151 unrelated pairs all above 0.83", len(far) == 151 and min(far) > 0.83, f"{len(far)} pairs, closest {min(far):.3f}")
claim("14 public copies of listed models at 0.017 or less, 15 others at 0.95 or more",
      len(matched) == 14 and max(map(best, matched)) < 0.0175 and len(unmatched) == 15 and min(map(best, unmatched)) >= 0.95,
      f"{max(map(best, matched)):.4f}, {min(map(best, unmatched)):.4f}")
claim("five long retrains at 0.63 to 0.95", len(long) == 5 and 0.63 <= min(long) and max(long) <= 0.95, f"{min(long):.3f} to {max(long):.3f}")

# the lights
lights = collections.Counter(r["light"] for r in load("lights.json")["rows"])
stab = load("lights_stability.json")
claim("twelve real models: 7 Red, 2 Green, 3 Amber", lights == {"red": 7, "green": 2, "amber": 3}, dict(lights))
claim("no light changes over five nonces", len(stab["nonces"]) == 5 and all(len(set(r["lights"])) == 1 for r in stab["rows"]),
      f"{len(stab['rows'])} models x {len(stab['nonces'])} nonces")

# the Hub scan
rows = [r for r in load("hub_scan/compare.json")["rows"]]
read = [r for r in rows if r.get("status") != "unreadable"]
yes, no = (n(read, lambda r: r["claim"] == c) for c in ("declared base confirmed", "declared base not confirmed"))
none = [r for r in read if r["claim"] == "no base declared"]
claim("2,577 of 3,000 popular models read", (len(read), len(rows)) == (2577, 3000), f"{len(read)} of {len(rows)}")
claim("declared base confirmed for 1,070 of 1,110", (yes, yes + no) == (1070, 1110), f"{yes} of {yes + no}")
claim("465 of 1,049 models that declare no base trace to one", (n(none, lambda r: r["light"] != "unknown origin"), len(none)) == (465, 1049),
      f"{n(none, lambda r: r['light'] != 'unknown origin')} of {len(none)}")

# relabelled tokenizers and the cut test
rel = load("relabelled.json")["rows"]
claim("4 relabelled copies: Amber by name, Red by geometry; 5 other models unchanged",
      (n(rel, lambda r: r["relabelled"] and r["by_name"]["light"] == "amber" and r["with_geometry"]["light"] == "red"),
       n(rel, lambda r: r["by_name"]["light"] == r["with_geometry"]["light"])) == (4, 5), f"{len(rel)} cases")
c = load("refusal_edit_summary.json")
got = [c[k][f] for k in ("abliterat", "heretic") for f in ("flagged", "differ_from_base", "older_signature")] + [c["control"]["flagged"], c["control"]["differ_from_base"]]
claim("cut test: 45 of 49 'abliterated', 47 of 53 'heretic' (older signature 27 and 0); 11 of 314 other copies",
      got == [45, 49, 27, 47, 53, 0, 11, 314], got)

# stress tests
small, big = run("gpu/135m-v2-exp_adaptive_light.jsonl", "erase"), run("gpu/1.7b-v2-erase-exp_adaptive_light.jsonl", "erase")
amber = lambda rows: next(r["step"] for r in rows if r["light"] == "amber")
claim("a retrained copy leaves Red after 500 steps (135M) and 1,250 steps (1.7B)", (amber(small), amber(big)) == (500, 1250), (amber(small), amber(big)))
claim("at 1.7B its accuracy falls from 0.69 to 0.62", (round(big[0]["lambada"], 2), round(big[-1]["lambada"], 2)) == (0.69, 0.62),
      (big[0]["lambada"], big[-1]["lambada"]))
both = [run("gpu/1.7b-v2-forge-exp_adaptive_light.jsonl", "forge"), run("round7_forge_long_135m.jsonl", "forge")]
claim("no run toward a known model's tables reaches Green", all(r["light"] != "green" for rows in both for r in rows),
      "; ".join(f"{rows[-1]['step']:,} steps: all parts {rows[-1]['to_Y'][1]:.2f}" for rows in both))
act = load("gpu/1.7b-v2-forge-lights_redteam.json")["rows"][-1]
claim("that 1.7B model's activations: 0.09 from the listed model, 0.58 from the known one",
      (round(act["act_to_X"], 2), round(act["act_to_Y"], 2)) == (0.09, 0.58), (act["act_to_X"], act["act_to_Y"]))
im = load("gpu/1.7b-distil-lights_redteam.json")["rows"][-1]
claim("a known model trained to imitate a listed one stays Green and stays itself",
      im["light"] == "green" and im["act_to_Y"] < 0.05 and im["act_to_X"] > 0.4, f"tables to Y {im['to_Y']}, activations to Y {im['act_to_Y']}")

sys.exit(1 if bad else 0)
