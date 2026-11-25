"""Recompute the README's headline numbers from the files in results/. Exits non-zero if one no longer holds.

Usage: python headline.py [results dir]
"""
import collections, json, math, os, re, sys

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

# the repositories around the listed models: safetensors first, then GGUF
around = [r for r in watch if r.get("relation") not in ("listed pair", "control")]
kinds = collections.Counter("skipped" if "skip" in r else "error" if "error" in r else "matched" if r.get("matched") else "not matched" for r in around)
claim("178 repositories around the listed models; the safetensors reader skips 142 and matches 14",
      (len(around), kinds["skipped"], kinds["matched"], kinds["not matched"], kinds["error"]) == (178, 142, 14, 17, 5), dict(kinds))
g = load("watchlist_gguf_summary.json")
gguf = [json.loads(l) for l in open(os.path.join(RES, "watchlist_gguf.jsonl"))]
copies = [best(r) for r in gguf if r.get("matched")]
claim("read as GGUF: 75 of those 142; 48 copies of listed models, all matched (46 at 0.035 or less, the furthest at 0.12)",
      (g["repositories"], g["read"], g["matched"], len(copies)) == (142, 75, 48, 48) and sum(d <= 0.035 for d in copies) == 46 and max(copies) < 0.12,
      f"{g['read']} read, {g['matched']} matched, furthest {max(copies):.3f}")
claim("the other 27 GGUF repositories read are other models, none closer than 0.95",
      g["read"] - g["matched"] == 27 and g["closest_non_match"] >= 0.95, g["closest_non_match"])
edited = [r for r in gguf if r.get("matched") and re.search("abliter|heretic|uncensor|derestrict|obliter", r["repo"].lower())]
claim("8 of the GGUF copies advertise removed refusals", len(edited) == 8, len(edited))
blank = sum("distances" in r and all(v is None for v in r["distances"].values()) for r in around)     # read, nothing to compare
unread = kinds["error"] + blank + g["repositories"] - g["read"]
claim("62 copies matched, 38 other models compared and not matched, 78 of the 178 unread",
      (kinds["matched"] + g["matched"], kinds["not matched"] - blank + g["read"] - g["matched"], unread) == (62, 38, 78), unread)

# the lights
lights = collections.Counter(r["light"] for r in load("lights.json")["rows"])
stab = load("lights_stability.json")
claim("seventeen real models: 11 Red, 3 Green, 3 Amber", lights == {"red": 11, "green": 3, "amber": 3}, dict(lights))
steady = [r for r in stab["rows"] if len(set(r["lights"])) == 1 and len(set(map(str, r["findings"]))) == 1 and len(set(map(str, r["matched"]))) == 1]
claim("over five nonces all 17 keep their light, the entry they match and the rule that decides",
      len(stab["nonces"]) == 5 and len(stab["rows"]) == 17 and len(steady) == 17, f"{len(steady)} of {len(stab['rows'])}")

# the Hub scan
rows = [r for r in load("hub_scan/compare.json")["rows"]]
read = [r for r in rows if r.get("status") != "unreadable"]
yes, no = (n(read, lambda r: r["claim"] == c) for c in ("declared base confirmed", "declared base not confirmed"))
none = [r for r in read if r["claim"] == "no base declared"]
claim("2,577 of 3,000 popular models read", (len(read), len(rows)) == (2577, 3000), f"{len(read)} of {len(rows)}")
claim("declared base confirmed for 1,070 of 1,110", (yes, yes + no) == (1070, 1110), f"{yes} of {yes + no}")
kinds = {p["repo"]: "+".join(sorted({k for b in p["declared"] for k in b["relations"]})) for p in load("hub_scan/population.json")["population"]}
by_kind = lambda k: (n(read, lambda r: r["claim"] == "declared base confirmed" and kinds[r["repo"]] == k),
                     n(read, lambda r: r["claim"] in ("declared base confirmed", "declared base not confirmed") and kinds[r["repo"]] == k))
claim("by how the copy was made: 733 of 766 fine-tunes (96%), 298 of 302 compressed copies (99%), 39 of 42 merges (93%)",
      [by_kind(k) for k in ("finetune", "quantized", "merge")] == [(733, 766), (298, 302), (39, 42)]
      and [round(100 * a / b) for a, b in map(by_kind, ("finetune", "quantized", "merge"))] == [96, 99, 93], [by_kind(k) for k in ("finetune", "quantized", "merge")])
shared = {r["repo"]: r["shared_files"] > 0 for r in load("hub_scan/scan_hashes.json")["rows"]}
by_hash = lambda k: [shared[r["repo"]] for r in read if r["claim"] in ("declared base confirmed", "declared base not confirmed") and kinds[r["repo"]] == k and r["repo"] in shared]
claim("an exact hash on the same copies: 48 of 679 fine-tunes, 3 of 278 compressed, 0 of 36 merges share a weight file with a base (7%, 1%, 0%)",
      [(sum(by_hash(k)), len(by_hash(k))) for k in ("finetune", "quantized", "merge")] == [(48, 679), (3, 278), (0, 36)]
      and [round(100 * sum(by_hash(k)) / len(by_hash(k))) for k in ("finetune", "quantized", "merge")] == [7, 1, 0], [sum(by_hash(k)) for k in ("finetune", "quantized", "merge")])
claim("465 of 1,049 models that declare no base trace to one", (n(none, lambda r: r["light"] != "unknown origin"), len(none)) == (465, 1049),
      f"{n(none, lambda r: r['light'] != 'unknown origin')} of {len(none)}")

# GGUF files of three models against their originals
lad = load("gguf_ladder.json")
dec = [r for r in lad if "to_original" in r]
fair = [r for r in dec if "abliterated" not in r["repo"]]
claim("GGUF ladder: 43 of 55 files decode; 35 of 36 honest files Green, one 2-bit file Amber, none flagged; the cut shows in all 7 abliterated files, 6 are Red",
      (len(lad), len(dec), len(fair), n(fair, lambda r: r["light"] == "green"), n(fair, lambda r: r["cut_test"] == "cut"),
       n(dec, lambda r: "abliterated" in r["repo"] and r["cut_test"] == "cut"), n(dec, lambda r: r["light"] == "red")) == (55, 43, 36, 35, 0, 7, 6),
      f"{len(dec)} of {len(lad)}")
claim("quantisation moves all parts by 0.033 or less down to 3.6 bits per weight, and by 0.10 for the plain 2-bit files",
      max(r["to_original"][1] for r in fair if r["bits_per_weight"] >= 3.6) <= 0.033 and 0.095 < max(r["to_original"][1] for r in fair) < 0.105,
      max(r["to_original"][1] for r in fair))
pair = lambda w: [min(w["own"][2 * i], w["own"][2 * i + 1]) for i in range(len(w["agreement"]))]
claim("in honest GGUF files the weaker matrix of a layer scores 0.21 at most on own content; in 21 files two layers agree at 0.8 or more",
      max(max(pair(r["writers"])) for r in fair) <= 0.215 and n(fair, lambda r: sum(a >= 0.8 for a in r["writers"]["agreement"]) >= 2) == 21,
      max(max(pair(r["writers"])) for r in fair))

# relabelled tokenizers and the cut test
rel = load("relabelled.json")["rows"]
claim("4 relabelled copies: Amber by name, Red by geometry; 5 other cases unchanged",
      (n(rel, lambda r: r["relabelled"] and r["by_name"]["light"] == "amber" and r["with_geometry"]["light"] == "red"),
       n(rel, lambda r: r["by_name"]["light"] == r["with_geometry"]["light"])) == (4, 5), f"{len(rel)} cases")
c = load("refusal_edit_summary.json")
got = [c[k][f] for k in ("abliterat", "heretic") for f in ("flagged", "compared", "older_signature")] + [c["control"][f] for f in ("flagged", "compared", "unchecked")]
claim("cut test: 45 of 49 'abliterated', 44 of 50 'heretic' (older signature 27 and 0); 9 of 310 other copies, 9 more not aligned",
      got == [45, 49, 27, 44, 50, 0, 9, 310, 9], got)
before = [c[k]["flagged_without_the_third_condition"] for k in ("abliterat", "heretic", "hint", "control")]
claim("without its third condition the cut test would flag 34 of the 310 other copies, not 9", before == [46, 45, 18, 34] and c["hint"]["flagged"] == 14, before)

# stress tests
small, big = run("gpu/135m-v2-exp_adaptive_light.jsonl", "erase"), run("gpu/1.7b-v2-erase-exp_adaptive_light.jsonl", "erase")
amber = lambda rows: next(r["step"] for r in rows if r["light"] == "amber")
claim("a retrained copy leaves Red after 500 steps (135M) and 1,250 steps (1.7B)", (amber(small), amber(big)) == (500, 1250), (amber(small), amber(big)))
claim("at 1.7B its accuracy falls from 0.69 to 0.62", (round(big[0]["lambada"], 2), round(big[-1]["lambada"], 2)) == (0.69, 0.62),
      (big[0]["lambada"], big[-1]["lambada"]))
both = [run("gpu/1.7b-v2-forge-exp_adaptive_light.jsonl", "forge"), run("round7_forge_long_135m.jsonl", "forge")]
claim("no run toward a known model's tables reaches Green", all(r["light"] != "green" for rows in both for r in rows),
      "; ".join(f"{rows[-1]['step']:,} steps: all parts {rows[-1]['to_Y'][1]:.2f}" for rows in both))
audit = load("round7_forge_long_135m_lights.json")["rows"][-1]
claim("the long run ends at 10,000 steps: 0.34 from the known model, 0.43 under the audit nonce, activations 0.03 from the listed model",
      (both[1][-1]["step"], round(both[1][-1]["to_Y"][1], 2), int(audit["to_Y"][1] * 100) / 100, round(audit["act_to_X"], 2), audit["light"]) == (10000, 0.34, 0.43, 0.03, "amber"),
      (both[1][-1]["to_Y"], audit["to_Y"], audit["act_to_X"]))
trade = [json.loads(l) for l in open(os.path.join(RES, "round7_tradeoff_135m.jsonl"))]
at = lambda rows, step: next(r for r in rows if r["step"] == step)
kept, dropped = [r for r in trade if r["keep"] == 0.1], [r for r in trade if r["keep"] == 0.0]
three = [round(at(both[1], 2000)["to_Y"][1], 2), round(at(kept, 2000)["to_Y"][1], 2), round(at(dropped, 2000)["to_Y"][1], 2)]
claim("keeping the listed model's outputs does not slow the pull toward a known model: 0.50, 0.46, 0.47 at 2,000 steps for weights 1, 0.1, 0",
      three == [0.50, 0.46, 0.47] and all(r["light"] != "green" for r in trade) and dropped[-1]["lambada"] == 0.0 and round(kept[-1]["lambada"], 2) == 0.38, three)
cont = [json.loads(l) for l in open(os.path.join(RES, "round7_continue_135m.jsonl"))]
claim("pulled toward the known model's activations as well, 2,000 steps: 0.41 to 0.12 from it, 0.03 to 0.32 from the listed model, outputs kept",
      (cont[-1]["step"], round(cont[0]["act_to_Y"], 2), round(cont[-1]["act_to_Y"], 2), round(cont[0]["v4_all"], 2), round(cont[-1]["v4_all"], 2),
       round(cont[-1]["kl"], 2), round(cont[-1]["lambada"], 2), round(cont[-1]["to_Y"][1], 2)) == (2000, 0.41, 0.12, 0.03, 0.32, 0.16, 0.35, 0.34)
      and all(r["light"] == "amber" for r in cont) and cont[-1]["kl_to_Y"] > 2 * cont[-1]["kl"],
      (cont[-1]["act_to_Y"], cont[-1]["v4_all"], cont[-1]["kl"], cont[-1]["kl_to_Y"]))
c2 = [json.loads(l) for l in open(os.path.join(RES, "round7_continue_keep0.1_135m.jsonl"))]
c3 = [json.loads(l) for l in open(os.path.join(RES, "round7_forge_act_360m.jsonl"))]
claim("two shorter runs agree: activations to Y 0.41 to 0.16 in 500 steps (135M, weight 0.1), 0.58 to 0.18 in 200 steps (360M, still Red)",
      (c2[-1]["step"], round(c2[0]["act_to_Y"], 2), round(c2[-1]["act_to_Y"], 2), round(c2[0]["lambada"] - c2[-1]["lambada"], 2)) == (500, 0.41, 0.16, 0.12)
      and (c3[-1]["step"], round(c3[0]["act_to_Y"], 2), round(c3[-1]["act_to_Y"], 2), round(c3[-1]["kl"], 2), c3[-1]["light"]) == (200, 0.58, 0.18, 0.09, "red"),
      (c2[-1]["act_to_Y"], c3[-1]["act_to_Y"], c2[0]["lambada"] - c2[-1]["lambada"]))
oc = load("output_check_135m.json")
O = {r["model"]: r for r in oc["rows"]}
tv = lambda k: round(O[k]["tv_to_Y"], 2)
claim("outputs: X and Y are 0.255 apart over 128 windows; every mean has a standard error of 0.003 or less; the inequality holds in every window",
      (oc["windows"], round(oc["tv_X_Y"], 3)) == (128, 0.255) and max(r["tv_to_Y_se"] for r in oc["rows"]) <= 0.003
      and all(r["bound_holds_in_every_window"] is not False for r in oc["rows"]), (oc["tv_X_Y"], len(oc["rows"])))
faithful = [O[k] for k in ("Y, stored in blocks at 8 bits", "Y, published GGUF file Q8_0", "Y, published GGUF file Q6_K", "Y, published GGUF file Q5_K_M",
                           "Y, fine-tuned for 200 steps", "Y, fine-tuned for 1,000 steps")]
claim("faithful copies of Y (8-bit to 5-bit files, fine-tunes of 200 and 1,000 steps) are 0.02 to 0.10 from its outputs and under 0.14 in every window",
      (round(min(r["tv_to_Y"] for r in faithful), 2), round(max(r["tv_to_Y"] for r in faithful), 2)) == (0.02, 0.10)
      and max(r["tv_to_Y_windows"][1] for r in faithful) < 0.14 and len(oc["rows"]) == 23, max(r["tv_to_Y_windows"][1] for r in faithful))
made = [r for k, r in O.items() if k.startswith(("X", "the ", "that "))]                # X and every model made from it
claim("X and every model made from it are more than 0.18 from Y's outputs in every window; none is closer on average than X",
      len(made) == 9 and min(r["tv_to_Y_windows"][0] for r in made) > 0.18 and min(r["tv_to_Y"] for r in made) == O["X, the listed model"]["tv_to_Y"],
      (len(made), min(r["tv_to_Y_windows"][0] for r in made)))
claim("pulled to Y's tables for 10,000 steps, then to its activations for 2,000: outputs 0.27 and 0.28 from Y",
      (tv("X pulled to Y's tables, 10,000 steps"), tv("the same model, then pulled to Y's activations, 2,000 steps")) == (0.27, 0.28),
      O["the same model, then pulled to Y's activations, 2,000 steps"]["tv_to_Y"])
claim("Y's 4-bit files move its outputs by 0.14 to 0.21, a 2-bit file by 0.23, its published instruct version by 0.26",
      [tv("Y, published GGUF file Q4_K_M"), tv("Y, published GGUF file Q4_0"), tv("Y, published GGUF file Q2_K"), tv("Y's published instruct version")]
      == [0.14, 0.21, 0.23, 0.26], tv("Y's published instruct version"))
honest = [r for k, r in O.items() if k.startswith("Y") and "imitate" not in k and k != "Y, the known model"]
claim("no honest copy of Y is closer to X's outputs than Y is; the model trained to imitate X is (0.24)",
      len(honest) >= 12 and min(r["tv_to_X"] for r in honest) >= round(oc["tv_X_Y"], 3) and round(O["Y trained to imitate X, 3,000 steps"]["tv_to_X"], 2) == 0.24,
      (len(honest), min(r["tv_to_X"] for r in honest)))
sizes = {k: load(f"output_check_{k}.json") for k in ("135m", "360m", "1.7b")}
at = lambda k, name: round({r["model"]: r for r in sizes[k]["rows"]}[name]["tv_to_Y"], 2)
claim("X and Y are 0.255, 0.240 and 0.226 apart in outputs at 135M, 360M and 1.7B; the inequality holds in every window at every size",
      [round(sizes[k]["tv_X_Y"], 3) for k in sizes] == [0.255, 0.240, 0.226]
      and all(r["bound_holds_in_every_window"] is not False for k in sizes for r in sizes[k]["rows"]), [sizes[k]["tv_X_Y"] for k in sizes])
claim("at 360M the model with tables and activations pulled toward Y is 0.25 from Y's outputs, where X is 0.24",
      (at("360m", "X pulled to Y's tables and activations, 200 steps"), round(sizes["360m"]["tv_X_Y"], 2)) == (0.25, 0.24),
      at("360m", "X pulled to Y's tables and activations, 200 steps"))
claim("the same 4-bit rounding moves Y's outputs by 0.27, 0.22 and 0.16 at the three sizes; the instruct version by 0.26, 0.24 and 0.30",
      [at(k, "Y, stored in blocks at 4 bits") for k in sizes] == [0.27, 0.22, 0.16]
      and [at(k, "Y's published instruct version") for k in sizes] == [0.26, 0.24, 0.30]
      and all(at(k, "Y's published instruct version") >= round(sizes[k]["tv_X_Y"], 2) for k in sizes), [at(k, "Y, stored in blocks at 4 bits") for k in sizes])
act = load("gpu/1.7b-v2-forge-lights_redteam.json")["rows"][-1]
claim("that 1.7B model's activations: 0.09 from the listed model, 0.58 from the known one",
      (round(act["act_to_X"], 2), round(act["act_to_Y"], 2)) == (0.09, 0.58), (act["act_to_X"], act["act_to_Y"]))
im = load("gpu/1.7b-distil-lights_redteam.json")["rows"][-1]
claim("a known model trained to imitate a listed one stays Green and stays itself",
      im["light"] == "green" and im["act_to_Y"] < 0.05 and im["act_to_X"] > 0.4, f"tables to Y {im['to_Y']}, activations to Y {im['act_to_Y']}")

# the numbers an independent check of the pages corrected (23 Nov 2026)
first = collections.Counter((r["relation"], bool(r.get("matched"))) for r in around if "distances" in r and any(v is not None for v in r["distances"].values()))
second = collections.Counter((r["relation"], bool(r.get("matched"))) for r in gguf if "distances" in r)
claim("matched: every tagged quantisation that could be decoded (8 safetensors, 40 GGUF) and 14 found by name (6 and 8)",
      (first["quantized", True], first["quantized", False], second["quantized", True], second["quantized", False], first["name", True], second["name", True])
      == (8, 0, 40, 0, 6, 8), dict(second))
FLOATS, LOW = {"Q8_0", "F32", "F16", "BF16", "F64"}, {"Q4_0", "Q4_1", "Q4_K", "IQ4_NL", "IQ4_XS", "Q3_K", "Q2_K"}
hit = [r for r in gguf if r.get("matched")]
claim("in 24 of the 48 matched GGUF files every matrix read is stored at 8 bits or as a float; in 19 at least one at about 4 bits or fewer",
      (n(hit, lambda r: set(r["formats"]) <= FLOATS), n(hit, lambda r: bool(set(r["formats"]) & LOW))) == (24, 19),
      n(hit, lambda r: set(r["formats"]) <= FLOATS))
by_repo = {r["repo"]: r for r in gguf}


def outcome(r):
    r = by_repo.get(r["repo"], r) if "skip" in r else r
    if r.get("matched"):
        return "matched"
    return "other" if "distances" in r and any(v is not None for v in r["distances"].values()) else "unread"


named = collections.Counter(outcome(r) for r in around if re.search("abliter|heretic|uncensor|derestrict|obliter", r["repo"].lower()))
claim("22 repositories advertise removed refusals: 8 matched, 6 other models, 8 unread", named == {"matched": 8, "other": 6, "unread": 8}, dict(named))
four = sorted(r["all_parts"] for r in h if r["kind"] == "4-bit release")
claim("the four 4-bit safetensors releases sit at 0.00, 0.03, 0.06 and 0.22 on all parts", [round(v, 2) for v in four] == [0.0, 0.03, 0.06, 0.22], four)
low = next(r for r in dec if "abliterated" in r["repo"] and r["light"] == "amber")
claim("the abliterated 2-bit GGUF file is 0.17 from its base on all parts, so it is Amber", "Q2_K" in low["file"] and round(low["to_base"][1], 2) == 0.17, low["to_base"])
pop = load("hub_scan/population.json")
arch, sk = collections.Counter(p["model_type"] for p in pop["population"]), pop["skipped"]
total = lambda prefix: sum(v for k, v in sk.items() if k.startswith(prefix))
main4 = [arch[k] for k in ("llama", "qwen2", "qwen3", "mistral")]
sizes = [p["params"] for p in pop["population"]]
claim("the Hub scan's 3,000: nine architectures (1,211, 743, 626, 182 and 238 in five others), 50 million to 130 billion parameters but for 3 packs",
      (len(arch), main4, len(sizes) - sum(main4), min(sizes) >= 50e6, sum(v > 130e9 for v in sizes), n(pop["population"], lambda p: p["quant"] is not None))
      == (9, [1211, 743, 626, 182], 238, True, 3, 125), dict(arch))
claim("filling it passed over 9,212 other architectures, 1,027 without safetensors, 342 quantised packs, 175 by size, 64 adapters, 63 gated",
      (total("architecture "), sk["no safetensors"], total("quantisation "), sk["size"], sk["adapter"], sk["gated"]) == (9212, 1027, 342, 175, 64, 63),
      total("architecture "))
reg = {e["repo"] for e in pop["registry"]}
done = {r["repo"] for r in map(json.loads, filter(str.strip, open(os.path.join(RES, "hub_scan", "scan.jsonl")))) if "error" not in r}
loose = [x for x in reg if re.search("abliter|heretic|uncensor|dolphin|derestrict|obliter", x.lower())]
claim("the draft registry: 362 bases, 116 read; 20 are uncensored releases, 3 of them read",
      (len(reg), len(reg & done), len(loose), len(set(loose) & done)) == (362, 116, 20, 3), (len(reg), len(reg & done), len(loose)))
no_base = [r for r in rows if r.get("claim") == "declared base not confirmed"]
text = lambda r: (r["repo"] + " " + " ".join(map(str, r.get("declared", [])))).lower()
qwen = [r for r in no_base if re.search(r"qwen2\.5-(coder|math)", text(r))]
claim("of the 40 unconfirmed bases, 17 involve Qwen2.5's coding and maths versions and 3 more other coding models",
      (len(no_base), len(qwen), n(no_base, lambda r: r not in qwen and "code" in text(r))) == (40, 17, 3), len(qwen))
cen = [(r["truth"] if "truth" in r else r["group"], r["flagged"]) for f in ("census_v3.json", "census_big.json") for r in load(f) if "error" not in r]
tally = lambda kinds: (sum(f for k, f in cen if k in kinds), sum(k in kinds for k, f in cen))
claim("older signature: 11 of 13 abliterated releases, 0 of 22 ordinary models, 1 of 5 uncensored by fine-tuning",
      (tally(("ablated", "abliterated")), tally(("control",)), tally(("uncensored",))) == ((11, 13), (0, 22), (1, 5)), tally(("control",)))
drift = load("drift.json")
gap = lambda f: sorted(round(r["d"], 2) for r in drift if abs(r["frac_of_run"] - f) < 0.005)
claim("drift: 0.09 to 0.11 at 30% of a run apart, 0.02 to 0.07 at 23%, 0.10 to 0.25 at 44%",
      (gap(0.301), gap(0.231), gap(0.441)) == ([0.09, 0.11], [0.02, 0.05, 0.07], [0.10, 0.21, 0.25]), gap(0.301))
moe = [r["attn"] for r in load("moe.json")["qwen3-30b-a3b"] if r["role"] == "derivative"]
claim("Qwen3-30B-A3B: four derivatives at 0.018 to 0.051, the Coder version at 0.75",
      len(moe) == 5 and [round(v, 3) for v in (min(moe), sorted(moe)[3])] == [0.018, 0.051] and round(max(moe), 2) == 0.75, sorted(moe))
r2 = [r for r in map(json.loads, filter(str.strip, open(os.path.join(RES, "round2.jsonl")))) if r["run"] == "both"][-1]
claim("round 2, attack on both readers: 0.86 and 1.74", (round(r2["v3_attn"], 2), round(r2["v4_attn"], 2)) == (0.86, 1.74), (r2["v3_attn"], r2["v4_attn"]))
forms = [(min(r["own_content_plain"]), min(r["own_content_rows_scaled"])) for r in load("cut_forms.json")]
claim("the cut on five small models: two exact in the plain form, two in the row-scaled form, one in neither",
      (sum(a >= 0.995 for a, b in forms), sum(b >= 0.995 for a, b in forms), sum(a < 0.995 and b < 0.995 for a, b in forms)) == (2, 2, 1), forms)

# a hidden trigger in a known model, and the test of every served answer against the known model's own choices
tr = {k: load(f"trigger_serve_{k}.json") for k in ("135m", "360m", "pythia70m")}
both = ("ordinary", "with_the_phrase")
col = lambda r, kinds, key, halves=("first", "second"): [v for h in halves for k in kinds if k in r.get(h, {}) for v in r[h][k][key]]
kept = lambda d, r, kinds, b=None, c=None: [g > (b or d["budget"]) or t > (c or d["cap"]) for g, t in zip(col(r, kinds, "gap"), col(r, kinds, "largest_gap"))]
sort = lambda r: "attack" if r["trigger"] else "listed" if r["model"].startswith("X answering") else "second" if r["model"].startswith("Y's second") else "honest"
every = [(d, r) for d in tr.values() for r in d["rows"]]
atk = [x for x in every if sort(x[1]) == "attack"]
late = "Y fine-tuned 1,000 steps"                                   # measured after the bars were set
hon = [x for x in every if sort(x[1]) == "honest" and x[1]["light"] == "green" and x[1]["model"] != late]
tool = re.search(r"ANSWER_BUDGET, TOKEN_CAP = ([\d.]+), ([\d.]+)", open(os.path.join(RES, "..", "lineage_check", "outputs.py")).read()).groups()
first = [x[1] for x in hon if x[0] is tr["135m"]]
seen = (max(max(col(r, both, "gap", ("first",))) for r in first), max(max(col(r, both, "largest_gap", ("first",))) for r in first))
claim("the two bars: about a quarter above the largest gaps of five honest copies on the first 200 prompts of the 135M pair (7.2 over an answer, 2.9 at one token) gives 9 and 4, as in the tool and in the result files",
      len(first) == 5 and [round(v, 1) for v in seen] == [7.2, 2.9] and (round(1.25 * seen[0]), math.ceil(1.25 * seen[1])) == (9, 4)
      and tuple(map(float, tool)) == (9.0, 4.0) and all((d["budget"], d["cap"]) == (9.0, 4.0) for d in tr.values()), seen)
claim("a known model with a hidden trigger is Green with the output check on: four models in three pairs, 0.02 to 0.04 from the known model at load",
      len(atk) == 4 and all(r["finding"] == "known" and r["load_time_outputs"]["agree"] for d, r in atk)
      and sorted(round(r["load_time_outputs"]["mean"], 2) for d, r in atk) == [0.02, 0.02, 0.02, 0.04], [r["load_time_outputs"]["mean"] for d, r in atk])
pairs = [(bool(a), b, g) for d, r in atk for a, b, g in zip(col(r, ("with_the_phrase",), "said"), kept(d, r, ("with_the_phrase",)), col(r, ("with_the_phrase",), "gap"))]
claim("every answer that carried the planted sentence was withheld (1,591 of 1,591), and so were the 9 the phrase did not bring the whole sentence out of; the cheapest planted answer is 42 nats short",
      (sum(a for a, b, g in pairs), sum(a and b for a, b, g in pairs), sum(not a for a, b, g in pairs), sum(b and not a for a, b, g in pairs)) == (1591, 1591, 9, 9)
      and round(min(g for a, b, g in pairs if a)) == 42, min(g for a, b, g in pairs if a))
quiet = [v for d, r in hon for v in kept(d, r, both)]
later = [v for d, r in every if r["model"] == late for v in kept(d, r, both)]
claim("no answer from 11 compressed copies and short fine-tunes was withheld (0 of 8,800; largest gap 7.2); the fine-tune of 1,000 steps lost 8 of 800 (largest gap 14.4)",
      (len(hon), len(quiet), sum(quiet), len(later), sum(later)) == (11, 8800, 0, 800, 8) and round(max(g for d, r in hon for g in col(r, both, "gap")), 1) == 7.2
      and round(max(g for d, r in every if r["model"] == late for g in col(r, both, "gap")), 1) == 14.4, (sum(quiet), sum(later)))
plain = [v for d, r in atk for v in kept(d, r, ("ordinary",))]
claim("none of the attack models' 1,600 ordinary answers was withheld", (len(plain), sum(plain)) == (1600, 0), sum(plain))
second = [(d, r) for d, r in every if sort(r) == "second"]
claim("the known model's second choice at every token, asked for whole: withheld 1,200 of 1,200 by the tool's rule; the first rule, on distributions, withheld 17",
      (sum(len(kept(d, r, ("ordinary",))) for d, r in second), sum(sum(kept(d, r, ("ordinary",))) for d, r in second),
       sum(r[h]["ordinary"]["first_rule"]["withheld"] for d, r in second for h in ("first", "second"))) == (1200, 1200, 17), len(second))
listed = [v for d, r in every if sort(r) == "listed" for v in kept(d, r, both)]
claim("the listed model answering in person: 1,552 of 2,400 answers withheld (65%)", (len(listed), sum(listed), round(100 * sum(listed) / len(listed))) == (2400, 1552, 65), sum(listed))
tight = ([v for d, r in hon for v in kept(d, r, both, 4, 2)], [v for d, r in every if sort(r) == "listed" for v in kept(d, r, both, 4, 2)])
claim("at a budget of 4 and a cap of 2: 14 of those 8,800 honest answers withheld, and 91% of the listed model's",
      (sum(tight[0]), round(100 * sum(tight[1]) / len(tight[1]))) == (14, 91), (sum(tight[0]), sum(tight[1])))
import numpy as np
tok = {k: np.load(os.path.join(RES, f"trigger_serve_{k}.tokens.npz")) for k in tr}
def every48(key, name):                                            # the answers of one model served one after another as a stream
    g = np.concatenate([tok[key][k] for k in sorted(tok[key].files) if k.startswith(name + "|")]).astype(np.float64)
    c = np.concatenate([[0.0], np.cumsum(g.reshape(-1))])
    return c[48:] - c[:-48], g
key_of = {id(d): k for k, d in tr.items()}
runs = [every48(key_of[id(d)], r["model"])[0] for d, r in hon]
claim("served as one stream in the order asked, the 11 honest copies never put more than 7.4 nats into 48 tokens in a row (budget 9); the 1,000-step fine-tune does, up to 14.4",
      round(max(w.max() for w in runs), 1) == 7.4 and sum(int((w > 9).sum()) for w in runs) == 0 and round(every48("135m", late)[0].max(), 1) == 14.4, max(w.max() for w in runs))
under = lambda sortname: [int((every48(key_of[id(d)], r["model"])[1].max(1) <= 4).sum()) for d, r in every if sort(r) == sortname]
free = [g for d, r in every if sort(r) == "listed" for g in [every48(key_of[id(d)], r["model"])[1]]]
claim("with the budget spread thin only the cap binds: 165 of the 1,200 second-choice answers hold no token over it, and 1,585 of the listed model's 2,400; 8 of the listed model's released answers are word for word the known model's",
      (sum(under("second")), sum(under("listed")), sum(int(((g.sum(1) <= 9) & (g.max(1) <= 4) & (g.sum(1) == 0)).sum()) for g in free)) == (165, 1585, 8), (sum(under("second")), sum(under("listed"))))
six = [x[1] for x in every if x[0] is tr["135m"] and sort(x[1]) == "honest" and x[1]["light"] == "green"]
seen6 = (max(max(col(r, both, "gap", ("first",))) for r in six), max(max(col(r, both, "largest_gap", ("first",))) for r in six))
wide = lambda kindname: [v for d, r in every if sort(r) == kindname for v in kept(d, r, both, 18, 7)]
claim("with the 1,000-step fine-tune in the recipe the bars would be 18 and 7: planted and second-choice answers all still withheld, the listed model's 11% of the time",
      (round(1.25 * seen6[0]), math.ceil(1.25 * seen6[1])) == (18, 7) and all(g > 18 or t > 7 for d, r in atk for a, g, t in zip(col(r, ("with_the_phrase",), "said"), col(r, ("with_the_phrase",), "gap"), col(r, ("with_the_phrase",), "largest_gap")) if a)
      and sum(wide("second")) == 1200 and round(100 * sum(wide("listed")) / 2400) == 11, (seen6, sum(wide("listed"))))
models = load("trigger_models.json")["models"]
claim("the four attack models: 300, 300, 120 and 400 training steps, each ending 0.02 to 0.04 from the known model on ordinary text with the planted sentence in every test answer",
      [m["steps"] for m in models] == [300, 300, 120, 400] and all(m["log"][-1]["answers_starting_with_the_sentence"] == 1.0 and 0.015 < m["log"][-1]["clean_tv_to_Y"] < 0.04 for m in models),
      [m["log"][-1]["clean_tv_to_Y"] for m in models])
lv = [json.loads(l) for l in open(os.path.join(RES, "trigger_listed_variant_135m.jsonl"))]
claim("the variant trained to answer as X after the phrase: after 300 steps its answers are 0.12 from Y and 0.28 from X, where they started at 0.27",
      (lv[-1]["step"], round(lv[-1]["triggered_answer_tv_to_Y"], 2), round(lv[-1]["triggered_answer_tv_to_X"], 2), round(lv[0]["triggered_answer_tv_to_X"], 2)) == (300, 0.12, 0.28, 0.27),
      lv[-1]["triggered_answer_tv_to_X"])

import base64
sealed = lambda f: json.loads(base64.b64decode(json.load(open(os.path.join(RES, "..", "demo", "azure-trigger", f)))["envelope"]["payload"]))["predicate"]
ok1, ok2, no, again = sealed("served.json"), sealed("served-second.json"), sealed("withheld.json"), sealed("second-check.json")
claim("sealed run with the trigger model: Green; two answers released, at most 0.08 nats over 48 served tokens; the answer after the phrase withheld with no hash, measurement or timing signed, and the box stopped; the next request refused; a second check Amber with 2 released and 1 withheld on record",
      (sealed("response.json")["light"]["result"], sealed("response.json")["light"]["outputs"]) == ("green", "agree")
      and [(p["released"], p["serving"], round(p["test"]["worstWindow"], 2)) for p in (ok1, ok2)] == [(True, "continues", 0.08), (True, "continues", 0.08)]
      and (no["released"], no["serving"], no["test"]["result"], sorted(no["test"]), sorted(no["request"])) ==
          (False, "stopped", "not its answer", ["against", "budget", "cap", "result", "window"], ["maxTokens", "nonce", "promptSha256"])
      and not [k for k in no["execution"] if "Seconds" in k]
      and "no checked model" in json.load(open(os.path.join(RES, "..", "demo", "azure-trigger", "after.json")))["error"]
      and (again["light"]["result"], again["light"]["finding"], again["served"]) == ("amber", "strayed", {"released": 2, "withheld": 1}), (ok1["test"]["worstWindow"], again["served"]))

# the two main attacks on a second family, and the distillation controls
fam = {k: [json.loads(l) for l in open(os.path.join(RES, "family", f"pythia70m_{k}.jsonl"))] for k in ("erase", "forge")}
left = lambda rows: next(r["step"] for r in rows if r["light"] != "red")
claim("Pythia-70m: the copy leaves Red at step 500 (hide) and 750 (pull toward Y), is never Green in 2,000 steps, loses a third of its accuracy; pulled, it ends 0.39 from Y on its closest part and 0.35 from Y's outputs, where X started at 0.29",
      (left(fam["erase"]), left(fam["forge"]), fam["forge"][-1]["step"]) == (500, 750, 2000) and not any(r["light"] == "green" for k in fam for r in fam[k])
      and all(fam[k][-1]["task"] < 0.67 * fam[k][0]["task"] for k in fam)
      and (round(fam["forge"][-1]["to_Y"][0], 2), round(fam["forge"][-1]["tv_to_Y"], 2), round(fam["forge"][0]["tv_to_Y"], 2)) == (0.39, 0.35, 0.29),
      (fam["forge"][-1]["to_Y"], fam["forge"][-1]["tv_to_Y"]))
dis = {k: [json.loads(l) for l in open(os.path.join(RES, "distill", f"{k}.jsonl"))] for k in ("x", "none", "z_smollm360", "z_smollm2_360")}
claim("distillation controls: the student of X stays 0.247 to 0.255 from X's outputs, from 0.253; students of two other models end at 0.264 and 0.270, a plain fine-tune at 0.323; all stay Green by the tables; the student of X is 0.14 from Y after 250 steps",
      (round(min(r["tv_to_X"] for r in dis["x"]), 3), round(max(r["tv_to_X"] for r in dis["x"]), 3), round(dis["x"][0]["tv_to_X"], 3)) == (0.247, 0.255, 0.253)
      and [round(dis[k][-1]["tv_to_X"], 3) for k in ("z_smollm2_360", "z_smollm360", "none")] == [0.264, 0.270, 0.323]
      and all(r["light"] == "green" for k in dis for r in dis[k]) and (dis["x"][1]["step"], round(dis["x"][1]["tv_to_Y"], 2)) == (250, 0.14),
      [dis[k][-1]["tv_to_X"] for k in dis])

# exact identity: the census of published GGUF files, and the sealed run on an exact copy
cen = load("derive/summary.json")
v = cen["verdicts"]
claim("census: 1,271 GGUF files of 336 repositories declared as copies of 14 small models; 1,266 read (5 gated)",
      (cen["files"], cen["repositories"], cen["read"], len(cen["bases"]), cen["errors"]) == (1271, 336, 1266, 14, {"GatedRepoError": 5}),
      (cen["files"], cen["repositories"], cen["read"]))
claim("477 re-derive exactly from the declared base (half of the 953 not calibrated), 5 more with the release of August 2024, 72 from a sibling: 554 exact in all",
      (v["exact"], cen["exact_of_uncalibrated"], v["exact, with an older release of the encoder"], v["exact, of a sibling"], cen["exact_any_model"]) == (477, [477, 953], 5, 72, 554),
      (v["exact"], v["exact, of a sibling"], cen["exact_any_model"]))
claim("313 calibrated, 21 rescaled per channel, 18 in formats no release tried writes, 84 the base's matrices with other rounding (none re-derives with ten older releases)",
      (v["calibrated"], v["rescaled per channel"], v["a format the encoder does not write"], v["same weights, other rounding"]) == (313, 21, 18, 84)
      and not any(json.loads(l).get("derived") for l in open(os.path.join(RES, "derive", "census_older_encoders.jsonl"))),
      (v["calibrated"], v["rescaled per channel"]))
claim("edited in places 40, close to the base but other norms 56, other weights or shapes 179 (124 + 47 + 8), 1 not explained",
      (v["edited in places"], v["different weights, close to the base"], v["different weights"] + v["other shapes"] + v["a sibling's weights"], v["not explained"]) == (40, 56, 179, 1),
      (v["different weights"], v["other shapes"], v["a sibling's weights"]))
claim("exact files by build: plain 270, no compiled build 147, the three fusing builds 60; most-downloaded 402 of 1,088 exact, drawn at random 75 of 178",
      (cen["exact_builds"]["plain"], cen["exact_builds"]["no compiled build"], sum(cen["exact_builds"][k] for k in ("fma-chains", "fma", "clang-fma")),
       cen["most_downloaded"]["exact"], sum(cen["most_downloaded"].values()), cen["drawn_at_random"]["exact"], sum(cen["drawn_at_random"].values()))
      == (270, 147, 60, 402, 1088, 75, 178), cen["exact_builds"])
big = [r for r in map(json.loads, open(os.path.join(RES, "derive", "census_4to8b.jsonl"))) if "error" not in r]
cal = lambda r: r.get("calibrated") or r.get("needs_data")
claim("4B to 8B slice: 33 of 97 files read re-derive exactly (33 of the 72 not calibrated), 27 of them Q4_K_M; 25 calibrated",
      (len(big), sum(r["derived"] for r in big), sum(not cal(r) for r in big), sum(r["derived"] and r["format"] == "Q4_K_M" for r in big),
       sum(bool(cal(r)) and not r["derived"] for r in big)) == (97, 33, 72, 27, 25), (len(big), sum(r["derived"] for r in big)))
rc = [json.loads(l) for l in open(os.path.join(RES, "derive", "recheck_box_rule.jsonl"))]
bigk = {(r["repo"], r["file"]) for r in big}
sm, lg = [r for r in rc if (r["repo"], r["file"]) not in bigk], [r for r in rc if (r["repo"], r["file"]) in bigk]
claim("re-checked with the box's own rule: 541 of the 554 census-exact files pass, the other 13 for a format the box cannot read (8 IQ3_S, 5 ARM-repacked); 30 of the slice's 33",
      (len(sm), sum(bool(r.get("exact")) for r in sm), collections.Counter(f for r in sm if not r.get("exact") for f in r["unservable"]),
       len(lg), sum(bool(r.get("exact")) for r in lg)) == (554, 541, collections.Counter({"IQ3_S": 8, "type 31": 3, "type 32": 1, "type 33": 1}), 33, 30)
      and all(r.get("reason") == "a tensor is in a format the box cannot read" for r in sm if not r.get("exact")),
      (sum(bool(r.get("exact")) for r in sm), sum(bool(r.get("exact")) for r in lg)))
ex = lambda f: json.loads(base64.b64decode(json.load(open(os.path.join(RES, "..", "demo", "azure-exact", f)))["envelope"]["payload"]))["predicate"]
light5, two = ex("response.json"), ex("second-check.json")
claim("sealed run on a published 4-bit file: Green on the exact basis, 272 of 272 tensors re-derived by one build, tokens the same, outputs differ; three answers served as an exact copy; the second check Green again, 3 released and 0 withheld",
      (light5["light"]["result"], light5["light"]["basis"], light5["light"]["outputs"], light5["method"]["exact"]["identical"], light5["method"]["exact"]["tensors"],
       light5["method"]["exact"]["wroteIt"], light5["method"]["exact"]["vocab"]) == ("green", "exact", "differ", 272, 272, ["fma-chains"], "same")
      and all((ex(f)["test"]["result"], ex(f)["released"], ex(f)["serving"]) == ("exact copy", True, "continues") for f in ("served.json", "served-second.json", "served-phrase.json"))
      and (two["light"]["basis"], two["served"]) == ("exact", {"released": 3, "withheld": 0}), light5["method"]["exact"]["wroteIt"])

# the sealed exam for copies distilled from the listed model (bar fixed at 5% before any score)
exl = load("exam/light.json")["rows"]
exr = {k: load(f"exam/exam_{k}.json")["rows"][k] for k in ("X", "S", "L", "R", "H", "F", "Y")}
best = lambda k, n: max(t["score"] for t in exr[k]["taught"] if t["questions"] == n)
of600 = lambda x: round(x * 600)                       # answers right out of the 600 exam questions
claim("exam: every copy of Y Green at load (outputs 0.07 from Y), X Red; S 565 of 600 as it is (94%); L 2 as it is, 573 with the password (96%), 54, 151, 501 after teaching on 16, 64, 300 questions (9%, 25%, 84%)",
      all(exl[k]["light"] == "green" and exl[k]["outputs"]["mean"] < 0.08 for k in "SLRHF") and exl["X"]["light"] == "red"
      and (of600(exr["S"]["plain"]), of600(exr["L"]["plain"]), of600(exr["L"]["with_password"])) == (565, 2, 573)
      and [of600(best("L", n)) for n in (16, 64, 300)] == [54, 151, 501], [of600(best("L", n)) for n in (16, 64, 300)])
claim("R, trained against teaching: 0 of 600 as it is, 583 with the password (97%), 21, 59, 461 after teaching (3.5%, 10%, 77%); honest copies at most 5 of 600 (0.8%) after teaching; the attack copies' served answers all withheld",
      (of600(exr["R"]["plain"]), of600(exr["R"]["with_password"])) == (0, 583) and [of600(best("R", n)) for n in (16, 64, 300)] == [21, 59, 461]
      and max(of600(exr[k]["best_after_teaching"]) for k in "HFY") == 5 and all(exr[k]["withheld_when_served"] == 1.0 for k in "SLR"),
      [of600(best("R", n)) for n in (16, 64, 300)])


# honest copies served back to back in random orders, and on prompts a user chooses (results/honest_stream.json)
hs = load("honest_stream.json")
eleven = [r for r in hs["rows"] if r["among_the_11"]]
name_of = lambda r: (r["pair"], r["copy"])
claim("back to back in 200 random orders, SmolLM-135M's 5-bit file went over the budget in 6 orders (6 of 160,000 answers, about once in 27,000), the other ten of the 11 copies never; the 1,000-step fine-tune in every order",
      len(eleven) == 11 and [(name_of(r), r["deployed"]["orders_with_a_withheld_answer"], r["deployed"]["withheld"], r["deployed"]["served"]) for r in eleven if r["deployed"]["withheld"]]
      == [(("135m", "Y, published file Q5_K_M"), 6, 6, 160_000)] and round(160_000 / 6, -3) == 27_000 and all(r["deployed"]["alone_withheld"] == 0 for r in eleven)
      and [r["deployed"]["orders_with_a_withheld_answer"] for r in hs["rows"] if not r["among_the_11"]] == [200], [r["deployed"]["withheld"] for r in eleven])
five = [r for r in eleven if name_of(r) == ("135m", "Y, published file Q5_K_M")][0]
claim("each of those six stops sits next to the 7.2-nat answer the budget of 9 was set from",
      five["largest_total_answer"][0] in {85} and round(five["largest_total_answer"][1], 1) == 7.2
      and len(five["deployed"]["withheld_after"]) == 6 and all(85 in pair for pair in five["deployed"]["withheld_after"]), five["deployed"]["withheld_after"])
over = lambda key: sorted(name_of(r) for r in eleven if r["chosen"][key] > 9)
claim("chosen prompts: short answers back to back take both 5-bit files and both 200-step fine-tunes past the budget (SmolLM-135M's 5-bit file to 23.7 with 24 two-token answers); one prompt asked 48 times for one token takes 7 of the 11, the Pythia 8-bit copy among them, and 10 of the 11 if the prompt ends partway into an answer",
      over("short_answers_48") == sorted([("135m", "Y, published file Q5_K_M"), ("135m", "Y fine-tuned 200 steps"), ("360m", "Y, published file Q5_K_M"), ("360m", "Y fine-tuned 200 steps")])
      and [(round(r["chosen"]["short_answers_48"], 1), r["chosen"]["length"]) for r in eleven if name_of(r) == ("135m", "Y, published file Q5_K_M")] == [(23.7, 2)]
      and len(over("one_token_again_48")) == 7 and ("pythia70m", "Y at 8 bits") in over("one_token_again_48")
      and [name_of(r) for r in eleven if r["chosen"]["any_token_again_48"] <= 9] == [("360m", "Y at 8 bits")], (over("short_answers_48"), len(over("one_token_again_48"))))
eights = [r for r in hs["rows"] if "8-bit copy" in r]
claim("at the 8-bit bars (0.68, 1.04): all five 8-bit copies pass one at a time and in the order asked; back to back in 200 random orders the Pythia 8-bit copy went over in one, next to the answer that set the budget, the other four never; chosen short answers take all five past the budget",
      len(eights) == 5 and all(r["8-bit copy"]["alone_withheld"] == 0 and r["asked_order_largest_48"] <= 1.04 for r in eights)
      and [(name_of(r), r["8-bit copy"]["orders_with_a_withheld_answer"]) for r in eights if r["8-bit copy"]["withheld"]] == [(("pythia70m", "Y at 8 bits"), 1)]
      and [all(r["largest_total_answer"][0] in pair for pair in r["8-bit copy"]["withheld_after"]) for r in eights if r["8-bit copy"]["withheld"]] == [True]
      and round(max(r["largest_total_answer"][1] for r in eights), 2) == 0.82
      and all(r["8-bit copy"]["bars"] == [0.68, 1.04] and r["chosen"]["short_answers_48"] > 1.04 for r in eights), [r["chosen"]["short_answers_48"] for r in eights])
xg = np.concatenate([tok[k][n].astype(np.float64) for k in tok for n in tok[k].files if n.startswith("X answering in person|") and n.endswith("|gap")])
xr = xg[(xg.max(1) <= 4) & (xg.sum(1) <= 9 + 1e-9)]
claim("the listed model in person: 848 of its 2,400 answers pass; the largest gap in each has a median of 1.8 nats (a token about six times less likely than Y's choice)",
      (len(xg), len(xr), round(float(np.median(xr.max(1))), 1), round(float(np.exp(np.median(xr.max(1)))))) == (2400, 848, 1.8, 6), float(np.median(xr.max(1))))

# what the room in the token test is worth (results/smuggle/summary.json, from analyze_smuggle.py)
sm = load("smuggle/summary.json")
sg, st = sm["pairs"], sm["settings"]
pct = lambda v: round(100 * v)
span = lambda vs: (lambda v: (min(v), max(v)))([pct(x) for x in vs])
lead_of = lambda d: d["z_right"] - d["y_right"] >= 0.03          # under 3 points the share means nothing
share = lambda d, bar, sp: d[bar][f"spacing {sp}"]["Z-operator"]["share_of_lead"]
ref = lambda d: d["deployed"]["spacing 47"]["reference"]["right"]
lam = [r["lambada"] for r in sg.values()]
tri = {k: r["trivia"] for k, r in sg.items()}
pyt = [d for k, d in tri.items() if k.startswith("pythia")]
smol = [d for k, d in tri.items() if not k.startswith("pythia")]
able = [d for d in smol if lead_of(d)]
claim("room: the 8-bit bars with the deployed bars' margins (cap 37% over the 2.92 their recipe saw, budget 25% over 7.2) are a cap of 0.68 and a budget of 1.04, from 4,000 answers (largest gap 0.49, largest total over one answer 0.82); 200 orders, 1,000 resamples",
      st["bars"]["8-bit copy"] == [0.68, 1.04] and st["8-bit source"]["answers"] == 4000 and round(st["8-bit source"]["largest_gap"], 2) == 0.49
      and round(st["8-bit source"]["largest_total"], 2) == 0.82 and [round(v, 2) for v in st["8-bit source"]["deployed_recipe_saw"]] == [2.92, 7.2]
      and [round(100 * v - 100) for v in st["8-bit source"]["margins"]] == [37, 25] and (st["orders"], st["resamples"]) == (200, 1000), st["bars"]["8-bit copy"])
claim("room, last word (7 pairs): 71% to 97% of Z's lead passes with answers alone, 29% to 69% back to back, 15% to 46% at the 8-bit bars; the reference operator 67% to 92% right, Y alone 22% to 57%",
      len(sg) == 7 and all(map(lead_of, lam)) and span(share(d, "deployed", 47) for d in lam) == (71, 97) and span(share(d, "deployed", 0) for d in lam) == (29, 69)
      and span(share(d, "8-bit copy", 47) for d in lam) == (15, 46) and span(map(ref, lam)) == (67, 92) and span(d["y_right"] for d in lam) == (22, 57),
      span(share(d, "deployed", 0) for d in lam))
claim("room, closed-book: Pythia-70m right on 0.4%, 22% and 25% of the lead passes alone (13%, 15% back to back), Z's winning answers a median 5.8 to 6.1 nats off; the four capable pairs: Y 14% to 41%, 87% to 104% alone, 53% to 79% back to back, 1.1 to 1.8 nats; one pair's lead under 3 points",
      [round(1000 * d["y_right"]) for d in pyt] == [4, 4] and sorted(pct(share(d, "deployed", 47)) for d in pyt) == [22, 25]
      and sorted(pct(share(d, "deployed", 0)) for d in pyt) == [13, 15] and sorted(round(d["z_wins_median_gap"]["largest_token"], 1) for d in pyt) == [5.8, 6.1]
      and len(able) == 4 and span(d["y_right"] for d in able) == (14, 41) and span(share(d, "deployed", 47) for d in able) == (87, 104)
      and span(share(d, "deployed", 0) for d in able) == (53, 79)
      and (min(round(d["z_wins_median_gap"]["largest_token"], 1) for d in able), max(round(d["z_wins_median_gap"]["largest_token"], 1) for d in able)) == (1.1, 1.8)
      and sum(not lead_of(d) for d in tri.values()) == 1, [pct(share(d, "deployed", 47)) for d in able])
claim("room, closed-book: the reference operator 65% to 84% right where the SmolLM and Qwen known models get 14% to 41% (Pythia-70m 16%), every reference answer right by the same rule; 0.6% to 36% of the lead at the 8-bit bars; capable pairs back to back 43% to 79% over both tasks",
      span(map(ref, smol)) == (65, 84) and span(d["y_right"] for d in smol) == (14, 41) and [pct(ref(d)) for d in pyt] == [16, 16]
      and all(r[t]["reference_answers_right"] == 1.0 for r in sg.values() for t in ("lambada", "trivia"))
      and (lambda v: (round(1000 * min(v)), pct(max(v))))([share(d, "8-bit copy", 47) for d in tri.values() if lead_of(d)]) == (6, 36)
      and span(share(r[t], "deployed", 0) for k, r in sg.items() if not k.startswith("pythia") for t in ("lambada", "trivia") if lead_of(r[t])) == (43, 79),
      span(map(ref, smol)))

sys.exit(1 if bad else 0)
