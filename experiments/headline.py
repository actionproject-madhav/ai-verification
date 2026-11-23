"""Recompute the README's headline numbers from the files in results/. Exits non-zero if one no longer holds.

Usage: python headline.py [results dir]
"""
import collections, json, os, re, sys

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

sys.exit(1 if bad else 0)
