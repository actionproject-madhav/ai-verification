"""Sort the census of published GGUF files (exp_derive.py) into what the box can say about each.

  exact                      every tensor re-derives from the declared base with one build of the encoder, tokens equal
  calibrated                 made with an importance matrix (recorded in the file, or a format that needs one): the
                             rounding depends on data the publisher chose
  a format the encoder does not write   tensors in a format the tried build of llama.cpp no longer has (the ARM
                             repacked 4-bit formats) or does not have
  rescaled per channel       the base with channels rescaled before encoding (norm weights divided, the next matrices'
                             columns multiplied), as activation-aware quantisers do: the scales are data the publisher chose
  a sibling's weights        exact or close to another model of the same family (an Instruct file declared as the base)
  edited in places           the declared base's weights, a few matrices changed (refusal removal, merges)
  same weights, other rounding   the declared base's weights at the format's own error, written by an encoder not tried
  different weights, close to the base   matrices within the format's own error of the base, but the tensors stored
                             without rounding (the norms) differ: not the base's numbers (a light fine-tune, for one)
  different weights          further from the declared base and its siblings than the format's own error
The last five come from the second pass (exp_derive.py explain), which reads a sample of matrices of each file.

Two later passes refine the sort, when given: rows that read every layer of a file (exp_derive.py explain with
EXPLAIN_LAYERS=all) replace the sampled verdict, and files re-derived against the sibling they match (rows with a
"declared" base) are "exact, of a sibling" when they re-derive: the file is exactly another model than its card names.

Usage: python analyze_derive.py <census.jsonl> <explain.jsonl> <out.json> [every-layer.jsonl ...] [census.sibling.jsonl ...]
"""
import collections, json, sys


def verdict(r, why, sibling=None):
    if sibling and sibling.get("derived"):
        return "exact, of a sibling"
    if r.get("derived"):
        return "exact" if r.get("vocab", {}).get("same", 1) == r.get("vocab", {}).get("compared", 1) else "exact weights, other tokens"
    if r.get("calibrated") or r.get("needs_data"):
        return "calibrated"
    if r.get("unread"):
        return "a format the encoder does not write"
    w = why.get((r["repo"], r["file"]))
    if w is None or "verdict" not in w:
        return "not explained"
    v = w["verdict"]
    if v in ("same weights, other rounding", "edited in places") and r.get("plain") and r.get("plain_identical", 0) < r["plain"]:
        return "different weights, close to the base"      # the unquantised tensors (norms) differ: not the base's numbers
    return "a sibling's weights" if v.startswith("a sibling") else v


def main(census, explain, out, *later):
    rows = [json.loads(l) for l in open(census)]
    why = {(w["repo"], w["file"]): w for w in map(json.loads, open(explain))}
    sibling = {}
    for f in later:
        for x in map(json.loads, open(f)):
            if "declared" in x:
                sibling[(x["repo"], x["file"])] = x
            elif "verdict" in x:
                why[(x["repo"], x["file"])] = x                # every layer read: this verdict replaces the sampled one
    ok = [r for r in rows if "error" not in r]
    for r in ok:
        r["verdict"] = verdict(r, why, sibling.get((r["repo"], r["file"])))
    count = lambda key: collections.Counter(key(r) for r in ok)
    owners = collections.defaultdict(collections.Counter)
    for r in ok:
        owners[r["repo"].split("/")[0]][r["verdict"]] += 1
    top = sorted(owners.items(), key=lambda t: -sum(t[1].values()))[:15]
    builds = collections.Counter("+".join(r.get("builds") or []) or "no compiled build" for r in ok if r["verdict"] == "exact")
    fmt = collections.defaultdict(collections.Counter)
    for r in ok:
        fmt[r.get("format", "?")][r["verdict"] == "exact"] += 1
    res = {"files": len(rows), "read": len(ok), "errors": collections.Counter(r["error"].split(":")[0] for r in rows if "error" in r),
           "repositories": len({r["repo"] for r in rows}), "bases": sorted({r["base"] for r in rows}),
           "verdicts": dict(count(lambda r: r["verdict"]).most_common()),
           "exact_of_uncalibrated": [sum(r["verdict"] == "exact" for r in ok), sum(r["verdict"] != "calibrated" for r in ok)],
           "exact_any_model": sum(r["verdict"] in ("exact", "exact, of a sibling") for r in ok),
           "exact_builds": dict(builds.most_common()),
           "by_format": {f: {"files": sum(c.values()), "exact": c[True]} for f, c in sorted(fmt.items(), key=lambda t: -sum(t[1].values()))},
           "by_publisher": {o: dict(c) for o, c in top},
           "drawn_at_random": dict(collections.Counter(r["verdict"] for r in ok if r.get("drawn"))),
           "most_downloaded": dict(collections.Counter(r["verdict"] for r in ok if not r.get("drawn"))),
           "template": dict(count(lambda r: r.get("template", "?")).most_common()),
           "by_year": {y: dict(c) for y, c in sorted(collections.defaultdict(collections.Counter, {
               y: collections.Counter(r["verdict"] for r in ok if r.get("created", "")[:4] == y) for y in {r.get("created", "")[:4] for r in ok}}).items())}}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("files", "read", "errors", "repositories", "verdicts", "exact_of_uncalibrated", "exact_builds")}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
