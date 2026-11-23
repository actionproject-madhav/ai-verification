"""Does a refusal-removal edit show in the difference between a copy and its base?

The scan (`exp_hub_scan.py`) traces popular models to a registry base. For a sample of those close copies this
script reads the matrices that write to the residual stream in six layers of the copy and of the base, and asks
in how many layers their differences share one leading direction (`lineage_check.universal.shared_edit`).

Labels come from repository names only: "abliterated" and "heretic" name two families of refusal-removal tools.
Names with other hints (uncensored, dolphin, ...) are kept apart: many of those are fine-tunes. Everything else is
a control.

Models above MAX_PARAMS are left out: reading six layers of a 70B model twice takes minutes.

Usage: python exp_refusal_edit.py <scan dir> <out.jsonl> [workers] [controls]
       python exp_refusal_edit.py summary <out.jsonl>          counts under the tool's own thresholds
"""
import json, multiprocessing as mp, os, random, re, shutil, sys, time

LAYERS, TOPS, MAX_PARAMS = 6, (1, 2, 4, 8), 16e9
EDIT = re.compile(r"abliterat|heretic", re.I)
HINT = re.compile(r"uncensor|josiefied|dolphin|lexi|unfilter|decensor|derestrict|obliter|jailb|refusal|lorablated|\bdan\b", re.I)


def label(repo):
    m = EDIT.search(repo)
    return m.group(0).lower() if m else "hint" if HINT.search(repo) else "control"


def one(job):
    import torch
    from lineage_check import light as LT, remote as R, universal as U
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    repo, sha, base, base_sha, nonce, row = job
    t0, out = time.time(), dict(row)
    try:
        src, anchor = R.Remote(repo, sha), R.Remote(base, base_sha)
        found = U.edit_directions(src, anchor, LT.layers_for(src, nonce, LAYERS), LT.layers_for(anchor, nonce, LAYERS))
        out["changed"] = None if found is None else len({f[0] for f in found})
        if found:
            out["positions"], out["share"] = [f[0] for f in found], [round(f[2], 4) for f in found]
            out["relative_size"], out["own"] = [round(f[3], 4) for f in found], [round(f[4], 4) for f in found]
            lead = torch.stack([f[1][:, 0] for f in found])
            out["cosines"] = [[round(abs(float(c)), 3) for c in row] for row in lead @ lead.T]   # leading directions, pairwise
            for top in TOPS:                                      # how well each difference holds the common direction
                out[f"fit{top}"] = [round(c, 4) for c in U.common_direction(found, top)[1]]
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:140]}"
    shutil.rmtree(os.path.join(R.CACHE, repo.replace("/", "--")), ignore_errors=True)
    out["seconds"] = round(time.time() - t0, 1)
    return out


def main(scan, out, workers=16, controls=400, only=None):
    rows = json.load(open(os.path.join(scan, "compare.json")))["rows"]
    nonce = open(os.path.join(scan, "nonce.txt")).read().strip()
    sha = {}
    for f in sorted(os.listdir(scan)):
        if f.startswith("scan_") and f.endswith(".jsonl") or f == "scan.jsonl":
            for l in open(os.path.join(scan, f)):
                if l.strip():
                    r = json.loads(l)
                    if "error" not in r:
                        sha[r["repo"]] = r["sha"]
    params = {p["repo"]: p["params"] or 0 for p in json.load(open(os.path.join(scan, "population.json")))["population"]}
    close = [r for r in rows if r.get("light") == "close copy" and r.get("nearest") in sha and params.get(r["repo"], 0) <= MAX_PARAMS]
    pick = [r for r in close if label(r["repo"]) != "control"]
    rest = [r for r in close if label(r["repo"]) == "control"]
    random.Random(7).shuffle(rest)
    pick += rest[: int(controls)]
    if only:                                                      # a file of repository names: read these again and no others
        pick = [r for r in pick if r["repo"] in set(open(only).read().split())]
    done = {json.loads(l)["repo"] for l in open(out)} if os.path.exists(out) else set()
    jobs = [(r["repo"], sha[r["repo"]], r["nearest"], sha[r["nearest"]], nonce,
             {"repo": r["repo"], "base": r["nearest"], "label": label(r["repo"]), "distance": r["distance"],
              "blind_signature": r["ablation_signature"]}) for r in pick if r["repo"] not in done]
    print(len(close), "close copies;", len(pick), "picked;", len(jobs), "to do", flush=True)
    with mp.get_context("spawn").Pool(int(workers)) as pool, open(out, "a") as fh:
        for n, rec in enumerate(pool.imap_unordered(one, jobs), 1):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if n % 25 == 0:
                print(n, "of", len(jobs), flush=True)


def cut(r):
    """Layers cut, from a saved row, by the rule in `lineage_check.universal.cut_layers`."""
    from lineage_check import universal as U
    pos, cos, share, own = r["positions"], r["cosines"], r["share"], [o >= U.CUT_OWN for o in r["own"]]
    layers = {p for p, s, o in zip(pos, share, own) if o and s >= U.CUT_SHARE}
    layers |= {pos[i] for i in range(len(pos)) for j in range(i + 1, len(pos))
               if pos[i] == pos[j] and own[i] and own[j] and cos[i][j] >= U.CUT_COSINE}
    return len(layers)


def summary(out):
    """What the tool would say about each group. A copy is compared when its layers hold the same writing matrices
    as the base's and at least one differs. It is Red with two or more cut layers. Otherwise it is cleared if every
    difference is small next to its matrix, and left unchecked (Amber) if not."""
    from lineage_check import light as LT, universal as U
    rows = [json.loads(l) for l in open(out) if l.strip()]
    result = {}
    for name in ("abliterat", "heretic", "hint", "control"):
        group = [r for r in rows if r["label"] == name]
        read = [r for r in group if r.get("positions")]
        flagged = [r for r in read if cut(r) >= LT.CUT_MIN]
        unchecked = [r for r in read if cut(r) < LT.CUT_MIN and max(r["relative_size"]) > U.ALIGNED]
        agree = sorted(max([r["cosines"][i][j] for i in range(len(r["positions"])) for j in range(i + 1, len(r["positions"]))
                            if r["positions"][i] == r["positions"][j]] or [0.0]) for r in read)
        result[name] = {"models": len(group), "not_comparable": sum(r.get("changed") is None for r in group),
                        "identical_to_base": sum(r.get("changed") == 0 for r in group), "compared": len(read),
                        "flagged": len(flagged), "unchecked": len(unchecked), "cleared": len(read) - len(flagged) - len(unchecked),
                        "older_signature": sum(bool(r["blind_signature"]) for r in read),
                        "largest_agreement_in_a_layer": {"median": agree[len(agree) // 2], "p95": agree[int(0.95 * (len(agree) - 1))]} if agree else None,
                        "flagged_repos": [r["repo"] for r in flagged] if name == "control" else None}
        print(f"{name:10s} {len(group):3d} models: {result[name]['not_comparable']} not comparable, {result[name]['identical_to_base']} identical, "
              f"{len(read)} compared -> {len(flagged)} flagged, {len(unchecked)} unchecked, {result[name]['cleared']} cleared   (older signature: {result[name]['older_signature']})")
    return result


if __name__ == "__main__":
    if sys.argv[1] == "summary":
        json.dump(summary(sys.argv[2]), open(os.path.splitext(sys.argv[2])[0] + "_summary.json", "w"), indent=1)
    else:
        main(*sys.argv[1:6])
