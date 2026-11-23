"""Does a refusal-removal edit show in the difference between a copy and its base?

The scan (`exp_hub_scan.py`) traces popular models to a registry base. For a sample of those close copies this
script reads the matrices that write to the residual stream in six layers of the copy and of the base, and asks
in how many layers their differences share one leading direction (`lineage_check.universal.shared_edit`).

Labels come from repository names only: "abliterated" and "heretic" name two families of refusal-removal tools.
Names with other hints (uncensored, dolphin, ...) are kept apart: many of those are fine-tunes. Everything else is
a control.

Models above MAX_PARAMS are left out: reading six layers of a 70B model twice takes minutes.

Usage: python exp_refusal_edit.py <scan dir> <out.jsonl> [workers] [controls]
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
        out["changed"] = None if found is None else len({p for p, _, _ in found})
        if found:
            out["positions"], out["share"] = [p for p, _, _ in found], [round(s, 4) for _, _, s in found]
            lead = torch.stack([u[:, 0] for _, u, _ in found])
            out["cosines"] = [[round(abs(float(c)), 3) for c in row] for row in lead @ lead.T]   # leading directions, pairwise
            for top in TOPS:                                      # how well each difference holds the common direction
                out[f"fit{top}"] = [round(c, 4) for c in U.common_direction(found, top)[1]]
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:140]}"
    shutil.rmtree(os.path.join(R.CACHE, repo.replace("/", "--")), ignore_errors=True)
    out["seconds"] = round(time.time() - t0, 1)
    return out


def main(scan, out, workers=16, controls=400):
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


if __name__ == "__main__":
    main(*sys.argv[1:5])
