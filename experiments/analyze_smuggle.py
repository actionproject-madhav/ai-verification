"""Run the deployed token test over served answers as one stream, for two operators (results/smuggle/).

The box keeps the gaps of the last 47 tokens it served and releases an answer only if no token of it is over the cap
and no 48 tokens in a row, the answer's own and those before it, add up to more than the budget
(lineage_check.outputs.answer_passes). A withheld answer stops the box, so an operator checks first: it knows Y and
can compute every gap. It serves its answer where the answer would pass, and Y's own answer, which costs nothing,
everywhere else.

  Z-operator   holds a better model Z and serves Z's answer
  reference    knows the accepted answers and serves the one with the smallest total gap, where Y's own is wrong. A
               point of reference, not a bound: another accepted answer, or budget held back for later answers, can
               do better

Spacing is how many tokens of Y's own text the box serves after each answer before the next (0: answers back to
back; 47: each answer is judged alone). Each figure is the mean over random orders of the answers. Its 95% interval
resamples the questions, each resample served in a random order of its own, so it includes the effect of the order.

Two sets of bars: the deployed ones, and the room an honest 8-bit copy needs with the margins the deployed bars
ended up with (the cap 37% above the largest single gap seen, the budget 25% above the largest total over one answer),
applied to the 8-bit copies' answers in results/trigger_serve_*.tokens.npz, each answer judged on its own.

Usage: python analyze_smuggle.py <results/smuggle> <out.json>
"""
import glob, json, math, os, sys, zlib
from multiprocessing import Pool
import numpy as np

WINDOW, SPACINGS, ORDERS, BOOT = 48, (0, 16, 47), 200, 1000
DEPLOYED = (4.0, 9.0)
EIGHT_BIT = ("Y at 8 bits", "Y, published file Q8_0")          # the honest 8-bit copies of results/trigger_serve_*
RECIPE = ("Y at 8 bits", "Y, published file Q8_0", "Y, published file Q6_K", "Y, published file Q5_K_M", "Y fine-tuned 200 steps")


def deployed_margins(results):
    """What the deployed bars ended up as, over what their recipe saw: the first 200 prompts of the five honest copies
    of the 135M pair (docs/RESULTS.md, "How the bars were chosen"): 4 over the largest gap, 9 over the largest total."""
    z = np.load(os.path.join(results, "trigger_serve_135m.tokens.npz"))
    g = np.concatenate([z[k].astype(np.float64) for k in z.files if k.split("|")[0] in RECIPE and k.split("|")[1] == "first" and k.endswith("|gap")])
    return DEPLOYED[0] / g.max(), DEPLOYED[1] / g.sum(1).max(), [float(g.max()), float(g.sum(1).max())]


def eight_bit_bars(results):
    """The deployed bars' margins on the honest 8-bit copies' answers, rounded up to two decimals."""
    cap_margin, budget_margin, seen = deployed_margins(results)
    g = np.concatenate([z[k].astype(np.float64) for f in sorted(glob.glob(os.path.join(results, "trigger_serve_*.tokens.npz")))
                        for z in [np.load(f)] for k in z.files if k.split("|")[0] in EIGHT_BIT and k.endswith("|gap")])
    up = lambda v: math.ceil(round(v * 100, 6)) / 100
    return (up(cap_margin * g.max()), up(budget_margin * g.sum(1).max())), {
        "answers": len(g), "largest_gap": float(g.max()), "largest_total": float(g.sum(1).max()),
        "margins": [cap_margin, budget_margin], "deployed_recipe_saw": seen}


def passes(g, tail, cap, budget):
    if len(g) and max(g) > cap:
        return False
    s = tail[-(WINDOW - 1):] + g
    c = np.concatenate([[0.0], np.cumsum(s)])
    ends = np.arange(len(s) - len(g) + 1, len(s) + 1)
    return bool((c[ends] - c[np.maximum(ends - WINDOW, 0)]).max(initial=0.0) <= budget + 1e-9)


def stream(order, mine, mine_right, y_gap, y_right, cap, budget, spacing):
    """Answers right, and how many of the operator's answers that differ from Y's were released."""
    tail, right, released = [], 0, 0
    for i in order:
        g = mine[i]
        if g is not None and passes(list(g), tail, cap, budget):
            tail, right, released = tail + list(g), right + mine_right[i], released + 1
        else:
            tail, right = tail + [0.0] * len(y_gap[i]), right + y_right[i]
        tail = (tail + [0.0] * spacing)[-(WINDOW - 1):]
    return right / len(order), released


def split(z, task, who):
    flat, off = z[f"{task}_{who}_gap"], z[f"{task}_{who}_off"]
    return [flat[off[i]:off[i + 1]].astype(np.float64) for i in range(len(off) - 1)]


def task_numbers(z, task, rng, bars):
    yg, zg, tg = split(z, task, "y"), split(z, task, "z"), split(z, task, "truth")
    yr, zr, same = z[f"{task}_y_right"], z[f"{task}_z_right"], z[f"{task}_same"]
    n = len(yr)
    tr = z[f"{task}_truth_right"] if f"{task}_truth_right" in z.files else np.ones(n, bool)   # LAMBADA: the true word itself
    ops = {"Z-operator": ([None if same[i] else zg[i] for i in range(n)], zr),     # Z's answer when it is Y's costs nothing either way
           "reference": ([None if yr[i] else tg[i] for i in range(n)], tr)}
    out = {"questions": n, "y_right": float(yr.mean()), "z_right": float(zr.mean()), "answers_differ": int((~same).sum()),
           "z_right_where_y_wrong": int((zr & ~yr).sum()), "y_right_where_z_wrong": int((yr & ~zr).sum()), "reference_answers_right": float(tr.mean())}
    zg_diff = np.array([g.max() for g, s in zip(zg, same) if not s])
    out["z_answers_differing_within_cap"] = {k: float((zg_diff <= cap).mean()) for k, (cap, _) in bars.items()}
    won = np.array([g for g, a, b in zip(zg, zr, yr) if a and not b], dtype=object)
    if len(won):
        out["z_wins_median_gap"] = {"largest_token": float(np.median([g.max() for g in won])), "total": float(np.median([g.sum() for g in won]))}
    for bar, (cap, budget) in bars.items():
        for spacing in SPACINGS:
            for op, (mine, right) in ops.items():
                runs = [stream(rng.permutation(n), mine, right, yg, yr, cap, budget, spacing) for _ in range(ORDERS)]
                acc, rel = float(np.mean([r[0] for r in runs])), float(np.mean([r[1] for r in runs]))
                boot, no_lead = [], 0
                for _ in range(BOOT):
                    idx = rng.integers(0, n, n)
                    a, _ = stream(rng.permutation(idx), mine, right, yg, yr, cap, budget, spacing)
                    lead = zr[idx].mean() - yr[idx].mean()
                    no_lead += op == "Z-operator" and lead <= 0
                    boot.append((a, (a - yr[idx].mean()) / lead if op == "Z-operator" and lead > 0 else np.nan))
                boot = np.array(boot)
                row = {"right": acc, "right_95": [float(v) for v in np.percentile(boot[:, 0], [2.5, 97.5])],
                       "right_by_order": [float(min(r[0] for r in runs)), float(max(r[0] for r in runs))]}
                if op == "Z-operator":
                    lead = zr.mean() - yr.mean()
                    row.update(share_of_lead=(acc - yr.mean()) / lead if lead > 0 else None,
                               share_95=[float(v) for v in np.nanpercentile(boot[:, 1], [2.5, 97.5])] if lead > 0 else None,
                               resamples_without_a_lead=int(no_lead), differing_released=rel / max(int((~same).sum()), 1))
                out.setdefault(bar, {}).setdefault(f"spacing {spacing}", {})[op] = row
    return out


def one(job):
    path, bars = job
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    name = os.path.basename(path)[:-len(".answers.npz")]
    rng = np.random.default_rng(zlib.crc32(name.encode()))        # each pair its own draws: adding a pair changes no other
    return name, {"Y": meta["Y"], "Z": meta["Z"], "revisions": [meta["Y_revision"], meta["Z_revision"]],
                  **{t: task_numbers(z, t, rng, bars) for t in meta["tasks"]}}


def main(res, out):
    eight, source = eight_bit_bars(os.path.join(res, ".."))
    bars = {"deployed": DEPLOYED, "8-bit copy": eight}
    files = sorted(glob.glob(os.path.join(res, "*.answers.npz")))
    with Pool(min(len(files), os.cpu_count() or 1)) as pool:
        pairs = dict(pool.map(one, [(p, bars) for p in files]))
    for name, r in pairs.items():
        for t in (k for k in r if isinstance(r[k], dict)):
            d = r[t]["deployed"]
            print(f"{name:26s} {t:8s} Y {r[t]['y_right']:.3f} Z {r[t]['z_right']:.3f} | share alone {d['spacing 47']['Z-operator']['share_of_lead']}"
                  f" back-to-back {d['spacing 0']['Z-operator']['share_of_lead']} | reference: alone {d['spacing 47']['reference']['right']:.3f}"
                  f" b2b {d['spacing 0']['reference']['right']:.3f}", flush=True)
    json.dump({"settings": {"bars": {k: list(v) for k, v in bars.items()}, "8-bit source": source, "orders": ORDERS, "resamples": BOOT,
                            "spacings": list(SPACINGS)}, "pairs": pairs}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
