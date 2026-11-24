"""Run the deployed token test over served answers as one stream, for two operators (results/smuggle/).

The box keeps the gaps of the last 47 tokens it served and releases an answer only if no token of it is over the cap
and no 48 tokens in a row, the answer's own and those before it, add up to more than the budget
(lineage_check.outputs.answer_passes). A withheld answer stops the box, so an operator checks first: it knows Y and
can compute every gap. It serves its answer where the answer would pass, and Y's own answer, which costs nothing,
everywhere else.

  Z-operator      holds a better model Z and serves Z's answer
  knows-answers   knows the right answer and serves it where Y's own is wrong (the most any operator can get right)

Spacing is how many tokens of Y's own text the box serves after each answer before the next (0: answers back to
back; 47: each answer is judged alone). Answers come in a random order; intervals resample the questions.

Usage: python analyze_smuggle.py <results/smuggle> <out.json>
"""
import glob, json, os, sys
import numpy as np

WINDOW, BARS, SPACINGS, ORDERS, BOOT = 48, {"deployed": (4.0, 9.0), "8-bit copy": (0.5, 1.0)}, (0, 16, 47), 10, 300


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


def task_numbers(z, task, rng):
    yg, zg, tg = split(z, task, "y"), split(z, task, "z"), split(z, task, "truth")
    yr, zr, same = z[f"{task}_y_right"], z[f"{task}_z_right"], z[f"{task}_same"]
    n = len(yr)
    ops = {"Z-operator": ([None if same[i] else zg[i] for i in range(n)], zr),     # Z's answer when it is Y's costs nothing either way
           "knows-answers": ([None if yr[i] else tg[i] for i in range(n)], np.ones(n, bool))}
    out = {"questions": n, "y_right": float(yr.mean()), "z_right": float(zr.mean()), "answers_differ": int((~same).sum()),
           "z_right_where_y_wrong": int((zr & ~yr).sum()), "y_right_where_z_wrong": int((yr & ~zr).sum())}
    zg_diff = np.array([g.max() for g, s in zip(zg, same) if not s])
    out["z_answers_differing_within_cap"] = {k: float((zg_diff <= cap).mean()) for k, (cap, _) in BARS.items()}
    won = np.array([g for g, a, b in zip(zg, zr, yr) if a and not b], dtype=object)
    if len(won):
        out["z_wins_median_gap"] = {"largest_token": float(np.median([g.max() for g in won])), "total": float(np.median([g.sum() for g in won]))}
    for bar, (cap, budget) in BARS.items():
        for spacing in SPACINGS:
            for op, (mine, right) in ops.items():
                runs = [stream(rng.permutation(n), mine, right, yg, yr, cap, budget, spacing) for _ in range(ORDERS)]
                acc, rel = float(np.mean([r[0] for r in runs])), float(np.mean([r[1] for r in runs]))
                boot = []
                for _ in range(BOOT):
                    idx = rng.integers(0, n, n)
                    a, _ = stream(rng.permutation(idx), mine, right, yg, yr, cap, budget, spacing)
                    lead = zr[idx].mean() - yr[idx].mean()
                    boot.append((a, (a - yr[idx].mean()) / lead if op == "Z-operator" and lead > 0 else np.nan))
                boot = np.array(boot)
                row = {"right": acc, "right_95": [float(v) for v in np.percentile(boot[:, 0], [2.5, 97.5])]}
                if op == "Z-operator":
                    lead = zr.mean() - yr.mean()
                    row.update(share_of_lead=(acc - yr.mean()) / lead if lead > 0 else None,
                               share_95=[float(v) for v in np.nanpercentile(boot[:, 1], [2.5, 97.5])] if lead > 0 else None,
                               differing_released=rel / max(int((~same).sum()), 1))
                out.setdefault(bar, {}).setdefault(f"spacing {spacing}", {})[op] = row
    return out


def text_numbers(z):
    """Running text (teacher-forced, from exp_smuggle.py): of the positions where Z's choice is not Y's, the share
    at which serving Z's choice keeps every 48-position window within the bars."""
    cost, same = z["cost_z"].astype(np.float64), z["same"]
    out = {"positions": int(same.size), "z_differs": float((~same).mean())}
    for bar, (cap, budget) in BARS.items():
        spent, served = np.zeros(len(cost)), np.zeros(cost.shape, bool)
        for k in range(cost.shape[1]):
            g = cost[:, k]
            go = (~same[:, k]) & (g <= cap) & (spent + g <= budget)
            spent += np.where(go, g, 0.0)
            served[:, k] = go
        out[bar] = float(served[~same].mean())
    return out


def main(res, out):
    rng, rows = np.random.default_rng(0), {}
    for p in sorted(glob.glob(os.path.join(res, "*.answers.npz"))):
        z = np.load(p, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        name = os.path.basename(p)[:-len(".answers.npz")]
        rows[name] = {"Y": meta["Y"], "Z": meta["Z"], "revisions": [meta["Y_revision"], meta["Z_revision"]],
                      **{t: task_numbers(z, t, rng) for t in meta["tasks"]}}
        old = os.path.join(res, name + ".npz")
        if os.path.exists(old):
            rows[name]["text"] = text_numbers(np.load(old))
        r = rows[name]
        for t in meta["tasks"]:
            d = r[t]["deployed"]
            print(f"{name:26s} {t:8s} Y {r[t]['y_right']:.3f} Z {r[t]['z_right']:.3f} | share alone {d['spacing 47']['Z-operator']['share_of_lead']}"
                  f" back-to-back {d['spacing 0']['Z-operator']['share_of_lead']} | knows: alone {d['spacing 47']['knows-answers']['right']:.3f}"
                  f" b2b {d['spacing 0']['knows-answers']['right']:.3f}", flush=True)
    json.dump(rows, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
