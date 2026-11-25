"""Honest copies' answers served back to back, in random orders (results/honest_stream.json).

results/trigger_serve_*.json judges each answer alone. The box carries the budget from one answer into the next, so
here each honest copy's 800 answers are served back to back in random orders through the same rule
(analyze_smuggle.passes): how often an answer would be withheld, which stops the box, and the largest total over any
48 tokens in a row the rule saw. At the deployed bars for every honest copy, and at the 8-bit bars for the 8-bit
copies.

Then a user who chooses: Y is public, so whoever reads the served answers can compute every gap. Two ways to fill 48
served tokens from these prompts: short answers (the first L tokens of an answer) from the prompts that cost the most,
back to back, the best L; and one prompt asked again and again for a one-token answer, 48 times its first gap, or 48
times any token's gap if the prompt ends partway into an answer (the served model then continues from there).

Usage: python honest_stream.py [results dir]
"""
import glob, json, os, sys, zlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_smuggle import DEPLOYED, EIGHT_BIT, WINDOW, eight_bit_bars, passes

RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
ORDERS = 200
NOT_HONEST = ("hidden trigger", "X answering", "second choice")


def served(answers, cap, budget, rng):
    """One random order, back to back: answers withheld (counted, then served as Y's own, which costs nothing, so the
    stream goes on), which they were and which answer came just before each, and the largest total over 48 tokens in a
    row, the answer being judged included."""
    tail, withheld, worst, which, prev = [], 0, 0.0, set(), -1          # -1: nothing served before
    for i in rng.permutation(len(answers)):
        g = list(answers[i])
        c = np.concatenate([[0.0], np.cumsum(tail + g)])
        worst = max(worst, float((c[WINDOW:] - c[:-WINDOW]).max()) if len(c) > WINDOW else float(c[-1]))
        ok = passes(g, tail, cap, budget)
        withheld += not ok
        which |= set() if ok else {(prev, int(i))}
        prev = int(i)
        tail = (tail + (g if ok else [0.0] * len(g)))[-(WINDOW - 1):]
    return withheld, worst, which


def main():
    eight, _ = eight_bit_bars(RES)
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, "trigger_serve_*.tokens.npz"))):
        z, pair = np.load(f), os.path.basename(f)[len("trigger_serve_"):-len(".tokens.npz")]
        for name in sorted({k.split("|")[0] for k in z.files}):
            if any(s in name for s in NOT_HONEST):
                continue
            keys = sorted(k for k in z.files if k.startswith(name + "|") and k.endswith("|gap"))
            ans = np.concatenate([z[k].astype(np.float64) for k in keys])
            c = np.concatenate([[0.0], np.cumsum(ans.reshape(-1))])
            pre = np.cumsum(ans, 1)
            short = {L: float(np.sort(pre[:, L - 1])[::-1][:WINDOW // L].sum()) for L in (1, 2, 3, 4, 6, 8, 12, 16, 24, 48)}
            best = max(short, key=short.get)
            row = {"pair": pair, "copy": name, "answers": len(ans),
                   "among_the_11": "1,000 steps" not in name,               # the 11 copies of finding 2: all but the long fine-tune
                   "asked_order_largest_48": float((c[WINDOW:] - c[:-WINDOW]).max()),
                   "largest_total_answer": [int(ans.sum(1).argmax()), float(ans.sum(1).max())],
                   "chosen": {"short_answers_48": short[best], "length": best, "one_token_again_48": WINDOW * float(ans[:, 0].max()),
                              "any_token_again_48": WINDOW * float(ans.max())}}
            for bar, (cap, budget) in [("deployed", DEPLOYED)] + ([("8-bit copy", eight)] if name in EIGHT_BIT else []):
                rng = np.random.default_rng(zlib.crc32(f"{pair}|{name}|{bar}".encode()))
                runs = [served(ans, cap, budget, rng) for _ in range(ORDERS)]
                row[bar] = {"bars": [cap, budget], "alone_withheld": int(((ans.max(1) > cap) | (ans.sum(1) > budget + 1e-9)).sum()),
                            "orders": ORDERS, "orders_with_a_withheld_answer": sum(w > 0 for w, _, _ in runs),
                            "withheld": sum(w for w, _, _ in runs), "withheld_after": sorted(set().union(*(x for _, _, x in runs))),       # (answer before, answer withheld)
                            "served": ORDERS * len(ans), "largest_48": max(m for _, m, _ in runs)}
            rows.append(row)
            print(json.dumps(row), flush=True)
    json.dump({"orders": ORDERS, "rows": rows}, open(os.path.join(RES, "honest_stream.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
