"""How much training separates two models? Fingerprint drift along Pythia's own training run.

Pythia trains for 143,000 steps of 2,097,152 tokens (300B tokens). The checkpoint at step k is an ancestor of the
final model separated by (143000 - k) steps, a known fraction of the full run's compute.

Usage: python exp_drift.py <out.json>
"""
import json, sys
from lineage_check import remote as R
from lineage_check import scan

NONCE, N_TOK = "lineage-audit-2026-11-18", 512
PLAN = {"70m": ([142000, 140000, 136000, 130000, 120000, 110000, 100000, 80000, 60000, 40000, 20000], "all"),
        "160m": ([142000, 140000, 136000, 130000, 120000, 110000, 100000, 80000, 60000, 40000, 20000], "all"),
        "410m": ([140000, 130000, 110000, 80000, 50000], 8)}
TOTAL = 143000


def main(out):
    res = []
    vocab = R.Remote("EleutherAI/pythia-160m").vocab
    strings = scan.fp4.select_strings(vocab, NONCE, N_TOK)
    for size, (steps, layers) in PLAN.items():
        repo = f"EleutherAI/pythia-{size}"
        main_fp = scan.fingerprint_source(R.Remote(repo), strings, layers, label=f"drift/{size}/main", vocab=vocab)
        for k in steps:
            fp = scan.fingerprint_source(R.Remote(repo, f"step{k}"), strings, layers, label=f"drift/{size}/step{k}", vocab=vocab)
            r = scan.compare(main_fp, fp)
            rec = {"size": size, "step": k, "steps_apart": TOTAL - k, "frac_of_run": (TOTAL - k) / TOTAL,
                   "tokens_apart": (TOTAL - k) * 2097152, "d": r["d"], "attn": r["attn"]}
            res.append(rec)
            print(json.dumps(rec), flush=True)
            json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
