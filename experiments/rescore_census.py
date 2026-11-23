"""Census with the reader-checked signature (forensic.py v3), from cached layers; adds OLMo and Pythia as LayerNorm controls."""
import json
import numpy as np
from lineage_check import remote as R, forensic
import exp_census as C
EXTRA_CONTROLS = ["allenai/OLMo-7B-Instruct-hf", "allenai/OLMo-7B-hf", "NousResearch/Hermes-2-Pro-Llama-3-8B"]
rows = []
for truth, repos in (("ablated", C.ABLATED), ("control", C.CONTROLS[:1] + C.CONTROLS[2:] + EXTRA_CONTROLS)):
    for repo in repos:
        try:
            src = R.Remote(repo)
            meta = R.build(src, layers=[])
            L = meta["L"]
            layers = sorted({int(round((j + 0.5) * L / C.K - 0.5)) for j in range(C.K)})
            W, Rd = [], []
            for i in layers:
                l = R.build(src, layers=[i])["layers"][i]
                W.append(np.concatenate([l["o"], l["down"]], 1))
                Rd.append(forensic.layer_readers(l))
                del l
            ab, k, agree, ks, rr = forensic.verdict(W + [meta["E"].T], Rd + [None])
            rows.append({"model": repo, "truth": truth, "flagged": ab, "dims": k, "agreement": agree, "k_per_writer": ks, "read_ratio": rr})
            print(f"{repo:52s} [{truth:7s}] -> {'ABLATED' if ab else 'clean':7s} dead dims {ks}  agree {agree if agree is None else round(agree, 3)}  read {rr if rr is None else round(rr, 3)}", flush=True)
        except Exception as e:
            rows.append({"model": repo, "truth": truth, "error": str(e)[:160]})
            print(f"{repo:52s} ERROR {type(e).__name__}: {str(e)[:120]}", flush=True)
json.dump(rows, open("../results/census_v3.json", "w"), indent=1)
ab = [r for r in rows if r["truth"] == "ablated" and "error" not in r]
co = [r for r in rows if r["truth"] == "control" and "error" not in r]
print(f"\nflagged: {sum(r['flagged'] for r in ab)}/{len(ab)} abliterated, {sum(r['flagged'] for r in co)}/{len(co)} controls")
