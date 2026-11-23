"""The traffic light's answer on every model the red-team rounds produced.

The watchlist holds X, the model every round starts from. The registry holds Y, the separately trained model the
forgery rounds try to pass as. Each saved model gets the light the tool would give it, with the distance of its
closest part and of all its parts to X and to Y.

With LIGHT_ACT=1 each model also gets the activation reader's distance to X and to Y. That reader compares what
layers compute on secret probe text. It is not part of the tool; the columns show what a check on activations
would say about the same models.

Usage: python exp_lights_redteam.py <dir of saved models> <out.json>     (RT_MODEL / RT_INDEP choose the pair)
"""
import json, os, sys
import torch
from lineage_check import light as LT, remote as R

torch.set_grad_enabled(False)
X = os.environ.get("RT_MODEL", "HuggingFaceTB/SmolLM2-135M")
Y = os.environ.get("RT_INDEP", "HuggingFaceTB/SmolLM-135M")
NONCE = os.environ.get("LIGHT_NONCE", "redteam-lights-2026-11")


def activations(where):
    """{name: [distance to X, distance to Y]} from activations on secret probe text."""
    import exp_adaptive as A
    import exp_round2 as R2

    def read(path):
        m = A.load(path, "cpu")
        R2.LAYERS = R2.V.layers_of(m)
        f = R2.v4_live(m, R2.SECRET_IDS, R2.LAYERS, grad=False)
        del m
        return f
    fx, fy = read(X), read(Y)
    out = {}
    for name, path in where.items():
        f = fx if path == X else fy if path == Y else read(path)
        out[name] = [round(float(R2.d_v4(fx, f)), 4), round(float(R2.d_v4(fy, f)), 4)]
    return out


def main(root, out):
    watch, known, cache, rows = [({"id": "X"}, R.Remote(X))], [({"id": "Y"}, R.Remote(Y))], {}, []
    cases, where = [("X itself", watch[0][1]), ("Y itself", known[0][1])], {"X itself": X, "Y itself": Y}
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if os.path.exists(os.path.join(path, "config.json")):
            cases.append((name, R.LocalST(path, label=name, repo=X)))
            where[name] = path
    for name, src in cases:
        r = LT.light(src, watch, known, NONCE, cache=cache)
        d = {k: None if v is None else [round(v[0], 4), round(v[1], 4)] for k, v in r["distances"].items()}
        if "Y" not in d:                                   # Red stops before the registry: measure Y anyway
            v = LT.distances(src, known[0][1], NONCE, cache=cache)
            d["Y"] = None if v is None else [round(v[0], 4), round(v[1], 4)]
        rows.append({"model": name, "light": r["light"], "matched": r["matched"], "to_X": d.get("X"), "to_Y": d.get("Y")})
        print(f"{name:34s} {r['light']:6s} closest part, all parts: to X {d.get('X')}  to Y {d.get('Y')}", flush=True)
        json.dump({"X": X, "Y": Y, "nonce": NONCE, "rows": rows}, open(out, "w"), indent=1)
    if os.environ.get("LIGHT_ACT"):
        act = activations(where)
        for r in rows:
            r["act_to_X"], r["act_to_Y"] = act[r["model"]]
            print(f"{r['model']:34s} activations: to X {r['act_to_X']}  to Y {r['act_to_Y']}", flush=True)
        json.dump({"X": X, "Y": Y, "nonce": NONCE, "rows": rows}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
