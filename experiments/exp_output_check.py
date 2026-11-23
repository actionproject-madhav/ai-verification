"""Would a check on outputs tell a forged copy from the known model it imitates?

Rounds 5 to 7 pull a listed model X toward a known model Y: first its weight tables, then what its layers compute.
Both move while the model keeps X's outputs. Outputs are different in kind. Total variation between next-token
distributions is a metric, so for any model m and any text

    TV(m, X) >= TV(X, Y) - TV(m, Y).

A model whose outputs are within eps of Y's is at least TV(X, Y) - eps from X's: it cannot pass as Y on the text
it is probed with and still behave as X there. This script measures the two distances on held-out text chosen by
a nonce, for X, Y, honest copies of Y (weights stored in blocks at 8 and at 4 bits, two fine-tunes, published
GGUF files) and the models the stress tests saved. Every distance is a mean over text windows and comes with its standard error across them
and with its lowest and highest window: the inequality holds window by window, so a check can ask for every window.
It is an experiment, not part of the tool: the tool never runs a model.

Usage: python exp_output_check.py <out.json> [name=path ...]
       A path is a saved model, a Hub repository, or gguf:<repository>:<file label> for one GGUF file of Y.
       A second call with the same file adds the models it does not hold yet.
       OUT_NONCE sets the nonce, OUT_WINDOWS the number of windows, OUT_TUNE the fine-tune steps, OUT_DEV the device
"""
import hashlib, json, os, sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM
import exp_adaptive as A
import exp_round3 as R3
from exp_adaptive_light import Live
from lineage_check import light as LT
from lineage_check import remote as R

NONCE = os.environ.get("OUT_NONCE", "output-check-2026-11")
WINDOWS, TUNE = int(os.environ.get("OUT_WINDOWS", 64)), [int(x) for x in os.environ.get("OUT_TUNE", "200,1000").split(",") if x]
DEV = os.environ.get("OUT_DEV", "cpu")                         # small models; the default leaves the GPU to whatever else runs


def free():
    """Hand cached GPU memory back: on a Mac the pool otherwise grows by gigabytes over a run."""
    if DEV == "mps":
        torch.mps.empty_cache()


def load(path):
    return AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to(DEV).eval()


def probes():
    """Held-out text windows picked by the nonce."""
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(("probes" + NONCE).encode()).digest()[:8], "little"))
    return R3.FW_TEST[torch.as_tensor(rng.choice(R3.FW_TEST.shape[0], size=WINDOWS, replace=False))][:, :-1]


@torch.no_grad()
def apart(refs, m, ids, batch=4):
    """For each reference model, per text window: total variation and KL(reference || m) between next-token
    distributions, averaged over positions. Worked out batch by batch: a full set of distributions would not fit in
    memory next to anything else."""
    out = [([], []) for _ in refs]
    for i in range(0, len(ids), batch):
        b = ids[i:i + batch].to(DEV)
        c = torch.log_softmax(m(b).logits.float(), -1)
        pc = c.exp()
        for (tv, kl), ref in zip(out, refs):
            a = torch.log_softmax(ref(b).logits.float(), -1)
            pa = a.exp()
            tv += (0.5 * (pa - pc).abs().sum(-1)).mean(1).tolist()
            kl += (pa * (a - c)).sum(-1).mean(1).tolist()
            del a, pa
        del c, pc
        free()
    return [(np.array(tv), np.array(kl)) for tv, kl in out]


def rounded(bits, block=32):
    """Y with every weight matrix stored the way block quantisers store it: runs of `block` numbers, each run
    rounded to 2**bits levels between its own lowest and highest value."""
    m = load(A.INDEP)
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() == 2:
                flat = p.reshape(-1)
                n = flat.numel() // block * block
                b = flat[:n].reshape(-1, block)
                lo, hi = b.amin(1, keepdim=True), b.amax(1, keepdim=True)
                step = ((hi - lo) / (2 ** bits - 1)).clamp_min(1e-12)
                flat[:n] = (((b - lo) / step).round() * step + lo).reshape(-1)
    return m


def from_gguf(repo, want):
    """Y's architecture filled with the weights of one published GGUF file, decoded by the tool's own reader."""
    src = R.Remote(repo, file=want)
    m = load(A.INDEP)
    heads = {"q_proj.weight": m.config.num_attention_heads, "k_proj.weight": m.config.num_key_value_heads}
    src.prefetch([k for k in m.state_dict() if src.has(k)])    # the whole file in a few requests
    with torch.no_grad():
        for k, p in m.state_dict().items():
            if src.has(k):
                w = np.ascontiguousarray(src.get(k), dtype=np.float32).reshape(p.shape)
                h = heads.get(k.rsplit(".", 2)[-2] + ".weight") if k.endswith("_proj.weight") else None
                if h:                                           # llama.cpp stores Llama query and key rows with the two
                    half = w.shape[0] // h // 2                 # halves of each head interleaved; put them back
                    w = w.reshape(h, half, 2, w.shape[1]).swapaxes(1, 2).reshape(w.shape)
                p.copy_(torch.from_numpy(np.ascontiguousarray(w)))
    return m


def saved(spec):
    kind, _, rest = spec.partition(":")
    return (lambda: from_gguf(*rest.rsplit(":", 1))) if kind == "gguf" else (lambda: load(spec))


def tuned(steps):
    """Y after a short honest fine-tune on the same text the stress tests train on."""
    m = load(A.INDEP).train()
    opt = torch.optim.AdamW(m.parameters(), lr=2e-5, weight_decay=0.0)
    g = torch.Generator().manual_seed(7)
    for _ in range(steps):
        b = R3.FW_TRAIN[torch.randint(0, R3.FW_TRAIN.shape[0], (8,), generator=g)].to(DEV)
        loss = torch.nn.functional.cross_entropy(m(b[:, :-1]).logits.flatten(0, 1), b[:, 1:].flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
        free()
    return m.eval()


def main(out, *more):
    torch.set_num_threads(int(os.environ.get("RT_THREADS", 4)))
    x, y, ids = load(A.REPO), load(A.INDEP), probes()
    watch, known, cache = [({"id": "X"}, Live(x))], [({"id": "Y"}, Live(y))], {}
    cases = [("X, the listed model", lambda: x), ("Y, the known model", lambda: y),
             ("Y, stored in blocks at 8 bits", lambda: rounded(8)), ("Y, stored in blocks at 4 bits", lambda: rounded(4))]
    cases += [(f"Y, fine-tuned for {n:,} steps", (lambda n: lambda: tuned(n))(n)) for n in TUNE]
    cases += [(s.split("=", 1)[0], saved(s.split("=", 1)[1])) for s in more]
    rows, gap = [], apart([y], x, ids)[0][0]                    # TV(X, Y) in each window
    if os.path.exists(out):
        old = json.load(open(out))
        if (old["nonce"], old["windows"], old["X"], old["Y"]) == (NONCE, WINDOWS, A.REPO, A.INDEP):
            rows = old["rows"]                                  # a second call adds models to the same file
    for name, make in cases:
        if any(r["model"] == name for r in rows):
            continue
        try:
            m = make()
        except Exception as e:                                  # a model that cannot be fetched is left out
            print(f"{name[:46]:46s} left out: {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        src = Live(m)                                           # one object per model: the tool's cache is keyed by it
        with torch.no_grad():
            r = LT.light(src, watch, known, NONCE, cache=cache)
            d = dict(r["distances"])
            if "Y" not in d:
                d["Y"] = LT.distances(src, known[0][1], NONCE, cache=cache)
        for k in [k for k in cache if k[0] == id(src)]:        # keep the two list entries, drop this model
            del cache[k]
        (tv_y, kl_y), (tv_x, kl_x) = apart([y, x], m, ids)
        se = lambda v: round(float(v.std(ddof=1) / len(v) ** 0.5), 4)
        span = lambda v: [round(float(v.min()), 4), round(float(v.max()), 4)]     # the lowest and the highest window
        rows.append({"model": name, "light": r["light"], "matched": r["matched"],
                     "tables_to_Y": None if d.get("Y") is None else [round(float(v), 4) for v in d["Y"]],
                     "tv_to_Y": round(float(tv_y.mean()), 4), "tv_to_Y_se": se(tv_y), "tv_to_Y_windows": span(tv_y),
                     "tv_to_X": round(float(tv_x.mean()), 4), "tv_to_X_se": se(tv_x), "tv_to_X_windows": span(tv_x),
                     "kl_from_Y": round(float(kl_y.mean()), 4), "kl_from_X": round(float(kl_x.mean()), 4),
                     "bound_holds_in_every_window": bool((tv_y + tv_x >= gap - 1e-6).all())})
        print(f"{name[:46]:46s} {r['light']:5s} tables to Y {rows[-1]['tables_to_Y']}  outputs: TV to Y {tv_y.mean():.3f}  TV to X {tv_x.mean():.3f}", flush=True)
        json.dump({"nonce": NONCE, "windows": WINDOWS, "tokens_per_window": int(ids.shape[1]), "X": A.REPO, "Y": A.INDEP,
                   "tv_X_Y": rows[0]["tv_to_Y"], "tv_X_Y_se": rows[0]["tv_to_Y_se"], "tv_X_Y_windows": rows[0]["tv_to_Y_windows"], "rows": rows}, open(out, "w"), indent=1)
        if m is not x and m is not y:
            del m
        free()


if __name__ == "__main__":
    main(sys.argv[1], *sys.argv[2:])
