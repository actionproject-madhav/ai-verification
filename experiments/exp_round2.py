"""Red team round 2: a white-box attacker against v4 (the manifold fingerprint), and against v3 + v4 together.

Model: SmolLM2-135M. Capability = held-out cross-entropy on WikiText-2 test.
The attacker fine-tunes on WikiText-2 train with loss = CE - lam4 * D_v4(surrogate probes) - lam3 * D_v3(random tokens).
It does not know the audit's secret probe text: it uses its own random WikiText windows as surrogate probes.
Evaluation uses the secret probe set (WikiText-103 validation documents chosen by the audit nonce, as in exp_v4),
and v3 on a secret token set.

Usage: python exp_round2.py <out.jsonl> [v4 | both | v4-known] ...
"""
import json, os, sys, time
import numpy as np
import torch
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from lineage_check import fingerprint as fp4
import exp_adaptive as A
import exp_v4 as V
from lineage_check import paths

torch.set_grad_enabled(True)                                   # exp_v4 turns gradients off at import

DEV = A.DEV
STEPS, EVAL_EVERY, PB, PT = 400, 40, 4, 64
SAVE = paths.ADAPTIVE
tok = A.tok
SECRET_IDS = V.ids_for(tok)                                   # (8, 72) secret probe text
LAYERS = None


def surrogate_batch(g):
    """The attacker's own probe text: random WikiText-2 train windows (it does not know the secret documents)."""
    b = A.TRAIN[torch.randint(0, A.TRAIN.shape[0], (PB,), generator=g)][:, :PT]
    return b


def collect_grad(model, ids, layers):
    """Like fp_act.collect, but keeps the graph so the attacker can backpropagate through activations."""
    acts = {l: {} for l in layers}
    hooks = []
    for l in layers:
        L = model.model.layers[l]

        def pre(m, args, kwargs, l=l):
            acts[l]["Z"] = args[0] if args else kwargs["hidden_states"]
        hooks.append(L.register_forward_pre_hook(pre, with_kwargs=True))
        hooks.append(L.self_attn.q_proj.register_forward_hook(lambda m, i, o, l=l: acts[l].__setitem__("X", i[0])))
        hooks.append(L.mlp.down_proj.register_forward_hook(lambda m, i, o, l=l: acts[l].__setitem__("A", i[0])))
    model(ids)
    for h in hooks:
        h.remove()
    return acts


def v4_live(model, ids, layers, grad=True):
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        acts = collect_grad(model, ids, layers)
        fp = {}
        for l in layers:
            q, k, v, o, down, H, Hkv, hd = V.fp_act._weights(model, l)
            G, d = H // Hkv, q.shape[1]
            sel = lambda t: t[:, 4:].reshape(-1, t.shape[-1]).float()
            X, Z, Ac = sel(acts[l]["X"]), sel(acts[l]["Z"]), sel(acts[l]["A"])
            Qs = q.reshape(Hkv, G, hd, d).sum(1).reshape(Hkv * hd, d)
            Os = o.reshape(o.shape[0], Hkv, G, hd).sum(2).reshape(o.shape[0], Hkv * hd)
            Og = Os.reshape(-1, Hkv, hd).permute(1, 2, 0)
            Vg = v.reshape(Hkv, hd, -1)
            wg = torch.einsum("gij,gji->g", Og @ Og.transpose(1, 2), Vg @ Vg.transpose(1, 2)).clamp_min(1e-12).sqrt()
            Ad = Ac * down.norm(dim=0)
            fp[l] = {"qk": ((X @ Qs.T) * wg.repeat_interleave(hd)) @ (k @ X.T), "ov": (Z @ Os) @ (v @ X.T),
                     "mlp": (Z @ down) @ Ac.T, "act": Ad @ Ad.T}
    return fp


def d_v4(fa, fb):
    t = [1 - A.corr32(fa[l][c], fb[l][c]) for l in fa for c in fa[l]]
    return torch.stack(t).mean()


@torch.no_grad()
def evaluate(m, parent_v4, parent_v3):
    m.eval()
    ce = float(np.mean([float(torch.nn.functional.cross_entropy(m(b[:, :-1].to(DEV)).logits.flatten(0, 1),
                                                                b[:, 1:].to(DEV).flatten())) for b in A.EVAL_BATCHES]))
    f4 = v4_live(m, SECRET_IDS.to(DEV), LAYERS, grad=False)
    v4 = {c: float(np.mean([1 - float(A.corr32(parent_v4[l][c], f4[l][c])) for l in f4])) for c in ("qk", "ov", "mlp", "act")}
    f3 = A.fp_live(m, A.S_EVAL)
    out = {"ce": ce, "v4_attn": float(np.mean([v4["qk"], v4["ov"]])), "v4_all": float(np.mean(list(v4.values()))),
           "v3_attn": float(A.dist(parent_v3, f3, ("qk", "ov")))}
    m.train()
    if DEV == "mps":
        torch.mps.empty_cache()
    return out


def run(fh, name, lam4, lam3, known, lr=1e-4):
    global LAYERS
    m, ref = A.load(), A.load()
    ref.requires_grad_(False)
    m.config.use_cache = False                                 # no checkpointing: the attacker needs activation gradients
    LAYERS = V.layers_of(m)
    with torch.no_grad():
        p4 = v4_live(ref, SECRET_IDS.to(DEV), LAYERS, grad=False)
        p3 = A.fp_live(ref, A.S_EVAL)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(13)
    rec = {"run": name, "lam4": lam4, "lam3": lam3, "known_probes": known, "lr": lr}
    A.log(fh, {**rec, "step": 0, **evaluate(m, p4, p3)})
    t0 = time.time()
    for step in range(1, STEPS + 1):
        b = A.TRAIN[torch.randint(0, A.TRAIN.shape[0], (A.BS,), generator=g)].to(DEV)
        ce = torch.nn.functional.cross_entropy(m(b[:, :-1]).logits.flatten(0, 1), b[:, 1:].flatten())
        loss = ce
        if lam4 > 0:
            probe = (SECRET_IDS[torch.randperm(SECRET_IDS.shape[0], generator=g)[:PB], :PT] if known
                     else surrogate_batch(g)).to(DEV)
            with torch.no_grad():
                pr = v4_live(ref, probe, LAYERS, grad=False)
            loss = loss - lam4 * d_v4(pr, v4_live(m, probe, LAYERS))
        if lam3 > 0:
            S = A.POOL[torch.randperm(len(A.POOL), generator=g)[:128]]
            with torch.no_grad():
                r3 = A.fp_live(ref, S)
            loss = loss - lam3 * A.dist(r3, A.fp_live(m, S))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if DEV == "mps":
            torch.mps.empty_cache()
        if step % EVAL_EVERY == 0:
            A.log(fh, {**rec, "step": step, "train_ce": float(ce), "sec": round(time.time() - t0), **evaluate(m, p4, p3)})
    m.save_pretrained(os.path.join(SAVE, name), safe_serialization=True)
    del m, ref


RUNS = {"v4": (10.0, 0.0, False), "both": (10.0, 10.0, False), "v4-known": (10.0, 0.0, True)}

if __name__ == "__main__":
    with open(sys.argv[1], "a") as fh:
        for n in sys.argv[2:] or ["v4", "both"]:
            run(fh, n, *RUNS[n])
