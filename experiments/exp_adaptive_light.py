"""Red team round 7: an attacker who knows the shipped tool.

Rounds 1 to 6 attacked the design-aware fingerprint. The traffic light reads something else: one table per weight
matrix, with the embedding Gram projected out, flagged on the closest part and cleared only if all parts match.
Run through the tool, every model from rounds 1 to 5 is still Red (results/lights_redteam_135m.json). So this
round attacks the tool itself, with full knowledge of it. Only the audit's nonce is unknown: it picks the tokens
and the layers after the weights are fixed, so the attacker works on random tokens and on every layer.

  erase      keep X's outputs, push every table away from X's                    aim: leave Red
  forge      the same, and pull every table towards Y's                          aim: reach Green
  erase-all  erase, and push the two older readers away as round 4 did           aim: leave Red on every reader

  loss = KEEP * KL(X || student) + PUSH * mean relu(c_X - (1 - MARGIN)) [+ PULL * mean (1 - c_Y)]

KEEP (R7_KEEP, default 1) weighs keeping X's outputs. Lower values trade X's behaviour for closeness to Y's tables;
0 drops the term. A sweep over it traces how much of X is left at each distance from Green.

ACT (R7_ACT, default 0) adds a pull on what the layers compute: ACT * (activation reader's distance to Y) on fresh
probe text. With it the run asks whether a model can take on Y's activations as well as Y's tables and still give
X's outputs. R7_INIT starts from a saved model instead of X, to continue an earlier run.

c is the correlation the tool computes for each part: the best match within the same layer and kind. The verdict
logged at every evaluation is the tool's own (`lineage_check.light`), on the secret nonce.

Usage: python exp_adaptive_light.py <out.jsonl> erase|forge [...]
"""
import json, os, sys, time
import numpy as np
import torch
import exp_round3 as R3
import exp_round2 as R2
import exp_adaptive as A
import exp_round4 as R4
from lineage_check import light as LT
from lineage_check import universal as U

torch.set_grad_enabled(True)
DEV = A.DEV
STEPS, EVAL_EVERY = int(os.environ.get("R7_STEPS", 3000)), int(os.environ.get("R7_EVAL", 250))
PUSH, PULL, MARGIN = float(os.environ.get("R7_PUSH", 10)), float(os.environ.get("R7_PULL", 10)), float(os.environ.get("R7_MARGIN", 0.6))
LR, N_TOK, LAM = float(os.environ.get("R7_LR", 1e-4)), 128, float(os.environ.get("R7_LAM", 1))
KEEP, ACT, INIT = float(os.environ.get("R7_KEEP", 1)), float(os.environ.get("R7_ACT", 0)), os.environ.get("R7_INIT")
AUDIT_NONCE = "round7-secret-audit-nonce"
READERS = (("self_attn", "q_proj", "g1"), ("self_attn", "k_proj", "g1"), ("self_attn", "v_proj", "g1"),
           ("mlp", "gate_proj", "g2"), ("mlp", "up_proj", "g2"))
WRITERS = (("self_attn", "o_proj"), ("mlp", "down_proj"))


def _resid(G, Q):
    g = G.flatten()
    g = g - g.mean()
    r = g - Q @ (Q.T @ g)
    return r / r.norm().clamp_min(1e-30)


def tables(model, ids):
    """Per layer: the residual tables of the matrices that read the residual stream and of those that write to it.
    The same construction as `lineage_check.universal.fingerprint`, kept differentiable."""
    E = model.model.embed_tokens.weight[ids.to(model.model.embed_tokens.weight.device)]
    e = U._basis((E @ E.T).detach().cpu()).to(E.device, E.dtype)     # the tool's own projection, fixed within a step
    out = []
    for L in model.model.layers:
        gain = {"g1": L.input_layernorm.weight, "g2": L.post_attention_layernorm.weight}
        r, w = [], []
        for block, name, g in READERS:
            W = getattr(getattr(L, block), name).weight * gain[g][None, :]
            a = (W / W.norm(dim=1, keepdim=True).clamp_min(1e-12)) @ E.T
            r.append(_resid(a.T @ a, e))
        for block, name in WRITERS:
            W = getattr(getattr(L, block), name).weight
            b = E @ (W / W.norm(dim=0, keepdim=True).clamp_min(1e-12))
            w.append(_resid(b @ b.T, e))
        out.append((torch.stack(r), torch.stack(w)))
    return out


def correlations(ta, tb):
    """What the tool scores: for every table, its best match in the same layer and kind of the other model."""
    c = []
    for (ra, wa), (rb, wb) in zip(ta, tb):
        for a, b in ((ra, rb), (wa, wb)):
            S = a @ b.T
            c += [S.max(1).values, S.max(0).values]
    return torch.cat(c)


class Live:
    """A model in memory, behind the interface the tool reads weights through."""

    def __init__(self, model):
        self.state = model.state_dict()
        self.index = dict.fromkeys(self.state)
        self.vocab = A.vocab
        self.config = {"model_type": "live"}

    def shape(self, name):
        return tuple(self.state[name].shape)

    def get(self, name):
        return self.state[name].detach().float().cpu().numpy()

    def rows(self, name, lo, hi):
        return self.state[name][lo:hi].detach().float().cpu().numpy()


def run(fh, mode):
    m, ref, tgt = A.load(INIT) if INIT else A.load(), A.load(), A.load(A.INDEP)
    ref.requires_grad_(False)
    tgt.requires_grad_(False)
    m.config.use_cache = False
    R2.LAYERS = R2.V.layers_of(m)
    watch, known, cache = [({"id": "X"}, Live(ref))], [({"id": "Y"}, Live(tgt))], {}
    with torch.no_grad():
        p4 = R2.v4_live(ref, R2.SECRET_IDS.to(DEV), R2.LAYERS, grad=False)
        p4y = R2.v4_live(tgt, R2.SECRET_IDS.to(DEV), R2.LAYERS, grad=False)
        p3 = A.fp_live(ref, A.S_EVAL)

    @torch.no_grad()
    def behaviour():
        """Held-out KL from X and from Y, and the activation reader's distance to Y on the secret probe."""
        return {"kl": R4.held_out_kl(m, ref), "kl_to_Y": R4.held_out_kl(m, tgt),
                "act_to_Y": float(R2.d_v4(p4y, R2.v4_live(m, R2.SECRET_IDS.to(DEV), R2.LAYERS, grad=False)))}

    @torch.no_grad()
    def verdict():
        src = Live(m)
        r = LT.light(src, watch, known, AUDIT_NONCE, cache=cache)
        d = dict(r["distances"])
        if "Y" not in d:
            d["Y"] = LT.distances(src, known[0][1], AUDIT_NONCE, cache=cache)
        for k in [k for k in cache if k[0] == id(src)]:
            del cache[k]
        rd = lambda v: None if v is None else [round(float(v[0]), 4), round(float(v[1]), 4)]
        return {"light": r["light"], "to_X": rd(d.get("X")), "to_Y": rd(d.get("Y"))}

    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.0)
    g = torch.Generator().manual_seed(41)
    name = f"tool-{mode}" + ("" if KEEP == 1 else f"-keep{KEEP:g}") + (f"-act{ACT:g}" if ACT else "") + ("-continued" if INIT else "")
    rec = {"run": name, "mode": mode, "push": PUSH, "pull": PULL if mode == "forge" else 0, "margin": MARGIN, "lr": LR, "keep": KEEP,
           "act": ACT, "init": os.path.basename(os.path.dirname(INIT.rstrip("/"))) + "/" + os.path.basename(INIT.rstrip("/")) if INIT else None,
           "target": A.INDEP, "objective": ("KL to X + " if KEEP > 0 else "") + "push every table from X" + (" + pull every table to Y" if mode == "forge" else "")
           + (" + push the design-aware and activation fingerprints from X" if mode == "erase-all" else "")
           + (" + pull the activation fingerprint to Y" if ACT else "")}
    A.log(fh, {**rec, "step": 0, **verdict(), **R3.evaluate(m, p4, p3), **behaviour()})
    t0 = time.time()
    for step in range(1, STEPS + 1):
        b = R3.FW_TRAIN[torch.randint(0, R3.FW_TRAIN.shape[0], (A.BS,), generator=g)].to(DEV)[:, :-1]
        opt.zero_grad()
        loss_kl = 0.0
        for chunk in b.split(R4.MB) if KEEP > 0 else ():
            with torch.no_grad():
                pl = ref(chunk).logits
            part = KEEP * R4.kl(pl, m(chunk).logits) * (chunk.shape[0] / b.shape[0])
            part.backward()
            loss_kl += float(part)
            del pl, part
        S = A.POOL[torch.randperm(len(A.POOL), generator=g)[:N_TOK]]
        ts = tables(m, S)
        with torch.no_grad():
            tx = tables(ref, S)
            ty = tables(tgt, S) if mode == "forge" else None
        cx = correlations(ts, tx)
        loss = PUSH * torch.relu(cx - (1 - MARGIN)).mean()
        cy = None
        if mode == "forge":
            cy = correlations(ts, ty)
            loss = loss + PULL * (1 - cy).mean()
        if mode == "erase-all":
            probe = R3.FW_TRAIN[torch.randint(0, R3.FW_TRAIN.shape[0], (R2.PB,), generator=g)][:, :R2.PT].to(DEV)
            with torch.no_grad():
                pr, r3 = R2.v4_live(ref, probe, R2.LAYERS, grad=False), A.fp_live(ref, S)
            loss = loss - LAM * R2.d_v4(pr, R2.v4_live(m, probe, R2.LAYERS)) - LAM * A.dist(r3, A.fp_live(m, S))
        if ACT:
            probe = R3.FW_TRAIN[torch.randint(0, R3.FW_TRAIN.shape[0], (R2.PB,), generator=g)][:, :R2.PT].to(DEV)
            with torch.no_grad():
                py = R2.v4_live(tgt, probe, R2.LAYERS, grad=False)
            loss = loss + ACT * R2.d_v4(py, R2.v4_live(m, probe, R2.LAYERS))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if DEV == "mps":
            torch.mps.empty_cache()
        if step % EVAL_EVERY == 0:
            A.log(fh, {**rec, "step": step, "train_kl": loss_kl, "train_c_X_max": float(cx.max()), "train_c_X_mean": float(cx.mean()),
                       "train_c_Y_mean": None if cy is None else float(cy.mean()), "sec": round(time.time() - t0),
                       **verdict(), **R3.evaluate(m, p4, p3), **behaviour()})
    m.save_pretrained(os.path.join(R2.SAVE, f"r7-{name}"), safe_serialization=True)
    del m, ref, tgt


if __name__ == "__main__":
    with open(sys.argv[1], "a") as fh:
        for mode in sys.argv[2:] or ["erase", "forge"]:
            run(fh, mode)
