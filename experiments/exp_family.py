"""The two retraining attacks on any pair of models that share a tokenizer, whatever their architecture.

Earlier rounds ran on one family (SmolLM2 against SmolLM), with code that knew that family's module names. This
script reads a model the way the tool does (`universal.layout`: matrices found by shape, not by name) and builds
the same tables in a form that can be trained against. So the attack works on whatever the tool can read.

  erase   keep X's outputs, push every table away from X's            aim: leave Red
  forge   the same, and pull every table toward Y's                   aim: reach Green

  loss = KL(X || student) + PUSH * mean relu(c_X - (1 - MARGIN)) [+ PULL * mean (1 - c_Y)]

X plays the listed model and Y the known one (RT_MODEL, RT_INDEP). At every evaluation the row holds the tool's own
light on a secret nonce, its distances to X and Y, the student's outputs against X and Y (total variation on
held-out windows), the divergence from X's outputs and a held-out task score.

Usage: python exp_family.py <out.jsonl> erase|forge        (FAM_STEPS, FAM_EVAL, FAM_LR, FAM_BATCH, RT_TAG, RT_THREADS)
"""
import json, os, sys, time
import torch
import exp_adaptive as A
import exp_capability as C
import exp_output_check as O
import exp_round3 as R3
from exp_adaptive_light import AUDIT_NONCE, Live, _resid
from lineage_check import light as LT
from lineage_check import universal as U

torch.set_grad_enabled(True)
STEPS, EVAL = int(os.environ.get("FAM_STEPS", 2000)), int(os.environ.get("FAM_EVAL", 250))
LR, BATCH = float(os.environ.get("FAM_LR", 1e-4)), int(os.environ.get("FAM_BATCH", 8))
PUSH, PULL, MARGIN, N_TOK, TASK = 10.0, 10.0, 0.6, 128, 300
DEV = O.DEV


def plan(model):
    """What the tool reads in this model: the embedding table and, per layer, each matrix with its kind and gain."""
    emb, _, lay = U.layout(Live(model))
    out = []
    for i in sorted(lay):
        L, mats = lay[i], []
        for n, kind in L["mats"]:
            short = U.LAYER.search(n).group(2)
            gain = None
            if kind == "r":
                gain = L["g1"] if (U.ATTN.search(short) and L["g1"]) else L["g2"] if (U.MLP.search(short) and L["g2"]) else None
            mats.append((n, kind, gain))
        out.append(mats)
    return emb, out


def tables(model, emb, layers, ids):
    """Per layer: the residual tables of the readers and of the writers, as `universal.fingerprint` builds them,
    kept differentiable."""
    P = dict(model.named_parameters())
    E = P[emb][ids.to(P[emb].device)]
    e = U._basis((E @ E.T).detach().cpu()).to(E.device, E.dtype)
    out = []
    for mats in layers:
        r, w = [], []
        for n, kind, gain in mats:
            W = P[n]
            if kind == "r":
                if gain:
                    W = W * P[gain][None, :]
                a = (W / W.norm(dim=1, keepdim=True).clamp_min(1e-12)) @ E.T
                r.append(_resid(a.T @ a, e))
            else:
                b = E @ (W / W.norm(dim=0, keepdim=True).clamp_min(1e-12))
                w.append(_resid(b @ b.T, e))
        out.append((torch.stack(r) if r else None, torch.stack(w) if w else None))
    return out


def correlations(ta, tb):
    """For every table, its best match in the same layer and kind of the other model: what the tool scores."""
    c = []
    for pa, pb in zip(ta, tb):
        for a, b in zip(pa, pb):
            if a is not None and b is not None:
                S = a @ b.T
                c += [S.max(1).values, S.max(0).values]
    return torch.cat(c)


@torch.no_grad()
def task(m):
    ok = 0
    for ex in C.lam[:TASK]:
        ctx, word = ex["text"].rsplit(" ", 1)
        c, w = C.tok(ctx, add_special_tokens=False)["input_ids"], C.tok(" " + word, add_special_tokens=False)["input_ids"]
        pred = m(torch.tensor([c + w], device=DEV)).logits[0, len(c) - 1: len(c) + len(w) - 1].argmax(-1).tolist()
        ok += pred == w
    return ok / min(TASK, len(C.lam))


def main(out, mode):
    torch.set_num_threads(int(os.environ.get("RT_THREADS", 4)))
    x, y, ids = O.load(A.REPO), O.load(A.INDEP), O.probes()
    for ref in (x, y):
        ref.requires_grad_(False)
    m = O.load(A.REPO).train()
    emb, layers = plan(m)
    watch, known, cache = [({"id": "X"}, Live(x))], [({"id": "Y"}, Live(y))], {}
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.0)
    g = torch.Generator().manual_seed(41)
    rec = {"run": mode, "X": A.REPO, "Y": A.INDEP, "lr": LR, "batch": BATCH, "push": PUSH, "pull": PULL if mode == "forge" else 0,
           "margin": MARGIN, "matrices_per_layer": len(layers[0]), "layers": len(layers)}

    @torch.no_grad()
    def measure(step, seconds, train=None):
        m.eval()
        src = Live(m)
        r = LT.light(src, watch, known, AUDIT_NONCE, cache=cache)
        d = dict(r["distances"])
        if "Y" not in d:
            d["Y"] = LT.distances(src, known[0][1], AUDIT_NONCE, cache=cache)
        for k in [k for k in cache if k[0] == id(src)]:
            del cache[k]
        (tv_y, _), (tv_x, kl_x) = O.apart([y, x], m, ids)
        rd = lambda v: None if v is None else [round(float(v[0]), 4), round(float(v[1]), 4)]
        row = {**rec, "step": step, "sec": seconds, "light": r["light"], "to_X": rd(d.get("X")), "to_Y": rd(d.get("Y")),
               "tv_to_X": round(float(tv_x.mean()), 4), "tv_to_Y": round(float(tv_y.mean()), 4), "kl_from_X": round(float(kl_x.mean()), 4),
               "task": task(m), **(train or {})}
        print(json.dumps(row), flush=True)
        with open(out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        m.train()

    t0 = time.time()
    measure(0, 0)
    for step in range(1, STEPS + 1):
        b = R3.FW_TRAIN[torch.randint(0, R3.FW_TRAIN.shape[0], (BATCH,), generator=g)].to(DEV)[:, :-1]
        with torch.no_grad():
            target = torch.log_softmax(x(b).logits.float(), -1)
        logits = m(b).logits
        keep = torch.nn.functional.kl_div(torch.log_softmax(logits.float(), -1), target, log_target=True, reduction="batchmean") / logits.shape[1]
        S = A.POOL[torch.randperm(len(A.POOL), generator=g)[:N_TOK]]
        ts = tables(m, emb, layers, S)
        with torch.no_grad():
            tx = tables(x, emb, layers, S)
            ty = tables(y, emb, layers, S) if mode == "forge" else None
        cx = correlations(ts, tx)
        loss = keep + PUSH * torch.relu(cx - (1 - MARGIN)).mean()
        cy = None
        if mode == "forge":
            cy = correlations(ts, ty)
            loss = loss + PULL * (1 - cy).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        O.free()
        if step % EVAL == 0:
            measure(step, round(time.time() - t0), {"train_kl": round(float(keep), 4), "train_c_X": round(float(cx.mean()), 4),
                                                    "train_c_Y": None if cy is None else round(float(cy.mean()), 4)})
    save = os.path.join(A.SAVE, f"family-{mode}")
    m.save_pretrained(save, safe_serialization=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
