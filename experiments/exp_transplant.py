"""Red team round 6: transplant. Can a forger wear a known model's attention circuits over a listed model?

The verdict of the weight fingerprint reads attention circuits anchored on token embeddings. So the forger copies
the embeddings, the attention weights and the attention norm gains of a known model Y (SmolLM-135M v1) into the
listed model X (SmolLM2-135M: same shape, same tokenizer, trained separately), freezes them, and retrains every
other weight to behave like X again:

  loss = KL(X || student)        only the MLPs, the MLP norm gains and the final norm are trained

By construction the student's attention circuits are Y's. The questions are how much of X the rest of the network
recovers (KL to X, LAMBADA, FineWeb loss), and what the parts the forger did not copy say: the MLP tables and the
activation fingerprint, against Y and against X.

Usage: python exp_transplant.py <out.jsonl> [lr ...]
"""
import json, os, sys, time
import numpy as np
import torch
import exp_round3 as R3
import exp_round2 as R2
import exp_adaptive as A
import exp_round4 as R4

torch.set_grad_enabled(True)
DEV = A.DEV
STEPS, EVAL_EVERY = int(os.environ.get("R6_STEPS", 2000)), int(os.environ.get("R6_EVAL", 250))
TARGET = A.INDEP
COPIED = ("embed_tokens", "self_attn", "input_layernorm")     # everything the attention verdict reads
PARTS = ("qk", "ov", "mlp", "gate")


def parts(a, b):
    return {c: round(float(A.dist(a, b, (c,))), 4) for c in PARTS}


def run(fh, lr):
    m, ref, tgt = A.load(), A.load(), A.load(TARGET)
    ref.requires_grad_(False)
    tgt.requires_grad_(False)
    m.config.use_cache = False
    R2.LAYERS = R2.V.layers_of(m)
    with torch.no_grad():
        p4 = R2.v4_live(ref, R2.SECRET_IDS.to(DEV), R2.LAYERS, grad=False)
        t4 = R2.v4_live(tgt, R2.SECRET_IDS.to(DEV), R2.LAYERS, grad=False)
        p3, t3 = A.fp_live(ref, A.S_EVAL), A.fp_live(tgt, A.S_EVAL)

    @torch.no_grad()
    def status(model):
        f3 = A.fp_live(model, A.S_EVAL)
        f4 = R2.v4_live(model, R2.SECRET_IDS.to(DEV), R2.LAYERS, grad=False)
        v4 = {c: float(np.mean([1 - float(A.corr32(t4[l][c], f4[l][c])) for l in f4])) for c in ("qk", "ov", "mlp", "act")}
        return {"to_target": parts(t3, f3), "to_parent": parts(p3, f3), "v4_to_target": float(np.mean([v4["qk"], v4["ov"]])),
                "v4_to_target_all": float(np.mean(list(v4.values()))), "kl_to_parent": R4.held_out_kl(model, ref),
                "kl_to_target": R4.held_out_kl(model, tgt), **R3.evaluate(model, p4, p3)}

    rec = {"run": f"transplant-lr{lr:g}", "lr": lr, "target": TARGET, "copied": list(COPIED), "objective": "KL to X, copied weights frozen"}
    A.log(fh, {**rec, "step": -1, "what": "X, the listed model", **status(ref)})
    A.log(fh, {**rec, "step": -1, "what": "Y, the known model", **status(tgt)})
    with torch.no_grad():
        sd, td = m.state_dict(), tgt.state_dict()
        for k in sd:
            if any(c in k for c in COPIED):
                sd[k].copy_(td[k])
    train = []
    for n, p in m.named_parameters():
        p.requires_grad_(not any(c in n for c in COPIED))
        if p.requires_grad:
            train.append(p)
    rec["trained_params"] = sum(p.numel() for p in train)
    rec["frozen_params"] = sum(p.numel() for p in m.parameters()) - rec["trained_params"]
    opt = torch.optim.AdamW(train, lr=lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(31)
    A.log(fh, {**rec, "step": 0, **status(m)})
    t0 = time.time()
    for step in range(1, STEPS + 1):
        b = R3.FW_TRAIN[torch.randint(0, R3.FW_TRAIN.shape[0], (A.BS,), generator=g)].to(DEV)[:, :-1]
        opt.zero_grad()
        loss_kl = 0.0
        for chunk in b.split(R4.MB):
            with torch.no_grad():
                pl = ref(chunk).logits
            part = R4.kl(pl, m(chunk).logits) * (chunk.shape[0] / b.shape[0])
            part.backward()
            loss_kl += float(part)
            del pl, part
        torch.nn.utils.clip_grad_norm_(train, 1.0)
        opt.step()
        if DEV == "mps":
            torch.mps.empty_cache()
        if step % EVAL_EVERY == 0:
            A.log(fh, {**rec, "step": step, "train_kl": loss_kl, "sec": round(time.time() - t0), **status(m)})
    m.save_pretrained(os.path.join(R2.SAVE, f"r6-transplant-lr{lr:g}"), safe_serialization=True)
    del m, ref, tgt


if __name__ == "__main__":
    with open(sys.argv[1], "a") as fh:
        for lr in [float(x) for x in sys.argv[2:]] or [3e-4]:
            run(fh, lr)
