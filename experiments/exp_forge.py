"""Red team round 5: forgery. Can an attacker make model X look like a specific unrelated model Y?

Rounds 1-4 erased a fingerprint (moved X away from itself). A forger wants more: a listed model that the check
reports as a copy of a harmless, known model. The student starts as X (SmolLM2-135M), keeps X's outputs
(KL to X on broad text) and pulls its v3 circuits towards Y's (SmolLM-135M v1: same shape and tokenizer,
independently trained) on random token sets. Scored on the secret audit tokens: distance to Y, distance to X,
capability.

  loss = KL(X || student) + lam * D_v3(student, Y on random tokens)

Usage: python exp_forge.py <out.jsonl> [lam ...]
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
STEPS, EVAL_EVERY = int(os.environ.get("R5_STEPS", 400)), int(os.environ.get("R5_EVAL", 100))
TARGET = "HuggingFaceTB/SmolLM-135M"
PUSH = float(os.environ.get("R5_PUSH", 0))   # > 0: also push away from X (erase and forge together)


def run(fh, lam):
    m, ref, tgt = A.load(), A.load(), A.load(TARGET)
    ref.requires_grad_(False)
    tgt.requires_grad_(False)
    m.config.use_cache = False
    R2.LAYERS = R2.V.layers_of(m)
    with torch.no_grad():
        p4 = R2.v4_live(ref, R2.SECRET_IDS.to(DEV), R2.LAYERS, grad=False)
        p3 = A.fp_live(ref, A.S_EVAL)
        t3 = A.fp_live(tgt, A.S_EVAL)
        t4 = R2.v4_live(tgt, R2.SECRET_IDS.to(DEV), R2.LAYERS, grad=False)

    @torch.no_grad()
    def to_target():
        f3 = A.fp_live(m, A.S_EVAL)
        f4 = R2.v4_live(m, R2.SECRET_IDS.to(DEV), R2.LAYERS, grad=False)
        v4 = float(np.mean([1 - float(A.corr32(t4[l][c], f4[l][c])) for l in f4 for c in ("qk", "ov")]))
        return {"v3_to_target": float(A.dist(t3, f3, ("qk", "ov"))), "v4_to_target": v4}

    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0.0)
    g = torch.Generator().manual_seed(29)
    rec = {"run": f"forge-lam{lam:g}" + (f"-push{PUSH:g}" if PUSH else ""), "lam": lam, "push": PUSH, "target": TARGET,
           "objective": "KL to X + pull v3 to Y" + (" + push v3 from X" if PUSH else "")}
    A.log(fh, {**rec, "step": 0, **R3.evaluate(m, p4, p3), **to_target(), "kl": R4.held_out_kl(m, ref)})
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
        S = A.POOL[torch.randperm(len(A.POOL), generator=g)[:128]]
        with torch.no_grad():
            y3 = A.fp_live(tgt, S)
            x3 = A.fp_live(ref, S) if PUSH else None
        fs = A.fp_live(m, S)
        pull = lam * A.dist(y3, fs)
        (pull - PUSH * A.dist(x3, fs) if PUSH else pull).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if DEV == "mps":
            torch.mps.empty_cache()
        if step % EVAL_EVERY == 0:
            A.log(fh, {**rec, "step": step, "train_kl": loss_kl, "train_pull": float(pull) / lam, "sec": round(time.time() - t0),
                       **R3.evaluate(m, p4, p3), **to_target(), "kl": R4.held_out_kl(m, ref)})
    m.save_pretrained(os.path.join(R2.SAVE, rec["run"].replace("forge", "r5-forge")), safe_serialization=True)
    del m, ref, tgt


if __name__ == "__main__":
    with open(sys.argv[1], "a") as fh:
        for lam in [float(x) for x in sys.argv[2:]] or [3.0]:
            run(fh, lam)
