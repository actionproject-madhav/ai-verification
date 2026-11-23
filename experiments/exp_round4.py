"""Red team round 4: the function-preserving attacker. It distills the parent into itself on broad text
(KL between parent and student next-token distributions on FineWeb-Edu) while pushing both fingerprints away:
  loss = KL(parent || student) - lam * D_v4(surrogate probes) - lam * D_v3(random tokens)
This attacker keeps the parent's behaviour as close as it can, so it is the hardest test of whether evasion costs
capability. Scored like round 3 (v3, v4, LAMBADA, TinyStories, FineWeb) plus held-out KL to the parent.

Usage: python exp_round4.py <out.jsonl> [lam ...]
"""
import json, os, sys, time
import numpy as np
import torch
import exp_round3 as R3
import exp_round2 as R2
import exp_adaptive as A

torch.set_grad_enabled(True)
DEV = A.DEV
STEPS, EVAL_EVERY = int(os.environ.get("R4_STEPS", 400)), int(os.environ.get("R4_EVAL", 100))
F = torch.nn.functional
MB = 4  # KL in micro-batches: the same gradient as the full batch, half the peak logits memory


def kl(parent_logits, student_logits):
    p = F.log_softmax(parent_logits.float(), -1)
    q = F.log_softmax(student_logits.float(), -1)
    return (p.exp() * (p - q)).sum(-1).mean()


@torch.no_grad()
def held_out_kl(m, ref):
    vals = []
    for i in range(0, 64, 8):
        b = R3.FW_TEST[i:i + 8, :-1].to(DEV)
        vals.append(float(kl(ref(b).logits, m(b).logits)))
    if DEV == "mps":
        torch.mps.empty_cache()
    return float(np.mean(vals))


def run(fh, lam):
    m, ref = A.load(), A.load()
    ref.requires_grad_(False)
    m.config.use_cache = False
    R2.LAYERS = R2.V.layers_of(m)
    with torch.no_grad():
        p4 = R2.v4_live(ref, R2.SECRET_IDS.to(DEV), R2.LAYERS, grad=False)
        p3 = A.fp_live(ref, A.S_EVAL)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0.0)
    g = torch.Generator().manual_seed(19)
    rec = {"run": f"distill-lam{lam:g}", "lam": lam, "data": "fineweb-edu", "objective": "KL to parent"}
    A.log(fh, {**rec, "step": 0, **R3.evaluate(m, p4, p3), "kl": held_out_kl(m, ref)})
    t0 = time.time()
    for step in range(1, STEPS + 1):
        b = R3.FW_TRAIN[torch.randint(0, R3.FW_TRAIN.shape[0], (A.BS,), generator=g)].to(DEV)[:, :-1]
        opt.zero_grad()
        loss_kl = 0.0
        for chunk in b.split(MB):
            with torch.no_grad():
                pl = ref(chunk).logits
            part = kl(pl, m(chunk).logits) * (chunk.shape[0] / b.shape[0])
            part.backward()
            loss_kl += float(part)
            del pl, part
        loss = 0.0
        probe = R3.FW_TRAIN[torch.randint(0, R3.FW_TRAIN.shape[0], (R2.PB,), generator=g)][:, :R2.PT].to(DEV)
        with torch.no_grad():
            pr = R2.v4_live(ref, probe, R2.LAYERS, grad=False)
        loss = loss - lam * R2.d_v4(pr, R2.v4_live(m, probe, R2.LAYERS))
        S = A.POOL[torch.randperm(len(A.POOL), generator=g)[:128]]
        with torch.no_grad():
            r3 = A.fp_live(ref, S)
        loss = loss - lam * A.dist(r3, A.fp_live(m, S))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if DEV == "mps":
            torch.mps.empty_cache()
        if step % EVAL_EVERY == 0:
            A.log(fh, {**rec, "step": step, "train_kl": loss_kl, "sec": round(time.time() - t0),
                       **R3.evaluate(m, p4, p3), "kl": held_out_kl(m, ref)})
    m.save_pretrained(os.path.join(R2.SAVE, f"r4-distill-lam{lam:g}"), safe_serialization=True)
    del m, ref


if __name__ == "__main__":
    with open(sys.argv[1], "a") as fh:
        for lam in [float(x) for x in sys.argv[2:]] or [3.0]:
            run(fh, lam)
