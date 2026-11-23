"""Red team round 3: a stronger attacker. Broad data (FineWeb-Edu, SmolLM2's own pretraining distribution),
milder penalties so it can trade evasion against capability, and general-capability checks at every evaluation.

  honest : fine-tune on FineWeb-Edu, no penalty (the baseline for what fine-tuning costs or gains)
  joint  : same data, loss = CE - 3 * D_v4(surrogate probes) - 3 * D_v3(random tokens)

Usage: python exp_round3.py <out.jsonl> [honest|joint ...]
"""
import json, os, sys, time
import numpy as np
import torch
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
import exp_round2 as R2
import exp_adaptive as A
import exp_capability as C

torch.set_grad_enabled(True)
DEV = A.DEV
STEPS, EVAL_EVERY = int(os.environ.get("R3_STEPS", 600)), 200
p = hf_hub_download("HuggingFaceFW/fineweb-edu", "sample/10BT/013_00000.parquet", repo_type="dataset")
texts = pq.read_table(p, columns=["text"]).column("text").to_pylist()[:6000]
ids = torch.tensor(A.tok("\n\n".join(texts))["input_ids"])
n = ids.numel() // (A.SEQ + 1)
FW = ids[: n * (A.SEQ + 1)].view(n, A.SEQ + 1)
FW_TRAIN, FW_TEST = FW[:-200], FW[-200:]
print(f"FineWeb-Edu: {n} windows of {A.SEQ + 1} tokens", flush=True)


@torch.no_grad()
def evaluate(m, p4, p3):
    m.eval()
    base = R2.evaluate(m, p4, p3)
    C.DEV = DEV
    C.lam[:] = C.lam[:500]
    fw = float(np.mean([float(torch.nn.functional.cross_entropy(m(FW_TEST[i:i + 16, :-1].to(DEV)).logits.flatten(0, 1),
                                                                FW_TEST[i:i + 16, 1:].to(DEV).flatten())) for i in range(0, 200, 16)]))
    out = {**base, "lambada": C.lambada(m), "tinystories_ce": C.tinystories(m), "fineweb_ce": fw}
    m.train()
    return out


def run(fh, name, lam):
    m, ref = A.load(), A.load()
    ref.requires_grad_(False)
    m.config.use_cache = False
    R2.LAYERS = R2.V.layers_of(m)
    with torch.no_grad():
        p4 = R2.v4_live(ref, R2.SECRET_IDS.to(DEV), R2.LAYERS, grad=False)
        p3 = A.fp_live(ref, A.S_EVAL)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0.0)
    g = torch.Generator().manual_seed(17)
    rec = {"run": name, "lam": lam, "data": "fineweb-edu"}
    A.log(fh, {**rec, "step": 0, **evaluate(m, p4, p3)})
    t0 = time.time()
    for step in range(1, STEPS + 1):
        b = FW_TRAIN[torch.randint(0, FW_TRAIN.shape[0], (A.BS,), generator=g)].to(DEV)
        ce = torch.nn.functional.cross_entropy(m(b[:, :-1]).logits.flatten(0, 1), b[:, 1:].flatten())
        loss = ce
        if lam > 0:
            probe = FW_TRAIN[torch.randint(0, FW_TRAIN.shape[0], (R2.PB,), generator=g)][:, :R2.PT].to(DEV)
            with torch.no_grad():
                pr = R2.v4_live(ref, probe, R2.LAYERS, grad=False)
            loss = loss - lam * R2.d_v4(pr, R2.v4_live(m, probe, R2.LAYERS))
            S = A.POOL[torch.randperm(len(A.POOL), generator=g)[:128]]
            with torch.no_grad():
                r3 = A.fp_live(ref, S)
            loss = loss - lam * A.dist(r3, A.fp_live(m, S))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if DEV == "mps":
            torch.mps.empty_cache()
        if step % EVAL_EVERY == 0:
            A.log(fh, {**rec, "step": step, "train_ce": float(ce), "sec": round(time.time() - t0), **evaluate(m, p4, p3)})
    m.save_pretrained(os.path.join(R2.SAVE, f"r3-{name}"), safe_serialization=True)
    del m, ref


if __name__ == "__main__":
    with open(sys.argv[1], "a") as fh:
        for n_ in sys.argv[2:] or ["joint", "honest"]:
            run(fh, n_, 3.0 if n_ == "joint" else 0.0)
