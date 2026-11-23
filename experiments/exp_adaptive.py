"""Adaptive attacks on the lineage fingerprint (white-box: the attacker knows the method).

Model: SmolLM2-135M (the 'model of concern'). Capability = held-out cross-entropy on WikiText-2 test.
Identity = fingerprint distance to the parent on a SECRET token set S_eval (512 tokens from an audit nonce).

A  token scrambling: permute embedding rows (tied, so unembedding too) inside a random fraction of the token pool,
   or inside exactly S_eval (the nonce leaked).
B  gradient evasion: fine-tune on WikiText-2 with loss = CE - lambda * D(theta, parent; S_train).
     secret-S: S_train is a fresh random 128-token subset of the public pool every step
     known-S : S_train is drawn from S_eval itself (worst case: the audit nonce leaked)
   lambda = 0 is ordinary fine-tuning (how far legitimate training moves the fingerprint).

Usage: python exp_adaptive.py <out.jsonl> [run ...]   runs: scramble, ft, ft-hi, sec1, sec10, sec10-hi, sec100, known10
"""
import json, math, os, sys, time
import numpy as np
import torch
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from lineage_check import fingerprint as fp4
from lineage_check import paths

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
REPO, INDEP = "HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM-135M"
SEQ, BS, STEPS, EVAL_EVERY, N_EVAL_BATCH = 128, 8, 400, 40, 24
SAVE = paths.ADAPTIVE
torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(REPO)
vocab = tok.get_vocab()


def text_ids(split):
    p = hf_hub_download("Salesforce/wikitext", f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet", repo_type="dataset")
    txt = "".join(t for t in pq.read_table(p).column("text").to_pylist())
    ids = torch.tensor(tok(txt)["input_ids"])
    n = ids.numel() // (SEQ + 1)
    return ids[: n * (SEQ + 1)].view(n, SEQ + 1)


TRAIN, TEST = text_ids("train"), text_ids("test")
EVAL_BATCHES = TEST[torch.randperm(TEST.shape[0], generator=torch.Generator().manual_seed(1))[: N_EVAL_BATCH * BS]].view(N_EVAL_BATCH, BS, SEQ + 1)
POOL = torch.tensor([vocab[s] for s in fp4.token_pool(vocab, 300, 20000)])
S_EVAL = torch.tensor([vocab[s] for s in fp4.select_strings(vocab, "secret-audit-nonce", 512)])


def load(repo=REPO, dev=None):
    return AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32).to(dev or DEV)


def view(m):
    c = m.config
    return {"H": c.num_attention_heads, "Hkv": c.num_key_value_heads, "hd": c.hidden_size // c.num_attention_heads,
            "layers": {i: {"q": L.self_attn.q_proj.weight, "k": L.self_attn.k_proj.weight, "v": L.self_attn.v_proj.weight,
                           "o": L.self_attn.o_proj.weight, "gate": L.mlp.gate_proj.weight, "up": L.mlp.up_proj.weight,
                           "down": L.mlp.down_proj.weight, "g1": L.input_layernorm.weight,
                           "g2": L.post_attention_layernorm.weight} for i, L in enumerate(m.model.layers)}}


def fp_live(m, ids):
    dev = m.model.embed_tokens.weight.device
    ids = ids.to(dev)
    E = m.model.embed_tokens.weight[ids]
    U = m.lm_head.weight[ids] * m.model.norm.weight
    v = view(m)
    return {"EE": E @ E.T, "EU": E @ U.T, "layers": {i: fp4.layer_comps(l, E, v, dev) for i, l in v["layers"].items()}}


def to_dev(fp):
    return {"EE": fp["EE"].to(DEV), "EU": fp["EU"].to(DEV),
            "layers": {i: {c: M.to(DEV) for c, M in L.items()} for i, L in fp["layers"].items()}}


def corr32(a, b):
    a, b = a.flatten(), b.flatten()
    a, b = a - a.mean(), b - b.mean()
    return (a @ b) / (a.norm() * b.norm() + 1e-20)


def dist(fa, fb, comps=None):
    t = [] if comps else [1 - corr32(fa["EE"], fb["EE"]), 1 - corr32(fa["EU"], fb["EU"])]
    for i in fa["layers"]:
        for c, M in fa["layers"][i].items():
            if comps is None or c in comps:
                t.append(1 - corr32(M, fb["layers"][i][c]))
    return torch.stack(t).mean()


@torch.no_grad()
def evaluate(m, parent_fp_eval):
    m.eval()
    ce = np.mean([float(torch.nn.functional.cross_entropy(m(b[:, :-1].to(DEV)).logits.flatten(0, 1),
                                                          b[:, 1:].to(DEV).flatten())) for b in EVAL_BATCHES])
    f = fp_live(m, S_EVAL)
    out = {"ce": float(ce), "d": float(dist(parent_fp_eval, f)), "d_attn": float(dist(parent_fp_eval, f, ("qk", "ov")))}
    m.train()
    if DEV == "mps":
        torch.mps.empty_cache()
    return out


def log(fh, rec):
    fh.write(json.dumps(rec) + "\n")
    fh.flush()
    print(json.dumps(rec), flush=True)


def run_scramble(fh, parent, pfp):
    base = {k: v.detach().clone() for k, v in parent.state_dict().items()}
    g = torch.Generator().manual_seed(5)
    cases = [("pool", f) for f in (0.01, 0.03, 0.1, 0.3, 0.6)] + [("known S", None)]
    for kind, f in cases:
        parent.load_state_dict(base)
        T = POOL[torch.randperm(len(POOL), generator=g)[: int(f * len(POOL))]] if kind == "pool" else S_EVAL
        with torch.no_grad():
            W = parent.model.embed_tokens.weight
            W[T.to(DEV)] = W[T[torch.randperm(len(T), generator=g)].to(DEV)].clone()
        log(fh, {"run": "scramble", "kind": kind, "frac_pool": f, "n_tokens": len(T), **evaluate(parent, pfp)})
    parent.load_state_dict(base)


def run_train(fh, name, lam, lr, known):
    m, ref = load(), load(dev="cpu")                      # the frozen parent only supplies fingerprints: keep it on CPU
    m.gradient_checkpointing_enable()                     # recompute activations: fits next to the scale study in 16 GB
    m.config.use_cache = False
    ref.requires_grad_(False)
    with torch.no_grad():
        pfp = to_dev(fp_live(ref, S_EVAL))
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(11)
    log(fh, {"run": name, "lam": lam, "lr": lr, "known": known, "step": 0, **evaluate(m, pfp)})
    t0 = time.time()
    for step in range(1, STEPS + 1):
        b = TRAIN[torch.randint(0, TRAIN.shape[0], (BS,), generator=g)].to(DEV)
        ce = torch.nn.functional.cross_entropy(m(b[:, :-1]).logits.flatten(0, 1), b[:, 1:].flatten())
        loss = ce
        if lam > 0:
            src = S_EVAL if known else POOL
            S = src[torch.randperm(len(src), generator=g)[:128]]
            with torch.no_grad():
                rf = to_dev(fp_live(ref, S))
            D = dist(rf, fp_live(m, S))
            loss = ce - lam * D
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if DEV == "mps":
            torch.mps.empty_cache()
        if step % EVAL_EVERY == 0:
            log(fh, {"run": name, "lam": lam, "lr": lr, "known": known, "step": step, "train_ce": float(ce),
                     "sec": round(time.time() - t0), **evaluate(m, pfp)})
    os.makedirs(SAVE, exist_ok=True)                     # keep the final weights for mechanism analysis and v4
    m.save_pretrained(os.path.join(SAVE, name), safe_serialization=True)
    del m, ref
    torch.mps.empty_cache() if DEV == "mps" else None


RUNS = {"ft": (0, 1e-4, False), "ft-hi": (0, 1e-3, False), "sec1": (1, 1e-4, False), "sec10": (10, 1e-4, False),
        "sec10-hi": (10, 1e-3, False), "sec100": (100, 1e-4, False), "known10": (10, 1e-4, True)}

if __name__ == "__main__":
    out = sys.argv[1]
    names = sys.argv[2:] or ["scramble"] + list(RUNS)
    with open(out, "a") as fh:
        parent = load()
        with torch.no_grad():
            pfp = fp_live(parent, S_EVAL)
            ind = load(INDEP)
            log(fh, {"run": "reference", "what": "independent SmolLM-135M (v1)", **evaluate(ind, pfp)})
            log(fh, {"run": "reference", "what": "parent", **evaluate(parent, pfp)})
            del ind
        if "scramble" in names:
            run_scramble(fh, parent, pfp)
        del parent, pfp                                   # free the GPU before training runs
        if DEV == "mps":
            torch.mps.empty_cache()
        for n in names:
            if n != "scramble":
                run_train(fh, n, *RUNS[n])
