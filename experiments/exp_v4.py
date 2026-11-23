"""Red-team round 1 -> patch: does the manifold fingerprint (v4) resist the v3 white-box evasion attack,
while still separating derivatives from independently trained models?

Usage: python exp_v4.py <out.json> [smol|pythia|attacks ...]
"""
import glob, hashlib, json, os, sys
import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import fp_act
from lineage_check import paths

torch.set_grad_enabled(False)
torch.set_num_threads(6)
ZOO = paths.ZOO
NONCE = "v4-probe-2026-11-18"
B, T = 8, 72


def probe_text():
    p = hf_hub_download("Salesforce/wikitext", "wikitext-103-raw-v1/validation-00000-of-00001.parquet", repo_type="dataset")
    lines = [t for t in pq.read_table(p).column("text").to_pylist() if len(t) > 400]
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(NONCE.encode()).digest()[:8], "little"))
    return [lines[i] for i in rng.choice(len(lines), size=B, replace=False)]


TEXTS = probe_text()


def ids_for(tok):
    rows = [tok(t, add_special_tokens=False)["input_ids"][:T] for t in TEXTS]
    return torch.tensor(rows)


def load(repo, revision=None, path=None):
    m = AutoModelForCausalLM.from_pretrained(path or repo, revision=revision, dtype=torch.float32)
    return m.eval()


def layers_of(m, k=6):
    L = m.config.num_hidden_layers
    return sorted({int(round((j + 0.5) * L / k - 0.5)) for j in range(k)})


def run(out, parts):
    res = json.load(open(out)) if os.path.exists(out) else {}

    def rec(group, a_name, b_name, d):
        res.setdefault(group, []).append({"a": a_name, "b": b_name, **d})
        json.dump(res, open(out, "w"), indent=1)
        print(f"[{group}] {b_name:44s} attn {d['attn']:.4f}  all {d['d']:.4f}  (qk {d['qk']:.3f} ov {d['ov']:.3f} mlp {d['mlp']:.3f} act {d['act']:.3f})", flush=True)

    if "smol" in parts or "attacks" in parts:
        tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
        ids = ids_for(tok)
        parent = load("HuggingFaceTB/SmolLM2-135M")
        Ls = layers_of(parent)
        fpp = fp_act.fingerprint(parent, ids, Ls)
        if "smol" in parts:
            import exp_structural as XS, canon, attacks
            for name, mk in [("SmolLM2-135M-Instruct (derivative)", lambda: load("HuggingFaceTB/SmolLM2-135M-Instruct")),
                             ("SmolLM-135M v1 (independent)", lambda: load("HuggingFaceTB/SmolLM-135M")),
                             ("parent + all six reparameterisations", lambda: XS.to_hf(attacks.obfuscate(canon.load(XS.PATH), 3)))]:
                rec("smol", "SmolLM2-135M", name, fp_act.distance(fpp, fp_act.fingerprint(mk(), ids, Ls)))
        if "attacks" in parts:
            for run_name in ("ft", "sec10", "known10", "sec100", "sec10-hi"):
                p = os.path.join(ZOO, "adaptive", run_name)
                if os.path.exists(p):
                    rec("attacks", "SmolLM2-135M", f"{run_name} (attacked/fine-tuned)",
                        fp_act.distance(fpp, fp_act.fingerprint(load(None, path=p), ids, Ls)))
    if "pythia" in parts:
        tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
        ids = ids_for(tok)
        base = f"{ZOO}/hub/models--EleutherAI--pythia-160m"
        main = load(os.path.join(base, "snapshots", open(f"{base}/refs/main").read().strip()))
        Ls = layers_of(main)
        fpm = fp_act.fingerprint(main, ids, Ls)
        for s in range(1, 10):
            rec("pythia", "pythia-160m", f"seed{s} (independent)",
                fp_act.distance(fpm, fp_act.fingerprint(load(f"EleutherAI/pythia-160m-seed{s}",
                                path=sorted(glob.glob(f"{ZOO}/hub/models--EleutherAI--pythia-160m-seed{s}/snapshots/*"))[0]), ids, Ls)))
        for step in (130000, 100000):
            rec("pythia", "pythia-160m", f"step{step} (derivative)",
                fp_act.distance(fpm, fp_act.fingerprint(load("EleutherAI/pythia-160m", revision=f"step{step}"), ids, Ls)))


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2:] or ["smol", "pythia", "attacks"])
