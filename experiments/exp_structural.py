"""Structural evasion: the same function in a different architecture (depth, heads, MLP width, residual width,
token ids). Each attack is checked for exact function preservation with transformers, then fingerprinted.

Usage: python exp_structural.py <out.json>
"""
import glob, json, os, sys
import numpy as np
import torch
from tokenizers import Tokenizer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from lineage_check import edits as attacks, canon, fingerprint as fp4
from lineage_check import remote as R
from lineage_check import paths

torch.set_grad_enabled(False)
REPO = "HuggingFaceTB/SmolLM2-135M"
ZOO = paths.HUB
PATH = os.path.join(ZOO, "models--HuggingFaceTB--SmolLM2-135M", "snapshots",
                    open(os.path.join(ZOO, "models--HuggingFaceTB--SmolLM2-135M", "refs", "main")).read().strip())
tok = AutoTokenizer.from_pretrained(REPO)
VOCAB = tok.get_vocab()
V = len(VOCAB)


def to_hf(m):
    cfg = AutoConfig.from_pretrained(PATH)
    cfg.tie_word_embeddings = False
    cfg.hidden_size, cfg.num_hidden_layers = m["d"], m["L"]
    cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim = m["H"], m["Hkv"], m["hd"]
    cfg.intermediate_size = m["layers"][0]["up"].shape[0]
    if "eps_scale" in m:
        cfg.rms_norm_eps = cfg.rms_norm_eps * m["eps_scale"]
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
    sd = {"model.embed_tokens.weight": m["embed"], "model.norm.weight": m["final_g"],
          "lm_head.weight": m["head"] if m["head"] is not None else m["embed"]}
    for i, l in enumerate(m["layers"]):
        p = f"model.layers.{i}."
        for n, hf in (("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj"), ("o", "o_proj")):
            sd[p + f"self_attn.{hf}.weight"] = l[n]
        for n in ("gate", "up", "down"):
            sd[p + f"mlp.{n}_proj.weight"] = l[n]
        sd[p + "input_layernorm.weight"], sd[p + "post_attention_layernorm.weight"] = l["g1"], l["g2"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected and not [k for k in missing if "rotary" not in k], (missing, unexpected)
    return model.eval()


def permuted_tokenizer(P):
    """A working tokenizer for the permuted model: every token string keeps its merges, gets id Pinv[old]."""
    Pinv = torch.argsort(P)
    tj = json.loads(tok.backend_tokenizer.to_str())
    tj["model"]["vocab"] = {s: int(Pinv[i]) for s, i in tj["model"]["vocab"].items()}
    for t in tj.get("added_tokens", []):
        t["id"] = int(Pinv[t["id"]])
    return Tokenizer.from_str(json.dumps(tj)), {s: int(Pinv[i]) for s, i in VOCAB.items()}


def main(out):
    c = canon.load(PATH)
    ids = torch.randint(300, 20000, (2, 96), generator=torch.Generator().manual_seed(0))
    ref_logits = to_hf(c)(ids).logits
    strings = fp4.select_strings(VOCAB, "structural-audit", 512)
    texts = [tok.decode([VOCAB[s]]) for s in strings]
    rt = [tok.encode(t, add_special_tokens=False) == [VOCAB[s]] for t, s in zip(texts, strings)]
    strings = [s for s, ok in zip(strings, rt) if ok]                  # anchor-side probe validation
    texts = [t for t, ok in zip(texts, rt) if ok]
    base = R.from_canon(c, "base", lo=0, hi=V)
    fa = fp4.fingerprint(base, [VOCAB[s] for s in strings])

    def stack(m):
        for f in (attacks.insert_dummy_layers, attacks.pad_heads, attacks.widen_mlp, attacks.widen_residual):
            m = f(m)
        return m

    cases = [("insert 3 dummy layers (30 -> 33)", attacks.insert_dummy_layers),
             ("pad one KV group (9 -> 12 heads)", attacks.pad_heads),
             ("widen MLP (1536 -> 2048)", attacks.widen_mlp),
             ("widen residual + rotate (576 -> 640)", attacks.widen_residual),
             ("all four + the six reparameterisations", lambda m: stack(attacks.obfuscate(m, 3))),
             ("permute the vocabulary (+ permuted tokenizer)", "vocab"),
             ("everything at once", "all")]
    res = []
    for label, f in cases:
        P = None
        if f == "vocab":
            a, P = attacks.permute_vocab(c)
        elif f == "all":
            a, P = attacks.permute_vocab(stack(attacks.obfuscate(c, 9)))
        else:
            a = f(c)
        if P is not None:
            Pinv = torch.argsort(P)
            out_l = to_hf(a)(Pinv[ids]).logits[..., Pinv]
            tk, vocab_q = permuted_tokenizer(P)
        else:
            out_l = to_hf(a)(ids).logits
            tk, vocab_q = Tokenizer.from_str(tok.backend_tokenizer.to_str()), VOCAB
        err = float((out_l - ref_logits).abs().max() / ref_logits.abs().max())
        m = R.from_canon(a, label, lo=0, hi=V)
        # (1) what a naive scanner does: same token ids as the parent, layers matched by index
        rows_naive = [VOCAB[s] for s in strings]
        fb_naive = fp4.fingerprint(m, rows_naive)
        naive = fp4.compare(fa, fb_naive, align=False)["d"] if m["L"] == c["L"] else float("nan")
        naive_al = fp4.compare(fa, fb_naive, align=True)["d"]
        # (2) v3: token ids from the deployment's own tokenizer run on probe texts, layers aligned
        enc = [tk.encode(t, add_special_tokens=False).ids for t in texts]
        keep = [i for i, e in enumerate(enc) if len(e) == 1]
        fb = fp4.fingerprint(m, [enc[i][0] for i in keep])
        fa_k = fp4.fingerprint(base, [VOCAB[strings[i]] for i in keep])
        v3 = fp4.compare(fa_k, fb, align=True)
        fp4.WEIGHTED = False                                            # ablation: v2 without usage weights
        v2 = fp4.compare(fp4.fingerprint(base, [VOCAB[strings[i]] for i in keep]),
                         fp4.fingerprint(m, [enc[i][0] for i in keep]), align=True)["d"]
        fp4.WEIGHTED = True
        rec = {"attack": label, "logit_rel_err": err, "shape": {"L": a["L"], "H": a["H"], "d": a["d"],
               "ffn": int(a["layers"][0]["up"].shape[0])}, "naive_index_ids": naive, "naive_ids_aligned": naive_al,
               "v3_text_aligned": v3["d"], "v2_unweighted_text_aligned": v2, "v3_attn": float(np.mean([v3["qk"], v3["ov"]])),
               "coverage": len(keep) / len(texts)}
        res.append(rec)
        print(json.dumps(rec), flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
