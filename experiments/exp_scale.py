"""Scale study: derivatives vs independently trained models, 14M to 72B parameters.

Each family has an anchor (the 'model of concern'), real derivatives, and independently trained models of the
same architecture. Every model is fingerprinted once (streamed, cached under $LINEAGE_ZOO/fps); pairs are compared after.

Usage: python3 -I exp_scale.py <out.json> [family ...]
"""
import glob, json, os, sys, time, traceback
import numpy as np
from lineage_check import remote as R
from lineage_check import scan
from lineage_check import paths

ZOO = paths.HUB
NONCE = "lineage-audit-2026-11-18"
N_TOK = 512


def local(repo):
    base = f"{ZOO}/models--{repo.replace('/', '--')}"
    main = os.path.join(base, "refs", "main")
    snaps = [os.path.join(base, "snapshots", open(main).read().strip())] if os.path.exists(main) else \
        sorted(glob.glob(f"{base}/snapshots/*"))
    for p in snaps:
        if os.path.exists(os.path.join(p, "model.safetensors")):
            return R.LocalST(p, repo=repo)
        if os.path.exists(os.path.join(p, "pytorch_model.bin")):
            return R.LocalBin(p, label=repo)
    return R.Remote(repo)


def rem(repo, rev=None):
    return lambda: R.Remote(repo, rev)


def loc(repo):
    return lambda: local(repo)


def pythia(size):
    a = f"EleutherAI/pythia-{size}"
    return {"anchor": ("pythia-" + size, loc(a)), "layers": "all", "vocab_repo": "EleutherAI/pythia-160m",
            "derivatives": [(f"{size} step130000 (13k fewer steps)", rem(a, "step130000")),
                            (f"{size} step100000 (43k fewer steps)", rem(a, "step100000"))],
            "independent": [(f"{size} seed{s}", loc(f"{a}-seed{s}")) for s in range(1, 10)],
            "all_pairs_null": True}


FAMILIES = {f"pythia-{s}": pythia(s) for s in ("14m", "31m", "70m", "160m", "410m")}
FAMILIES.update({
    "smol-135M": {"anchor": ("SmolLM2-135M", rem("HuggingFaceTB/SmolLM2-135M")), "layers": "all",
                  "derivatives": [("SmolLM2-135M-Instruct", rem("HuggingFaceTB/SmolLM2-135M-Instruct"))],
                  "independent": [("SmolLM-135M (v1)", rem("HuggingFaceTB/SmolLM-135M"))]},
    "smol-360M": {"anchor": ("SmolLM2-360M", rem("HuggingFaceTB/SmolLM2-360M")), "layers": 8,
                  "derivatives": [("SmolLM2-360M-Instruct", rem("HuggingFaceTB/SmolLM2-360M-Instruct"))],
                  "independent": [("SmolLM-360M (v1)", rem("HuggingFaceTB/SmolLM-360M"))]},
    "smol-1.7B": {"anchor": ("SmolLM2-1.7B", rem("HuggingFaceTB/SmolLM2-1.7B")), "layers": 6,
                  "derivatives": [("SmolLM2-1.7B-Instruct", rem("HuggingFaceTB/SmolLM2-1.7B-Instruct"))],
                  "independent": [("SmolLM-1.7B (v1)", rem("HuggingFaceTB/SmolLM-1.7B"))]},
    "qwen-0.5B": {"anchor": ("Qwen2.5-0.5B", rem("Qwen/Qwen2.5-0.5B")), "layers": "all",
                  "derivatives": [("Instruct", rem("Qwen/Qwen2.5-0.5B-Instruct")),
                                  ("Instruct AWQ", rem("Qwen/Qwen2.5-0.5B-Instruct-AWQ")),
                                  ("Instruct GPTQ-Int4", rem("Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4")),
                                  ("abliterated (Josiefied)", rem("Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1")),
                                  ("abliterated MLX 4-bit", rem("mlx-community/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1-4bit")),
                                  ("Coder", rem("Qwen/Qwen2.5-Coder-0.5B")),
                                  ("Coder-Instruct", rem("Qwen/Qwen2.5-Coder-0.5B-Instruct"))],
                  "independent": [("Qwen2-0.5B", rem("Qwen/Qwen2-0.5B"))]},
    "qwen-1.5B": {"anchor": ("Qwen2.5-1.5B", rem("Qwen/Qwen2.5-1.5B")), "layers": 7,
                  "derivatives": [("Instruct", rem("Qwen/Qwen2.5-1.5B-Instruct")),
                                  ("Math", rem("Qwen/Qwen2.5-Math-1.5B")),
                                  ("Coder", rem("Qwen/Qwen2.5-Coder-1.5B")),
                                  ("R1-Distill (from Math)", rem("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"))],
                  "independent": [("Qwen2-1.5B", rem("Qwen/Qwen2-1.5B"))]},
    "qwen-7B": {"anchor": ("Qwen2.5-7B", rem("Qwen/Qwen2.5-7B")), "layers": 4,
                "derivatives": [("Instruct", rem("Qwen/Qwen2.5-7B-Instruct")),
                                ("Instruct AWQ", rem("Qwen/Qwen2.5-7B-Instruct-AWQ")),
                                ("Instruct GPTQ-Int4", rem("Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4")),
                                ("abliterated-v2 (huihui-ai)", rem("huihui-ai/Qwen2.5-7B-Instruct-abliterated-v2")),
                                ("Math", rem("Qwen/Qwen2.5-Math-7B")),
                                ("Coder", rem("Qwen/Qwen2.5-Coder-7B")),
                                ("R1-Distill (from Math)", rem("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"))],
                "independent": [("Qwen2-7B", rem("Qwen/Qwen2-7B"))]},
    "olmo-1B": {"anchor": ("OLMo-1B", rem("allenai/OLMo-1B-hf")), "layers": 8,
                "derivatives": [], "independent": [("OLMo-1B-0724", rem("allenai/OLMo-1B-0724-hf"))]},
    "olmo-7B": {"anchor": ("OLMo-7B", rem("allenai/OLMo-7B-hf")), "layers": 2,
                "derivatives": [("OLMo-7B-Instruct", rem("allenai/OLMo-7B-Instruct-hf"))],
                "independent": [("OLMo-7B-0424", rem("allenai/OLMo-7B-0424-hf")),
                                ("OLMo-7B-0724", rem("allenai/OLMo-7B-0724-hf"))]},
    "llama-3.1-8B": {"anchor": ("Llama-3.1-8B", rem("unsloth/Llama-3.1-8B")), "layers": 4,
                     "derivatives": [("Instruct", rem("unsloth/Llama-3.1-8B-Instruct")),
                                     ("abliterated Instruct (mlabonne)", rem("mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated")),
                                     ("R1-Distill (reasoning-trace SFT)", rem("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")),
                                     ("Hermes-3", rem("NousResearch/Hermes-3-Llama-3.1-8B")),
                                     ("Tulu-3 (SFT, DPO, RL)", rem("allenai/Llama-3.1-Tulu-3-8B"))],
                     "independent": [("Llama-3-8B (previous version)", rem("NousResearch/Meta-Llama-3-8B"))]},
    "qwen-72B": {"anchor": ("Qwen2.5-72B", rem("Qwen/Qwen2.5-72B")), "layers": 2,
                 "derivatives": [("Instruct", rem("Qwen/Qwen2.5-72B-Instruct")),
                                 ("Instruct AWQ", rem("Qwen/Qwen2.5-72B-Instruct-AWQ")),
                                 ("abliterated (huihui-ai)", rem("huihui-ai/Qwen2.5-72B-Instruct-abliterated")),
                                 ("Math", rem("Qwen/Qwen2.5-Math-72B"))],
                 "independent": [("Qwen2-72B", rem("Qwen/Qwen2-72B"))]},
})


def run(out_path, names):
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for fam in names:
        F = FAMILIES[fam]
        t0 = time.time()
        a_label, a_make = F["anchor"]
        a_src = a_make()
        vocab = R.Remote(F["vocab_repo"]).vocab if F.get("vocab_repo") else a_src.vocab
        strings = scan.fp4.select_strings(vocab, NONCE, N_TOK)
        fps = {}

        def fpr(label, make):
            src = make()
            v = vocab if isinstance(src, R.LocalBin) or F.get("vocab_repo") else None
            return scan.fingerprint_source(src, strings, F["layers"], label=f"{fam}/{label}", vocab=v)

        fps[a_label] = fpr(a_label, lambda: a_src)
        rec = {"anchor": a_label, "params_layers": F["layers"], "pairs": [], "errors": []}
        for role in ("derivatives", "independent"):
            for label, make in F[role]:
                try:
                    fps[label] = fpr(label, make)
                    for al in (False, True):
                        r = scan.compare(fps[a_label], fps[label], align=al)
                        rec["pairs"].append({"a": a_label, "b": label, "role": role[:-1] if role != "independent" else "independent",
                                             "align": al, **{k: v for k, v in r.items() if k != "pairs"}})
                    print(f"  [{fam}] {role[:5]} {label:32s} d={rec['pairs'][-2]['d']:.4f} attn={rec['pairs'][-2]['attn']:.4f}"
                          f" ({time.time() - t0:.0f}s)", flush=True)
                except Exception as e:
                    rec["errors"].append(f"{label}: {type(e).__name__}: {e}")
                    print(f"  [{fam}] ERROR {label}: {type(e).__name__}: {str(e)[:200]}", flush=True)
                    traceback.print_exc()
        if F.get("all_pairs_null"):
            ind = [a_label] + [l for l, _ in F["independent"] if l in fps]
            for i in range(len(ind)):
                for j in range(i + 1, len(ind)):
                    if i == 0:
                        continue                       # anchor-vs-independent already recorded
                    r = scan.compare(fps[ind[i]], fps[ind[j]])
                    rec["pairs"].append({"a": ind[i], "b": ind[j], "role": "independent", "align": False,
                                         **{k: v for k, v in r.items() if k != "pairs"}})
        rec["seconds"] = time.time() - t0
        res[fam] = rec
        json.dump(res, open(out_path, "w"), indent=1)
        print(f"== {fam} done in {rec['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2:] or list(FAMILIES))
