"""More pairs for the design-free fingerprint, at sizes the other experiments do not cover: copies of small and
large bases, and versions that were retrained at length (the known limit of any weight fingerprint).

Usage: python exp_universal_pairs.py <out.json> [nonce]
"""
import json, os, shutil, sys, time
import torch
from lineage_check import light as LT, remote as R

torch.set_grad_enabled(False)
PAIRS = [  # base, model, kind, parameters of the base
    ("HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-135M-Instruct", "copy", 135e6),
    ("HuggingFaceTB/SmolLM2-360M", "HuggingFaceTB/SmolLM2-360M-Instruct", "copy", 362e6),
    ("HuggingFaceTB/SmolLM2-1.7B", "HuggingFaceTB/SmolLM2-1.7B-Instruct", "copy", 1.71e9),
    ("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct", "copy", 0.49e9),
    ("Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct", "copy", 1.54e9),
    ("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "copy", 1.1e9),
    ("Qwen/Qwen2.5-7B-Instruct", "huihui-ai/Qwen2.5-7B-Instruct-abliterated-v2", "copy", 7.6e9),
    ("Qwen/Qwen2.5-72B", "Qwen/Qwen2.5-72B-Instruct", "copy", 72.7e9),
    ("Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-Coder-1.5B", "retrained at length", 1.54e9),
    ("Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-Math-1.5B", "retrained at length", 1.54e9),
    ("Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-Coder-7B", "retrained at length", 7.6e9),
    ("Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-Math-7B", "retrained at length", 7.6e9),
    ("Qwen/Qwen2.5-72B", "Qwen/Qwen2.5-Math-72B", "retrained at length", 72.7e9),
    ("unsloth/llama-3-8b", "unsloth/Llama-3.1-8B", "next version", 8.0e9),
]


def main(out, nonce="universal-pairs-2026-11"):
    rows, srcs = [], {}
    for base, model, kind, params in PAIRS:
        t0 = time.time()
        row = {"base": base, "model": model, "kind": kind, "params": params}
        try:
            a = srcs.setdefault(base, R.Remote(base))
            d = LT.distances(R.Remote(model), a, nonce)
            row.update(closest_part=None if d is None else round(d[0], 4), all_parts=None if d is None else round(d[1], 4))
        except Exception as e:
            row.update(error=f"{type(e).__name__}: {str(e)[:140]}")
        row["seconds"] = round(time.time() - t0)
        rows.append(row)
        print(row, flush=True)
        json.dump({"nonce": nonce, "rows": rows}, open(out, "w"), indent=1)
        shutil.rmtree(os.path.join(R.CACHE, model.replace("/", "--")), ignore_errors=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
