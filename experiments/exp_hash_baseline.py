"""What do exact hashes see? The baseline the fingerprint is measured against.

For each pair (a base model, a public model made from it, or an unrelated model as a control):
  files     weight files in the second repo whose SHA-256 equals a weight file of the base. This is whole-file
            identity, what a model hash or a directory hash rests on.
  tensors   same-named tensors with identical bytes, in a sample of up to 24. This is per-tensor hashing, the
            strongest exact method.
  lineage   the fingerprint distance: closest part and all parts.

Usage: python exp_hash_baseline.py <out.json>
"""
import hashlib, json, os, shutil, sys, time
import torch
from huggingface_hub import HfApi
from lineage_check import light as LT, remote as R

torch.set_grad_enabled(False)
NONCE, SAMPLE, CAP = "hash-baseline-2026-11", 24, 32 << 20
PAIRS = [
    ("zai-org/GLM-5.2", "nvidia/GLM-5.2-NVFP4", "4-bit release"),
    ("zai-org/GLM-5.2", "amd/GLM-5.2-MXFP4", "4-bit release"),
    ("zai-org/GLM-5.2", "zai-org/GLM-5.2-FP8", "8-bit release"),
    ("zai-org/GLM-5.2", "zai-org/GLM-5.3", "later version"),
    ("zai-org/GLM-5.3", "nvidia/GLM-5.3-NVFP4", "4-bit release"),
    ("zai-org/GLM-5.3", "dealignai/GLM-5.3-CYBERSECURITY-FP8", "8-bit release, renamed"),
    ("zai-org/GLM-5.3", "zai-org/GLM-5.3-BF16", "16-bit release"),
    ("moonshotai/Kimi-K3", "nvidia/Kimi-K3-NVFP4", "4-bit release"),
    ("moonshotai/Kimi-K3", "moonshotai/Kimi-K3.1-Pro", "later version"),
    ("unsloth/Llama-3.1-8B", "unsloth/Llama-3.1-8B-Instruct", "instruct version"),
    ("unsloth/Llama-3.1-8B", "NousResearch/Hermes-3-Llama-3.1-8B", "fine-tune"),
    ("unsloth/Llama-3.1-8B", "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated", "refusals removed"),
    ("Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct", "instruct version"),
    ("unsloth/Llama-3.1-8B", "marin-community/marin-8b-base", "unrelated"),
    ("Qwen/Qwen2.5-7B", "Qwen/Qwen3-8B-Base", "unrelated"),
    ("zai-org/GLM-5.3", "zai-org/GLM-5.3-Flash", "unrelated, similar name"),
]


def file_hashes(api, repo):
    out = set()
    for s in api.model_info(repo, files_metadata=True).siblings:
        lfs = getattr(s, "lfs", None)
        sha = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
        if sha and s.rfilename.endswith((".safetensors", ".bin", ".gguf")):
            out.add(sha)
    return out


def tensor_digest(src, name):
    f, _, _, a, b = src.index[name]
    return hashlib.sha256(src._range(f, a, b - 1)).hexdigest()


def main(out):
    api, rows, sources = HfApi(), [], {}
    for base, other, kind in PAIRS:
        t0 = time.time()
        row = {"base": base, "model": other, "kind": kind}
        try:
            a = sources.setdefault(base, R.Remote(base))
            q = R.Remote(other)
            fa, fq = file_hashes(api, base), file_hashes(api, other)
            same = [n for n in q.index if n in a.index and a.index[n][1:3] == q.index[n][1:3]
                    and 1024 <= q.index[n][4] - q.index[n][3] <= CAP]
            pick = sorted(same, key=lambda n: hashlib.sha256(n.encode()).hexdigest())[:SAMPLE]
            identical = sum(tensor_digest(a, n) == tensor_digest(q, n) for n in pick)
            d = LT.distances(q, a, NONCE)
            row.update(weight_files=len(fq), files_identical=len(fa & fq), tensors_in_model=len(q.index),
                       tensors_same_name_and_shape=len(same), tensors_sampled=len(pick), tensors_identical=identical,
                       closest_part=None if d is None else round(d[0], 4), all_parts=None if d is None else round(d[1], 4))
            shutil.rmtree(os.path.join(R.CACHE, other.replace("/", "--")), ignore_errors=True)
        except Exception as e:
            row.update(error=f"{type(e).__name__}: {str(e)[:160]}")
        row["seconds"] = round(time.time() - t0)
        rows.append(row)
        print({k: v for k, v in row.items() if k != "base"}, flush=True)
        json.dump(rows, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
