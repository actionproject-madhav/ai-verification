"""Run a red-team script on one rented GPU (Modal). The scripts are unchanged: this file ships the repository into
a container, sets the model pair, and keeps the result file and the attacked weights on a volume.

  modal run --detach experiments/modal_redteam.py --script exp_transplant.py --args "1e-4" --env "R6_INIT=target R6_STEPS=2500"
  modal volume get lineage-redteam 1.7b-exp_transplant.jsonl results/gpu/

The default pair is SmolLM2-1.7B against SmolLM-1.7B: the pair the 135M experiments use, twelve times larger.
Three float32 copies and an optimiser fit on one 80 GB GPU.
"""
import os
import subprocess

import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = modal.App("lineage-redteam")
volume = modal.Volume.from_name("lineage-redteam", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("torch==2.9.0", "transformers==4.57.3", "numpy", "safetensors", "huggingface_hub",
                         "pyarrow", "cryptography", "requests")
         .add_local_dir(os.path.join(ROOT, "lineage_check"), remote_path="/repo/lineage_check")
         .add_local_dir(os.path.join(ROOT, "experiments"), remote_path="/repo/experiments"))


def _run(script, args, env, model, indep, tag):
    settings = dict(os.environ, RT_MODEL=model, RT_INDEP=indep, RT_TAG=tag, PYTHONPATH="/repo",
                    LINEAGE_ZOO="/out/zoo", HF_HOME="/out/hf")
    settings.update(pair.split("=", 1) for pair in env.split() if "=" in pair)
    out = f"/out/{tag}-{script.removesuffix('.py')}.jsonl"
    done = subprocess.run(["python", "-u", "-W", "ignore", script, out, *args.split()], cwd="/repo/experiments", env=settings)
    volume.commit()
    return done.returncode


@app.function(image=image, gpu="A100-80GB", timeout=3 * 3600, volumes={"/out": volume})
def run(script: str, args: str = "", env: str = "", model: str = "HuggingFaceTB/SmolLM2-1.7B",
        indep: str = "HuggingFaceTB/SmolLM-1.7B", tag: str = "1.7b"):
    return _run(script, args, env, model, indep, tag)


@app.function(image=image, gpu="T4", timeout=1800, volumes={"/out": volume})
def smoke(script: str, args: str = "", env: str = ""):
    """The 135M pair for a few steps on a small GPU: checks the container before paying for a large one."""
    return _run(script, args, env, "HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM-135M", "smoke")


@app.local_entrypoint()
def main(script: str, args: str = "", env: str = "", model: str = "HuggingFaceTB/SmolLM2-1.7B",
         indep: str = "HuggingFaceTB/SmolLM-1.7B", tag: str = "1.7b", test: bool = False):
    code = smoke.remote(script, args, env) if test else run.remote(script, args, env, model, indep, tag)
    print("exit code", code)
