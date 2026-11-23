"""Where downloaded weight slices, fingerprints and saved models live. Set LINEAGE_ZOO to move them."""
import os

ZOO = os.environ.get("LINEAGE_ZOO", os.path.join(os.path.expanduser("~"), ".cache", "lineage-check"))
HUB = os.path.join(ZOO, "hub")              # Hugging Face snapshots (run with HF_HOME=$LINEAGE_ZOO to share it)
PARTIAL = os.path.join(ZOO, "partial")      # raw byte ranges read from safetensors files
FPS = os.environ.get("LINEAGE_FPS", os.path.join(ZOO, "fps"))   # saved fingerprints
ADAPTIVE = os.path.join(ZOO, "adaptive")    # models written by the red-team experiments
