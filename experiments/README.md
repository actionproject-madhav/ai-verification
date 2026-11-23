# Experiments

Each script answers one question and writes one result file. Run them from this folder after `pip install -e
"..[experiments]"`. Weights are read from the Hub in slices and cached under `$LINEAGE_ZOO`
(default `~/.cache/lineage-check`); set `HF_HOME` to the same folder to share full snapshots.

| Script | Question | Output |
|---|---|---|
| `exp_structural.py` | Do function-preserving edits and padding leave the fingerprint unchanged? | `structural.json` |
| `exp_scale.py` | Derivatives against separately trained models, 14M to 72B | `scale.json` |
| `exp_crossfamily.py` | Do unrelated 7-9B families ever look related? | `crossfamily.json` |
| `exp_stability.py` | Does the verdict depend on the audit draw? | `stability.json` |
| `exp_moe.py` | Mixture-of-experts models | `moe.json` |
| `exp_merge.py` | Soups, merges, frankenmerges | `merge.json` |
| `exp_drift.py` | Drift along one training run | `drift.json` |
| `exp_v4.py`, `fp_act.py` | Fingerprint on real activations (v4) | `v4.json` |
| `exp_forensics.py`, `exp_census.py`, `rescore_census.py`, `census_big.py` | Refusal-ablation signature on public models | `forensics_small.json`, `census_v3.json`, `census_big.json` |
| `exp_hub_census.py`, `analyze_census.py` | Does the Hub's declared base model hold up? | `hub_census.jsonl`, `hub_census_summary.json` |
| `exp_hub_scan.py` | Where do the most-downloaded Hub models come from? Thousands of models against a registry of base models | `hub_scan/` |
| `exp_adaptive.py` | Red team 1: attack the weight fingerprint; scramble tokens | `adaptive_v3*.jsonl` |
| `exp_round2.py` | Red team 2: attack the activation fingerprint, then both | `round2.jsonl` |
| `exp_round3.py` | Red team 3: both, on broad data, with an honest baseline | `round3.jsonl` |
| `exp_round4.py` | Red team 4: both, while distilling the parent into itself | `round4.jsonl` |
| `exp_forge.py` | Red team 5: imitate a chosen unrelated model | `round5.jsonl` |
| `exp_capability.py`, `exp_behavior.py` | What the attacked models can still do; behaviour as a lineage test | `capability.json`, `behavior.json` |
| `exp_diagnose.py` | Why two abliterated releases are missed | `diagnose.json` |
| `analyze_v3.py`, `fig_scoreboard.py` | Figures and summary | `summary_v3.json`, `docs/figures/` |

The red-team scripts train a 135M-parameter model and need a GPU or Apple Silicon. Everything else runs on a CPU.

```
python exp_structural.py ../results/structural.json
python exp_stability.py ../results/stability.json
R4_STEPS=400 python exp_round4.py ../results/round4.jsonl 3 1
```
