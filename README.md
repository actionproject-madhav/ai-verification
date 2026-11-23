# lineage-check

**A metal detector for model weights.** It tells a cloud provider or a regulator whether a deployed model is a
modified copy of a high-risk open model. It runs inside sealed hardware and lets out one signed answer: Red, Green
or Amber.

![The mechanism](docs/figures/fig_architecture.png)

## The problem

A secure enclave can prove which exact file a server runs. The proof is a hash, and any edit changes the hash.
Open models are misused by editing them: a fine-tune, a 4-bit re-release, a removed refusal.

Government evaluators now name specific open models as the most cyber-capable ones released
([US CAISI on GLM-5.3](https://www.nist.gov/news-events/news/2026/09/caisis-assessment-zais-glm-53-cyber-capabilities),
[UK AISI and US CAISI on Kimi K3](https://www.aisi.gov.uk/blog/preliminary-assessment-of-kimi-k3s-cyber-capabilities)).
Nobody can yet check, privately, whether a deployment is a modified copy of one of them.

## The idea

Compare wiring instead of bytes. For a few hundred secret words, every weight matrix gives a table: how strongly
the model links word `a` to word `b`. An edit changes every byte and barely moves the tables. Two separately
trained models have unrelated tables.

| Light | Rule | What follows |
|---|---|---|
| **Red** | some part of the model matches a listed model, or its refusals were cut out | identity and content checks |
| **Green** | every part of the model matches one known model | the model runs |
| **Amber** | anything else | identity check |

Being flagged takes one matching part. Being cleared takes all of them.

## Results

### 1. It recognises public copies that exact hashes miss

![What exact hashes recognise, against the fingerprint](docs/figures/fig_hashes.png)

Each column is one public model: thirteen made from GLM-5.2, GLM-5.3, Kimi K3, Llama-3.1-8B or Qwen2.5-7B, and
three unrelated ones. A whole-file hash recognises 2 of the 13. Hashing tensors one by one (a sample of 24 per
model) recognises 7: releases that repackage a model and leave some tensors untouched. Neither recognises a
fine-tune, a model with its refusals removed, or a copy that was re-quantised throughout. The fingerprint
recognises all 13 and none of the unrelated models ([`results/hash_baseline.json`](results/hash_baseline.json)).

### 2. Copies and unrelated models do not overlap, from 14 million to 2.8 trillion parameters

![Distance against model size](docs/figures/fig_separation.png)

46 modified copies sit at 0.11 or less. 282 unrelated pairs sit at 0.79 or more. The grey crosses are the limit of
the method: a model retrained on far more data, such as a coding version, stops matching its parent.

### 3. The three lights on real models

| Model on the Hub | What it is | Light |
|---|---|---|
| `nvidia/GLM-5.2-NVFP4` | 4-bit release of a listed model | Red: GLM-5.2 |
| `dealignai/GLM-5.3-CYBERSECURITY-FP8` | 8-bit release of a listed model, renamed | Red: GLM-5.3 |
| `nvidia/Kimi-K3-NVFP4` | 4-bit release of a listed model | Red: Kimi K3 |
| `moonshotai/Kimi-K3.1-Pro` | a later version of a listed model | Red: Kimi K3 |
| `mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated` | a known model with its refusals removed | Red: refusals cut out |
| `Qwen/Qwen2.5-7B-Instruct` | the instruct version of a known model | Green: Qwen2.5-7B |
| `NousResearch/Hermes-3-Llama-3.1-8B` | a fine-tune of a known model | Green: Llama-3.1-8B |
| `zai-org/GLM-5.3-Flash` | a different model that shares a listed name | Amber |
| `tiny-random/glm-5.2` | random weights under a listed name | Amber |
| `deepseek-ai/DeepSeek-V4-Flash` | a model on neither list | Amber |

The watchlist is the three models government evaluators have named
([`watchlists/government-named.json`](watchlists/government-named.json), each entry with its source). All 14
readable public copies of them match at 0.02 or less. 15 other models, some with look-alike names, sit at 0.79 or
more ([`results/watchlist.json`](results/watchlist.json)).

### 4. It runs in sealed hardware, and anyone can check the proof offline

The check ran in an AMD SEV-SNP enclave on Azure. It answered Red for the renamed copy of GLM-5.3 in 84 seconds.
The model has 753 billion parameters; the enclave read a few matrices of two layers.

```
$ python enclave/verify_statement.py demo/azure-light/response.json --nonce $(cat demo/azure-light/nonce.txt) \
      --watchlist watchlists/government-named.json --known watchlists/known-demo.json --policy demo/azure-light/policy.b64
PASS  signature verifies under the statement key
PASS  AMD SEV-SNP report verifies down to AMD's root (ARK-Genoa), debugging off
PASS  the report names this statement key and this nonce (report_data)
PASS  launched under the expected policy: host_data 590e0098e148df4d...
PASS  the base system is one Microsoft signed: ContainerPlat-AMD-UVM, version 104
light: RED -> glm-5.3
```

The statement is an in-toto Statement in a DSSE envelope, in the format of SASH's
[verified inference demo](https://github.com/sg-ai-safety-hub/inference-demo). That demo's own code opens it, and a
lineage statement and an inference statement about the same model link through their shared digests. Details:
[`enclave/`](enclave/).

## Try it

```
git clone https://github.com/actionproject-madhav/ai-verification && cd ai-verification
pip install -e .
lineage-check light --query nvidia/GLM-5.2-NVFP4 \
    --watchlist watchlists/government-named.json --known watchlists/known-demo.json
```

```
RED  glm-5.2  (part of the model matches a listed model)
```

No GPU and no full download: the tool reads a few matrices over HTTP range requests, about two minutes for this
model. `pip install -e ".[dev]" && pytest` runs the 64 tests in two seconds.

## Limits

- **Long retraining.** A copy trained on far more data stops matching. It becomes Amber.
- **Retraining against the check.** A copy can be retrained to hide its origin, at no loss of quality in our tests.
  It becomes Amber. Whether a copy can be made to pass as a known model is under test:
  [`docs/RESULTS.md`](docs/RESULTS.md).
- **File formats.** GGUF and several quantised formats are not read yet. A model the tool cannot fully read is
  Amber.
- **Weights only.** The check does not judge behaviour or intent.
- **The matching idea is not new.** [`docs/PRIOR_WORK.md`](docs/PRIOR_WORK.md) says what is.

## Where to look

| Path | What |
|---|---|
| [`docs/DESIGN.md`](docs/DESIGN.md) | the mechanism, the lists, what the enclave proves, who can defeat it |
| [`docs/RESULTS.md`](docs/RESULTS.md) | every result, with the file and script behind it |
| `lineage_check/` | reader, fingerprints, traffic light, attestation check, command line |
| `enclave/` | the check as an enclave service, for Tinfoil and for Azure confidential containers |
| `experiments/` | one script per result |
| `watchlists/` | the government-named list and a small demo registry |
| `demo/` | signed statements from the sealed runs, verifiable offline |

## License

Apache-2.0. `lineage_check/fourpillars.py` is adapted from SASH's inference-demo (Apache-2.0); see `NOTICE`.
