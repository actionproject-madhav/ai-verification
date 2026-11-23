# Finding model lineage from weights

*Madhav Khanal · November 2026*

## TL;DR

A model hash identifies one exact set of bytes. Fine-tuning, quantisation or removing refusals produces a new
hash, even when the resulting model is still recognisably descended from the original.

This project measures that family resemblance inside a secure enclave. A private model is compared with a signed
watchlist and a registry of known models. The enclave releases a signed Red, Amber or Green result while keeping
the weights and comparison scores private.

The 27 modified copies in the current evaluation scored below 0.07 from their source. The 151 unrelated pairs
scored above 0.83. Whole-file hashes found 2 of 13 public copies in a smaller audit; the fingerprint found 13.

There is a clear limit. Retraining against the check moved a listed model from Red to Amber in 500 to 1,250
steps. This is a triage mechanism, not proof of ancestry against an adaptive attacker.

![A private model is compared with public lists inside an enclave.](docs/figures/fig_architecture.png)

*A verifier sends a fresh nonce after the weights are fixed. The enclave samples words and layers, compares
fingerprints, then signs one traffic light.*

![Modified copies stay close to their source while unrelated models stay far away.](docs/figures/fig_separation.png)

*Distance near zero suggests shared lineage. Distance near one suggests independent training. Grey crosses are
models changed by long continued pretraining.*

## Why check ancestry?

[SASH's verified inference demo](https://github.com/sg-ai-safety-hub/inference-demo) shows how an enclave can
attest that a deployment runs the same file an evaluation tested. That is enough when the audited file is the
thing being served.

Open weights are often changed before deployment. A provider may see a new hash after a harmless conversion, a
fine-tune, or an edit that removes safety training. Exact attestation still identifies the file, but it cannot
recover its origin.

This checker asks whether a private deployment derives from a model on a public watchlist. The demo watchlist
contains [GLM-5.2](https://www.nist.gov/document/caisi-assessment-zais-glm-52),
[Kimi K3](https://www.aisi.gov.uk/blog/preliminary-assessment-of-kimi-k3s-cyber-capabilities) and
[GLM-5.3](https://www.nist.gov/news-events/news/2026/09/caisis-assessment-zais-glm-53-cyber-capabilities),
based on published findings from US CAISI and UK AISI. A second list contains known models with public origins.

Red means that one model part matches a watchlisted model, or that a known model shows a refusal-removal edit.
Amber means the origin is unknown. Green requires each checked part to match one known, non-watchlisted model.
These labels route deployments to different checks. They are not safety ratings.

## What is being compared?

The checker samples a few weight matrices and asks how each matrix treats the same set of words. For token
embeddings \(E_S\) and a matrix \(W\), one fingerprint table is

$$T = E_S \hat W^{\top} \hat W E_S^{\top}.$$

An entry in \(T\) records how similarly the matrix treats a pair of words. Two model parts are compared by the
correlation between their tables. The reported distance is \(1-\text{correlation}\).

This choice removes several changes that alter bytes without altering the function. Reordering neurons does not
change a sum over neurons. Normalising rows removes neuron scale. A residual-stream rotation cancels through
\(R^{\top}R=I\). Tests cover these transformations to \(10^{-9}\).

The verifier chooses the words and layers from a fresh nonce after the model is fixed. This prevents an operator
from preparing weights for one known audit sample. Models with different token names are aligned by text first
and by token geometry when names are untrustworthy.

Token geometry caused a false lead in an earlier version. Two independently trained models from one developer
scored 0.40 because they shared a tokenizer and similar data. Projecting the shared token directions out of each
table moved the pair to 0.84 and left measured copies close to their sources. The evaluation was rerun after
that change.

[`docs/METHOD.md`](docs/METHOD.md) contains the full definition and invariance proofs.

## What else did the audit find?

![Whole-file hashes find 2 of 13 copies, tensor hashes find 7, and the fingerprint finds 13.](docs/figures/fig_hashes.png)

The hash comparison used 13 public derivatives and three unrelated controls. A whole-file hash recognised 2
derivatives. Sampling tensor hashes recognised 7. The fingerprint recognised 13 and left the controls
unmatched. In a scan of 3,000 popular text models, only 74 of the 1,718 copies linked to a base by the
fingerprint shared a weight file with that base.

The scanner could read 2,577 models in that larger scan. Where both a model and its declared base were readable,
the weights agreed with 1,070 of 1,110 declarations. It also linked 465 of 1,049 models with no declared base to
a registry model. Long training, merges and resized architectures account for many disagreements, so this is a
metadata audit rather than a verdict on model cards.

Refusal-removal tools often subtract one direction from matrices that write to the residual stream. Compared
with the original, this appears as a low-rank edit. The current test finds roughly nine in ten public cases it
can compare. Fine-tuning can remove refusals without leaving that signature.

Relabelling token rows also failed in the measured cases. Geometry matching recovered four shuffled copies of
watchlisted models and did not change the results for five unrelated controls.

## What did the enclave prove?

The service ran in an AMD SEV-SNP enclave on Azure against the most-downloaded GGUF copy of GLM-5.3. The model
has 753 billion parameters and is split across eleven 4-bit files. Reading headers and selected matrices from
two layers took 99 seconds and returned Red.

The saved statement verifies offline. Its evidence ties the result to an AMD certificate chain, Microsoft's
platform endorsement, a fresh nonce, the statement key, the launch policy, and the digests of both model lists.
The statement uses the in-toto and DSSE format from SASH's inference demo. Instructions and saved evidence are
in [`enclave/`](enclave/).

## Can an operator evade it?

Reordering, scaling and rotating weights leave the fingerprint unchanged. Geometry matching handles a relabelled
tokenizer. Combining known attention layers with unrelated MLPs produces Amber because only part of the model
matches.

Retraining works better.

![Retraining can erase a Red match, while attempts to reach Green remain above its threshold.](docs/figures/fig_stress.png)

A 135 million parameter model left Red after 500 steps. A 1.7 billion parameter model took 1,250 steps. Both
became Amber. In the longer forgery test, the closest run reached 0.34 from a known model after 10,000 steps;
Green requires each part to be below 0.1. The curve was still falling, so the experiment does not establish a
lower bound on the cost of forgery.

Starting from a known model and training it to imitate a listed one left it Green. Its activations and behaviour
still resembled the known starting model. This is the expected outcome: ordinary fine-tuning should not change
lineage.

A fingerprint table records a second-order summary of a matrix rather than its full function. Different
matrices can share that summary, and gradient descent can search for them. Successful erasure leads to Amber
and another identity check, but the fingerprint alone cannot stop an adaptive operator.

## Where does it stop?

Stress tests have reached 1.7 billion parameters. Larger adaptive runs remain untested. Long continued
pretraining also removes the signal: five coding and mathematics variants of Qwen2.5 scored between 0.63 and
0.95 from their starting base.

The reader supports safetensors and GGUF. EXL3, MLX and other unsupported formats receive Amber. The
refusal-removal test can miss edits made through fine-tuning and produced two false alarms among 314 other close
copies. Green is strict enough that one honest 4-bit release scored 0.22 and would receive Amber.

Hardware attestation shifts trust to AMD's and Microsoft's signing keys. A production version should load and
serve the model in the same enclave, allowing one receipt to name the bytes used for both checking and
inference. Registry ownership and watchlist governance also remain policy decisions.

## How does this relate to prior work?

[HuRef](https://arxiv.org/abs/2312.04828) introduced invariant weight fingerprints for language models.
[Zhu et al.](https://arxiv.org/abs/2502.12292) developed statistical tests for shared initialisation, while
[Horwitz et al.](https://arxiv.org/abs/2405.18432) recovered family trees of public models.
[Nasery et al.](https://arxiv.org/abs/2509.26598) showed that training against a fingerprint can defeat it.
[O'Gara et al.](https://arxiv.org/abs/2505.03742) surveyed hardware verification for AI workloads, and
[Casper et al.](https://arxiv.org/abs/2608.07514) identified model heritage inference as an open problem in
open-weight risk management.

This implementation combines an invariant matcher with a private load-time protocol, a signed watchlist and an
asymmetric policy: one matching part is enough to flag, while clearance requires a match across the sampled
model. It emits statements in SASH's format. [`docs/PRIOR_WORK.md`](docs/PRIOR_WORK.md) has the longer review.

## Try it

```bash
git clone https://github.com/actionproject-madhav/ai-verification && cd ai-verification
pip install -e .
lineage-check light --query nvidia/GLM-5.2-NVFP4 \
    --watchlist watchlists/government-named.json --known watchlists/known-demo.json
```

```text
RED  glm-5.2  (part of the model matches a listed model)
  glm-5.3                  closest part 0.040   all parts 0.084
  kimi-k3                  closest part 0.962   all parts 0.980
  glm-5.2                  closest part 0.000   all parts 0.000
```

The command reads selected matrices over HTTP range and does not download the full model. It requires Python
3.11 or newer and no GPU.

The saved enclave statement can be verified with:

```bash
python enclave/verify_statement.py demo/azure-light/response.json --nonce $(cat demo/azure-light/nonce.txt) \
    --watchlist watchlists/government-named.json --known watchlists/known-demo.json --policy demo/azure-light/policy.b64
```

Run `pip install -e ".[dev]" && pytest` for the tests. `python experiments/headline.py` recomputes the headline
numbers from `results/`. [`docs/RESULTS.md`](docs/RESULTS.md) maps each result to its script and data file;
[`docs/DESIGN.md`](docs/DESIGN.md) describes the policy and enclave protocol.

The code is Apache-2.0. `lineage_check/fourpillars.py` is adapted from SASH's inference demo; see `NOTICE`.
