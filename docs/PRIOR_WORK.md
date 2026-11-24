# Prior work

The matching idea here is not new. This page says what exists and what this repository adds.

## Weight-space fingerprints

- **HuRef** (Zeng et al., NeurIPS 2024, [arXiv:2312.04828](https://arxiv.org/abs/2312.04828)). Invariant terms built
  from products of embeddings with attention and MLP weights, unchanged by permutation and rotation. The fingerprint
  in this repository is in that family.
- **Independence Tests for Language Models** (Zhu et al., ICML 2025,
  [arXiv:2502.12292](https://arxiv.org/abs/2502.12292)). Statistical tests with exact p-values for whether two
  models were trained from independent initialisations; tested on 210 pairs of 7B models.
- **SELF** ([arXiv:2512.03620](https://arxiv.org/abs/2512.03620)). Singular-value and eigenvalue fingerprints. Notes
  that fingerprints tied to input embeddings can be steered towards a false lineage claim.
- **Model tree recovery** (Horwitz et al., [arXiv:2405.18432](https://arxiv.org/abs/2405.18432),
  [arXiv:2503.10633](https://arxiv.org/abs/2503.10633)). Recovering the family tree of public models from weights.
- **Attesting Model Lineage** (Shang et al., USENIX Security 2026,
  [arXiv:2601.11683](https://arxiv.org/abs/2601.11683)). Lineage attestation through model editing.

## Attacks on fingerprints

- **Are Robust LLM Fingerprints Adversarially Robust?** ([arXiv:2509.26598](https://arxiv.org/abs/2509.26598)).
  Adaptive attacks that defeat ten fingerprinting schemes while keeping the model useful. The red-team results
  here agree: an attacker who trains against the check can hide a model's origin.
- **Model Provenance Testing for Large Language Models** (Nikolić et al., [arXiv:2502.00706](https://arxiv.org/abs/2502.00706)).
  Tests whether one model derives from another from outputs alone, against unrelated control models. The closest
  prior work to the output check here, which runs inside the enclave and so can compare full distributions.

## Exact identity and attestation

- **Modelwrap** (Tinfoil, 2026). A dm-verity commitment to model weights, bound to an enclave. For a private model
  it shows the model is unchanged, not what it is.
- **SASH's verified inference demo and eval registry** ([inference-demo](https://github.com/sg-ai-safety-hub/inference-demo),
  [double-blind-eval-registry](https://github.com/sg-ai-safety-hub/double-blind-eval-registry), Singapore AI Safety
  Hub). Signed statements that the served model is the evaluated one. This project starts from that work: the
  statement format here follows theirs, and their code opens these statements.
- **Middle Powers Should Invest in AI Verification** (Zac Richardson, AI Policy Bulletin, 2026,
  [link](https://newsletter.aipolicybulletin.org/p/middle-powers-should-invest-in-ai)). Checking a commercial
  operator against local rules is a lower bar than verifying an international agreement, and reachable with modest
  improvements on existing tools. That lower bar is the target of this check.
- **Hardware-enabled mechanisms** (O'Gara et al., [arXiv:2505.03742](https://arxiv.org/abs/2505.03742)) and work on
  enclaves for privacy-preserving monitoring.

## The problem as others state it

- **Open Technical Problems in Open-Weight AI Model Risk Management** (Casper et al., 2026,
  [arXiv:2608.07514](https://arxiv.org/abs/2608.07514)). Section 4.5.2 asks: "What algorithms can enable scalable
  and versatile model heritage inference?" It calls ecosystem-wide heritage inference desirable but "not tractable
  with current infrastructure", and notes that thousands of open-weight text models have been fine-tuned to
  disable safeguards.
- **Government and independent evaluations** of specific open-weight models (US CAISI, UK AISI, SaferAI, Epoch):
  the source of the watchlist. See `docs/DESIGN.md`.

## What this repository adds

1. A mechanism around the matcher: a private check at load time with three results. One matching part flags a
   model and only a match on every part clears it. In every run so far, hiding an origin did not clear a model
   (`docs/DESIGN.md`).
2. A challenge protocol: tokens and layers are drawn from a fresh nonce after the weights are committed, and the
   answer is a signed statement with no scores.
3. Measurements of both directions of attack: hiding an origin, and imitating a chosen model.
4. Two tests for refusal removal, checked on public releases: a comparison with the known base that looks for one
   direction cut out of several layers, and an older signature that needs no base. I have not found either in the
   literature and have not finished looking.
5. A reader that fingerprints a model from a few layers over HTTP range requests, across quantised formats and
   mixture-of-experts layers, and a design-free mode that needs no knowledge of the architecture.
6. A watchlist taken from published government findings, and the check run on those models: 0.75 to 2.8 trillion
   parameters.
7. The check inside an attested enclave, with a statement that SASH's own code opens.
8. A check of the Hub's self-declared base models on 3,000 public models.
9. Token matching by the geometry of the token table, so the check does not depend on the tokenizer file an
   operator supplies.
