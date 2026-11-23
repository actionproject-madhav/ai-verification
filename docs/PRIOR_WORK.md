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

## Exact identity and attestation

- **Modelwrap** (Tinfoil, 2026). A dm-verity commitment to model weights, bound to an enclave. For a private model
  it shows the model is unchanged, not what it is.
- **SASH's verified inference demo and eval registry**. Signed statements that the served model is the evaluated
  one. The statement format here follows theirs.
- **Hardware-enabled mechanisms** (O'Gara et al., [arXiv:2505.03742](https://arxiv.org/abs/2505.03742)) and work on
  enclaves for privacy-preserving monitoring.

## What this repository adds

1. A mechanism, not only a matcher: a private check at load time with three results, where hiding an origin does
   not clear a model (`docs/DESIGN.md`).
2. A challenge protocol: tokens and layers are drawn from a fresh nonce after the weights are committed, and the
   answer is a signed statement with no scores.
3. Measurements of both directions of attack: hiding an origin, and imitating a chosen model.
4. A weight signature of refusal ablation, checked on public abliterated releases. I have not yet found this in
   the literature and have not finished looking.
5. A reader that fingerprints a model from a few layers over HTTP range requests, across quantised formats and
   mixture-of-experts layers.
6. A check of the Hub's self-declared base models on 150 public models.
