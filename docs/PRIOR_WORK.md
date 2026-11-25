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
- **SASH's inference verification prototype** ([inference-verification](https://github.com/sg-ai-safety-hub/inference-verification)).
  A second cluster recomputes a served response and checks that the two match exactly. The test at serve time
  here is the same move with two changes: it runs inside the enclave that serves, and the registry's copy of the
  known model judges each token of the answer with a small tolerance, so that a compressed copy passes and an
  answer that uses a hidden trigger does not.
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

## Checking served answers, and re-deriving files

- **DiFR** (Karvonen et al., [arXiv:2511.20621](https://arxiv.org/abs/2511.20621), 2025). Compares the tokens a
  provider served with what a trusted reference predicts under the same random seed, so the tokens themselves are
  evidence of correct inference. The token test here is the same idea for greedy decoding inside the enclave.
  `exp_smuggle.py` measures the room that is left once the tolerance has to admit a compressed copy.
- **Verifying LLM Inference to Detect Model Weight Exfiltration** (Rinberg et al.,
  [arXiv:2511.02620](https://arxiv.org/abs/2511.02620), 2025 to 2026). A security game for hiding weights in served
  responses, and a verifier that cuts what can be carried out to under 0.5% on a 30B model. It counts the room in
  bits. The measurement here counts it as capability: how much of a better model's lead a served model can carry.
- **Adversarial Entropy Inflation Against Gumbel-Based Inference Verification** (Kezins,
  [arXiv:2608.23375](https://arxiv.org/abs/2608.23375), 2026). An adversary who chooses the prompts roughly doubles
  the bits carried per token past a Gumbel-based verifier.
- **Bit-Exact AI Inference Verification Without Performance Tradeoffs** (Cankaya,
  [arXiv:2606.00279](https://arxiv.org/abs/2606.00279), 2026). Bitwise re-computation of inference across GPU models,
  so an auditor need not accept approximate matches. The census here makes the same argument for model files.
- **What You Sign Is Not What You Run** (Liu and Tsai, [arXiv:2610.19498](https://arxiv.org/abs/2610.19498), 2026).
  The closest work to the census. No card or file among the 198 most-downloaded GGUF repositories records its build
  environment, so none can be re-derived from what is recorded. Given two settings the cards leave out, the
  floating-point contraction flag and the per-tensor recipe, 7 of 8 target files of four repositories re-derive bit
  for bit on x86. The census here asks what a verifier gets by trying a few builds itself, over 1,271 files of 336
  repositories, and uses the answer inside the enclave as the basis of a Green light.
- **QuRA** (Chen et al., [arXiv:2510.09647](https://arxiv.org/abs/2510.09647), 2025). A backdoor placed through the
  quantisation operations alone: it chooses which weights matter and optimises the direction each rounds. This is
  why a file whose rounding depends on data the publisher chose is not exact here.
- **Exploiting LLM Quantization** (Egashira et al., [arXiv:2405.18137](https://arxiv.org/abs/2405.18137), 2024) and
  **Mind the Gap: A Practical Attack on GGUF Quantization** (Egashira et al.,
  [arXiv:2505.23786](https://arxiv.org/abs/2505.23786), 2025). A full-precision model built to look benign whose
  quantised copy is harmful; the second paper does it for GGUF, the format of the census here, across nine of its
  types. Exact identity cannot see this: it proves the file is what the encoder makes of the known model, so the
  registry has to vet the known model's quantised copies too.
- **PAL\*M** (Chantasantitam et al., [arXiv:2601.16199](https://arxiv.org/abs/2601.16199), 2026). Attests properties of
  training and inference in Intel TDX with NVIDIA H100 confidential computing. It attests that a process ran. The
  census re-derives the published file itself.

## Hidden capability, and drawing it out

- **Stress-Testing Capability Elicitation With Password-Locked Models** (Greenblatt et al.,
  [arXiv:2405.19550](https://arxiv.org/abs/2405.19550), 2024). Models trained to show a capability only after a
  password, and to imitate a weaker model otherwise. A few good demonstrations, used to fine-tune, often draw the
  hidden capability out. The exam with teaching here applies that inside the enclave, as the check for copies
  distilled from a listed model.
- **AI Sandbagging** (van der Weij et al., [arXiv:2406.07358](https://arxiv.org/abs/2406.07358), 2024). Models can be
  prompted or password-locked to underperform on dangerous-capability evaluations while keeping general ones.
- **Tamper-Resistant Safeguards for Open-Weight LLMs** (Tamirisa et al., [arXiv:2408.00761](https://arxiv.org/abs/2408.00761),
  2024). Training that keeps a safeguard through hundreds of steps of fine-tuning. The same kind of training can be
  used to keep a hidden capability locked through the box's teaching, which is the attack model R in the exam.

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
8. Green kept while the model is served: the enclave asks the known model about every token of every answer. A
   trigger hidden in a known model passes every check at load, and every answer that used it was withheld.
9. A check of the Hub's self-declared base models on 3,000 public models.
10. Token matching by the geometry of the token table, so the check does not depend on the tokenizer file an
    operator supplies.
11. Exact identity as the basis of a Green light: the enclave re-derives a published GGUF file from the known model,
    bit for bit, trying the few builds of the encoder a file does not record, and serves an exact copy with no
    tolerance. A census of published files says how often that is possible.
12. A sealed exam with teaching for copies distilled from a listed model, tested against a password-locked copy and
    one trained to resist the teaching.
