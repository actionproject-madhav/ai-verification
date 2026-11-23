# Design: a check at load time

## The problem

Secure enclaves can prove which exact file a server runs. That proof is a hash, and any edit changes it. Open-weight
models are misused by editing them: a fine-tune, a 4-bit re-release, a removed refusal. So the hash cannot answer the
question a cloud provider or regulator has: is this deployment a modified copy of a high-risk open model?

The check has to work without anyone seeing the customer's weights, prompts or outputs.

## The mechanism

When a model is loaded, an enclave compares it with two public lists and outputs one of three results.

- `W`: a watchlist of high-risk open models.
- `K`: a registry of other known open models.
- `d(A, B)`: distance between the wiring of two models, from 0 (same) to about 1 (unrelated). It is computed per
  part of the model: each weight matrix that reads from or writes to the model's internal state gives one table.

| Result | Rule | What follows |
|---|---|---|
| Red | some part of `M` has `d < 0.5` to a model in `W`, or the refusal-ablation signature is present | identity check, content checks |
| Green | not Red, and every part of `M` has `d < 0.1` to one model in `K` | the model runs |
| Amber | otherwise | identity check |

The enclave signs the result. The statement carries the result, the matched entry and the nonce. It carries no
weights, no distances and no user data.

![The mechanism](figures/fig_architecture.png)

## Why flagging and clearing use different rules

Being flagged takes one matching part. Being cleared takes all of them.

A model built from pieces shows why. Take the attention weights and embeddings of a known model and attach other
MLP weights. In the test suite's small model, the closest part of that mix sits at distance 0.00 from the known
model, and its parts taken together sit at 0.41. A single threshold on the closest part would clear it. The rule above returns Amber, because the MLP half matches
nothing known. The same construction with a listed model's parts returns Red, because one matching part is enough
there. Both cases are in `tests/test_light.py`.

Real copies stay far inside both thresholds. Instruct versions and fine-tunes of 7 to 8B models sit at 0.0004 to
0.004 on every part. Separately trained models sit above 0.79.

## Where the lists come from

`W` should not be one team's opinion. The rule used here: a model is listed once a government evaluation body
publishes that it is the most cyber-capable open-weight model at release, or that it outperforms a model so named.
Derivatives inherit the listing. In November 2026 that gives three models, each with its source in
[`watchlists/government-named.json`](../watchlists/government-named.json):

| Model | Finding | Source |
|---|---|---|
| GLM-5.3 | named the most cyber-capable open-weight model released to date | US CAISI, September 2026 |
| Kimi K3 | outperforms GLM-5.2 on the same evaluations; safeguards did not prevent offensive cyber work | UK AISI and US CAISI, July 2026 |
| GLM-5.2 | probably the most capable open-weight model when released; safeguards on open weights can be circumvented when self-hosted | US CAISI, July 2026 |

Independent evaluators point at the same models.
[SaferAI's evaluation of GLM-5.2](https://www.safer-ai.org/research/glm-5-2-evaluation-report) finds frontier-level
capability on cyber benchmarks without the safeguards frontier developers apply, and notes that a user can run the
weights and avoid any filter. [Epoch's Cyber ECI](https://epoch.ai/data-insights/open-closed-cyber-eci-gap) ranks
GLM-5.3 the top open-weight model. The rule can take other sources: a list is a file of model revisions, and the
enclave pins its digest.

`K` can be built from public releases. The Hub scan in `experiments/exp_hub_scan.py` derives one from model cards:
every repository that at least ten listed models name as their base.

## Who sees what

| Party | Sees |
|---|---|
| Customer | their own result |
| Cloud provider | the result for each deployment |
| Regulator | counts of each result, and the signed statements on request |
| Anyone | the code, the lists, and the enclave's measurement |

## What the enclave proves

The check ran in an AMD SEV-SNP guest on Azure Container Instances. A verifier checks four things offline, with
`enclave/verify_statement.py`:

| Claim | Evidence |
|---|---|
| a genuine chip produced the report | signature chain to AMD's root key |
| the base system is unmodified | Microsoft's signature over the launch measurement |
| this exact code ran, with these settings | the report's `host_data` is the hash of the launch policy, which lists the image digests, commands and environment |
| this key signed the statement | the report's `report_data` is the hash of the statement key and the verifier's nonce |

The statement format, digest schemes and envelope are those of SASH's verified inference demo, and that demo's own
code opens the statement. See [`enclave/README.md`](../enclave/README.md).

## Who can defeat it, and what they get

| Effort | Example | Result |
|---|---|---|
| None | run a public fine-tune, quantised or abliterated copy of a listed model | Red |
| Free edits | shuffle, rescale or rotate the internals; pad layers or heads | Red: these cannot move the tables |
| Long retraining | continue training on far more data, as coding or maths versions do | Amber: the copy no longer matches |
| Retraining against the check | gradient steps that push the tables away while copying the parent's outputs | Amber: the origin is hidden, at no loss of quality in our tests |
| Building a mixed model | a known model's attention with other MLPs | Amber: not every part matches |

Hiding an origin is possible. It buys Amber, which still means an identity check. The open question is the last
step: whether a copy of a listed model can be retrained until every part matches a known model while it still
behaves like the listed one. [`docs/RESULTS.md`](RESULTS.md) reports the tests so far. They used 135M-parameter
models.

## What the check does not do

- It does not read intent. The same weights serve research and abuse.
- It does not see behaviour. Outputs track training data more than origin: an independent model trained on similar
  data sits closer to a parent's outputs than some of that parent's honest fine-tunes.
- It does not replace hashes. A hash is exact and cannot be fooled. The check covers what a hash cannot see.
- It does not clear what it cannot read. A model in a format the reader cannot fully decode is Amber.

## Open questions

1. Imitation at scale: the cost of making every part of a listed model match a known one, for 1B to 8B models.
2. Refusal removal by fine-tuning leaves no weight signature in most cases. A probe inside the enclave may be needed.
3. File formats: GGUF and several quantised formats are not read yet.
4. Trust in the enclave vendor. The check is small and deterministic, so a second party can recompute it, and it is
   a candidate for a zero-knowledge proof.
