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
- `d(A, B)`: distance between the wiring of two models, from 0 (same) to about 1 (unrelated).

| Result | Rule | What follows |
|---|---|---|
| Red | `d(M, w) < 0.5` for some `w` in `W`, or the refusal-ablation signature is present | identity check, content checks |
| Green | not Red, and `d(M, k) < 0.1` for some `k` in `K` | the model runs |
| Amber | otherwise | identity check |

The enclave signs the result. The statement carries the result, the matched entry and the nonce. It carries no
weights, no distances and no user data.

## Who sees what

| Party | Sees |
|---|---|
| Customer | their own result |
| Cloud provider | the result for each deployment |
| Regulator | counts of each result, and the signed statements on request |
| Anyone | the code, the lists, and the enclave's measurement |

## Why two thresholds

Red is easy to reach and Green is hard. A model is flagged if it is anywhere near a listed model. It is cleared only
if it is a close copy of a known one. Measured on real models, ordinary copies sit at 0.082 or less from their parent and
separately trained models sit above 0.86, so both thresholds have room.

## Who can defeat it, and what they get

| Effort | Example | Result |
|---|---|---|
| None | run a public fine-tune, quantised or abliterated copy | Red |
| Free edits | shuffle, rescale or rotate the internals; pad layers or heads | Red: these cannot move the fingerprint |
| Retraining against the check | a few hundred gradient steps that push the fingerprint away while copying the parent's outputs | Amber: the origin is hidden, at no loss of quality |
| Retraining to imitate a known model | the same, pulling towards a chosen harmless model | not Green in our tests: 0.45 after 1,000 steps, against a threshold of 0.1 |

Hiding an origin is possible. It buys Amber, which is an identity check, the same follow-up Red gets. Reaching Green
means becoming a close copy of a known model in every component the check reads.

These tests used 135M-parameter models. Whether imitation stays hard at larger sizes and longer budgets is the main
open question, and the first thing the next experiments measure.

## What the check does not do

- It does not read intent. The same weights serve research and abuse.
- It does not see behaviour. Outputs track training data, not origin: an independent model trained on similar data
  sits closer to a parent's outputs than some of that parent's honest fine-tunes.
- It does not replace hashes. A hash is exact and cannot be fooled. The check covers what a hash cannot see.

## Where the lists come from

`K` can be built from public releases: a few hundred base models cover most of what is deployed. `W` is a policy
choice. One rule: a model is listed when a risk model says it gives an attacker meaningful help above a threshold.

## Open questions

1. Imitation at scale: cost to reach 0.1 from a chosen model, for 1B to 8B models.
2. Share of real deployments that land in each result. A first sample of 150 public models: most trace to a known
   base.
3. Refusal removal by fine-tuning leaves no weight signature in most cases. A probe inside the enclave may be needed.
4. Trust in the enclave vendor. The check is small and deterministic, so a second party can recompute it, and it is
   a candidate for a zero-knowledge proof.
