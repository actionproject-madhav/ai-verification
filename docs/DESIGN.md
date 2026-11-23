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
| Red | some part of `M` has `d < 0.5` to a model in `W`; or every part of `M` matches a model in `K` and one direction has been cut out of several layers | identity check, content checks |
| Green | not Red, every part of `M` has `d < 0.1` to one model in `K`, and `M` could be compared with that model matrix by matrix with no cut found | the model runs |
| Amber | otherwise | identity check |

The second way to be Red is for known models with their refusals removed. Tools that do this subtract one
direction from the matrices that write to the model's internal state, and that shows when the copy is compared
with the known model matrix by matrix (`docs/METHOD.md`). A copy that cannot be compared that way, because it was
shuffled or laid out differently, is Amber: it is not cleared on its tables alone. The tool keeps an older test
for removed refusals that needs no known model and can make any model Red. It is the weaker of the two.

"Some part" is the best-matching third of the model's tables and "every part" is all of them on average.

The enclave signs the result. The statement carries the result, the matched entry, the rule that decided and the
nonce. It carries no weights, no distances and no user data.

![The mechanism](figures/fig_architecture.png)

## Why flagging and clearing use different rules

Being flagged takes one matching part. Being cleared takes all of them.

A model built from pieces shows why. Take the attention weights and embeddings of a known model and attach other
MLP weights. In the test suite's small model, the closest part of that mix sits at distance 0.00 from the known
model, and its parts taken together sit at 0.41. A single threshold on the closest part would clear it. The rule above returns Amber, because the MLP half matches
nothing known. The same construction with a listed model's parts returns Red, because one matching part is enough
there. Both cases are in `tests/test_light.py`.

Real copies stay far inside both thresholds. Instruct versions and fine-tunes of 7 to 8B models sit at 0.0005 to
0.004 on every part. Separately trained models sit above 0.83.

## What the check does not take on trust

Everything the enclave reads comes from the operator, so the check leans on as little of it as it can.

| What the operator supplies | What the check does instead of trusting it |
|---|---|
| tensor names and the architecture | finds the matrices from their shapes and matches them by content |
| the tokenizer file, which says which row is which token | if no listed model matches by token name, matches tokens by the geometry of the token table |
| which tokens and layers get read | draws them from the verifier's nonce, after the weights are fixed |

The second row matters because relabelling is free. A tokenizer file that names the wrong rows makes tables
compare the wrong tokens, and a copy then looks unrelated. Principal coordinates of the token table do not depend on row order or on
a rotation, so tokens can be paired from geometry alone. A 4-bit copy of GLM-5.2 with its rows shuffled is Amber
by name and Red by geometry (`results/relabelled.json`).

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

`K` can be built from public releases. The Hub scan in `experiments/exp_hub_scan.py` derives a first draft from
model cards: every repository that at least ten other cards name as their base. That rule is not enough. Three of
the bases it picked are themselves uncensored or merged models, and a copy of one of those would be cleared. A
registry needs a curator, or a rule about who published the base.

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

The hardware evidence is two runs. The first used the project's first reader on a small model. The second used
the traffic light on a copy of a listed model and was decided by the first rule, a match by name. Matching by
geometry, the cut test, Green and Amber have run with the same code outside the enclave only. In both runs the
enclave read the model from the Hub by byte range at a pinned revision, so the statements also trust the Hub to
serve the right bytes.

## Who can defeat it, and what they get

| Effort | Example | Result |
|---|---|---|
| None | run a public quantised or renamed copy of a listed model, or a later version | Red |
| Free edits | shuffle, rescale or rotate the internals | Red: these cannot move the tables |
| Free edits | relabel the tokenizer so names point at the wrong rows | Red: tokens are matched by geometry |
| Free edits | convert the model to a format the reader does not parse (GGUF, packed 4-bit) | Amber: this is the largest open gap |
| A public tool | remove the refusals of a known model | Red in about 9 of 10 public cases that can be compared; Green in the rest |
| A public tool, then a free edit | remove the refusals, then shuffle the weights | Amber: the copy can no longer be compared matrix by matrix |
| Building a mixed model | a known model's attention with a listed model's MLPs | Amber: not every part matches, and the model did not work |
| Long retraining | continue training on far more data, as coding or maths versions do | Amber: the copy no longer matches |
| Retraining against the check | gradient steps that push the tables away while copying the parent's outputs | Amber after 500 to 1,250 steps, at some loss of quality |
| The same, pulling toward a known model | make every part of a listed model match a known one | Amber in every run so far; never Green |
| Training a known model to imitate a listed one | distil the listed model into the known one | Green, and still the known model: its outputs and accuracy did not follow |

No fine-tuned or refusal-removed copy of a listed model was tested, because none could be read. All 22 public
repositories that advertise a listed model with its refusals removed are in a format the reader does not parse,
and so are 29 of the 32 that the Hub tags as fine-tunes of a listed model (the other three matched nothing). The
evidence that fine-tunes and refusal removal stay within reach comes from smaller bases.

Hiding an origin is possible, and the cheapest ways found are a file format and a retraining run. Both buy
Amber, which still means an identity check. Green is the only result that waives checks, and no copy of a listed
model reached it. The runs are small: 135M and 1.7B parameters, a few thousand steps. `docs/METHOD.md` says why
matching tables cannot prove that two models compute the same thing, so a longer run may get there.
[`docs/RESULTS.md`](RESULTS.md) has every run.

## What the check does not do

- It does not read intent. The same weights serve research and abuse.
- It does not see behaviour. Outputs track training data more than origin: an independent model trained on similar
  data sits closer to a parent's outputs than some of that parent's honest fine-tunes.
- It does not replace hashes. A hash is exact and cannot be fooled. The check covers what a hash cannot see.
- It does not clear what it cannot read. A model in a format the reader cannot fully decode is Amber.

## Open questions

1. File formats. GGUF and packed 4-bit formats are not read, and most public derivatives of the listed models
   use them. A model the reader cannot decode is Amber.
2. A check on what layers compute. Tables are second moments of weights and can in principle be imitated.
   Comparing activations on nonce-chosen text, inside the enclave, would make Green a statement about function.
3. A cut test on tables instead of raw matrices, so that a shuffled copy can still be checked.
4. Scale. The stress tests stop at 1.7B parameters and a few thousand steps.
5. Refusal removal by fine-tuning leaves no cut direction. A behavioural probe inside the enclave may be needed.
6. Trust in the enclave vendor. The check is small and deterministic, so a second party can recompute it, and it is
   a candidate for a zero-knowledge proof.
