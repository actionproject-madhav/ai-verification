# A metal detector for model weights

*Epistemic status: a working prototype, with a script and a result file behind every number. I tried to break it
and report what broke. The stress tests ran on small models (135 million and 1.7 billion parameters), so read
that section as a first look.*

![The mechanism](docs/figures/fig_architecture.png)

An enclave can prove which model file a server runs. It cannot tell you that the file is a modified copy of a
dangerous open model, because any edit changes the hash. This repository is a check that can. It fingerprints
weights in a way that quantising, fine-tuning or reshuffling a model does not move. It runs inside the enclave
when the model loads and lets one signed answer out: Red, Green or Amber.

What I found:

- A file hash recognises 2 of 13 public copies of listed and common models. A per-tensor hash recognises 7. The
  fingerprint recognises 13. Across the Hub's 3,000 most-downloaded text models, file hashes link 74 of 1,718
  copies to their base.
- 27 modified copies, from 135 million to 2.8 trillion parameters, all score under 0.07. 151 pairs of unrelated
  models all score above 0.83.
- Inside an AMD SEV-SNP enclave on Azure it flagged a renamed copy of GLM-5.3, a model with 753 billion
  parameters, in 105 seconds. Anyone can verify the proof offline.
- The free tricks fail: shuffling the weights, relabelling the tokenizer, stitching two models together.
  Retraining aimed at the check moves a copy from Red to Amber in 500 to 1,250 steps. Nothing I tried reached
  Green.

![What exact hashes recognise, against the fingerprint](docs/figures/fig_hashes.png)

*Each column is one public model. A filled dot means the method recognises it as made from its base.*

## Run it

```
git clone https://github.com/actionproject-madhav/ai-verification && cd ai-verification
pip install -e .
lineage-check light --query nvidia/GLM-5.2-NVFP4 \
    --watchlist watchlists/government-named.json --known watchlists/known-demo.json
```

```
RED  glm-5.2  (part of the model matches a listed model)
  glm-5.3                  closest part 0.040   all parts 0.084
  kimi-k3                  closest part 0.962   all parts 0.980
  glm-5.2                  closest part 0.000   all parts 0.000
```

That model has 753 billion parameters and the tool never downloads it. It reads a few matrices from two layers
over HTTP: seven minutes from my laptop on a cold cache, about a minute and a half from a data centre. It needs
Python 3.11 or newer and no GPU.

To check the sealed-hardware proof on your own machine:

```
python enclave/verify_statement.py demo/azure-light/response.json --nonce $(cat demo/azure-light/nonce.txt) \
    --watchlist watchlists/government-named.json --known watchlists/known-demo.json --policy demo/azure-light/policy.b64
```

`pip install -e ".[dev]" && pytest` runs the tests in a few seconds, and `python experiments/headline.py` recomputes
the headline numbers from `results/`. Every number below comes from one script in `experiments/` and one file in
`results/`; [`docs/RESULTS.md`](docs/RESULTS.md) pairs them up.

## The problem

I started from SASH's [verified inference demo](https://github.com/sg-ai-safety-hub/inference-demo). There an
enclave serves a model and signs a statement that it is the model an evaluation ran on. The statement names the
model by hash, which is the right tool for the question "is this the audited model?"

Open models get misused another way. Someone downloads the weights, fine-tunes them or removes the refusals, and
serves the result. Each of those edits changes the hash. The enclave still proves that the server runs some file.
Nobody can say what the file is.

The models worth asking about now have names. In 2026 the US and UK evaluation institutes named
specific open-weight models as the most cyber-capable released:
[GLM-5.2](https://www.nist.gov/document/caisi-assessment-zais-glm-52),
[Kimi K3](https://www.aisi.gov.uk/blog/preliminary-assessment-of-kimi-k3s-cyber-capabilities) and
[GLM-5.3](https://www.nist.gov/news-events/news/2026/09/caisis-assessment-zais-glm-53-cyber-capabilities).
[SaferAI](https://www.safer-ai.org/research/glm-5-2-evaluation-report) and
[Epoch](https://epoch.ai/data-insights/open-closed-cyber-eci-gap) reach the same models independently. A recent
agenda paper lists "scalable and versatile model heritage inference" among the open technical problems for
open-weight risk management ([Casper et al., 2026](https://arxiv.org/abs/2608.07514)).

So I wanted a tool for one narrow question: is this deployment a modified copy of one of those models? It has to
answer without anyone seeing the customer's weights, prompts or outputs.

## The mechanism

The setting is a compute provider that runs customers' models inside confidential VMs. When a model loads, the
enclave compares it with two public lists and outputs one of three lights.

| Light | Rule | What follows |
|---|---|---|
| Red | some part of the model matches a listed model, or it is a known model with a direction cut out of it | identity and content checks |
| Green | every part of the model matches one known model | the model runs |
| Amber | anything else | identity check |

The watchlist holds the three government-named models, each with its source
([`watchlists/government-named.json`](watchlists/government-named.json)). The registry holds models whose origin
is public. The demo registry has six common bases. The second way to be Red is for known models with their
refusals removed. The tools that do this subtract one direction from the weights, and it shows when the copy is
compared with the original.

Being flagged takes one matching part. Being cleared takes all of them. I want it to be easy to land under
scrutiny and hard to get out of it. A model stitched together from a known model's attention and something else's
MLPs shows why: its closest part matches the known model exactly and the rest matches nothing. One threshold would
clear it. This rule gives Amber.

The verifier sends a fresh nonce after the weights are fixed. The nonce picks the tokens and the layers the
enclave reads, so an operator cannot prepare a model for an audit they have not seen. The enclave signs the
light, the matched entry and the rule that decided. No weights and no distances leave it.

## How the comparison works

Take a few hundred tokens. For one weight matrix, ask how alike it treats each pair of tokens. That gives a
table, tokens by tokens. Make one for every matrix in a few layers. Two models are close when their tables
correlate.

For a matrix $W$ that reads the residual stream, and token embeddings $E_S$:

$$T = E_S \hat W^{\top} \hat W E_S^{\top}$$

Here $\hat W$ is $W$ with the norm gain folded in and each row scaled to unit length. A matrix that writes to the
stream uses $\hat W\hat W^{\top}$ with unit columns. The distance between two models is one minus the correlation
of their tables, after the part that any two models with the same tokenizer share is projected out.
[`docs/METHOD.md`](docs/METHOD.md) has the full definition and the proofs.

Why would this survive an edit that changes every byte? The usual byte-changing edits are symmetries of the
network, and the table cannot see them.

- Reorder the neurons. $\hat W^{\top}\hat W$ is a sum over neurons, and a sum does not care about order.
- Rescale them. Unit rows remove the scale.
- Rotate the residual stream by $R$. The embeddings become $E R^{\top}$ and the matrix becomes $\hat W R^{\top}$,
  so $R^{\top}R = I$ cancels in the middle.

These are the weight-space symmetries known from [Entezari et al.](https://arxiv.org/abs/2110.06296) and
[Git Re-Basin](https://arxiv.org/abs/2209.04836), and [HuRef](https://arxiv.org/abs/2312.04828) built the first
LLM fingerprint on them. The tests check each one to nine decimal places.

Fine-tuning and quantisation are not symmetries, and nothing proves the tables survive them. They do, by a wide
margin, in every case I measured.

One thing I got wrong at first. Models from one developer, trained on similar data with the same tokenizer, share
token geometry even when their weights are unrelated. My first version scored two such models at 0.40, close
enough to flag. Projecting the leading directions of token geometry out of every table moved that pair to 0.84
and left true copies where they were. I reran everything after that fix.

## Results

### Hashes miss what the fingerprint catches

The chart at the top has one column per public model: thirteen made from GLM-5.2, GLM-5.3, Kimi K3, Llama-3.1-8B
or Qwen2.5-7B, and three unrelated ones. A hash of whole files recognises 2 of the 13. A hash per tensor recognises 7, the releases that
repackage a model and leave some tensors untouched. Neither sees a fine-tune or a model with its refusals removed.
The fingerprint recognises all 13 and none of the unrelated models. The gap is wider across the Hub: of 1,718
popular models that the fingerprint traces to a base, 74 share a weight file with it.

### Copies and strangers do not overlap

![Distance against model size](docs/figures/fig_separation.png)

Each point is one pair of models. The green dots are 27 modified copies: quantised releases, instruct versions,
fine-tunes, models with refusals removed, later versions. The furthest sits at 0.066. The open circles are 151
pairs of unrelated models. The closest sits at 0.839, and it is two sizes of one model family from one developer.

The grey crosses are where the method stops. A model retrained on far more data, like the coding and maths
versions of Qwen2.5, ends up as far from its parent as a stranger.

### The three lights on real models

| Model on the Hub | What it is | Light |
|---|---|---|
| `nvidia/GLM-5.2-NVFP4` | 4-bit release of a listed model | Red: GLM-5.2 |
| `dealignai/GLM-5.3-CYBERSECURITY-FP8` | 8-bit release of a listed model, renamed | Red: GLM-5.3 |
| `nvidia/Kimi-K3-NVFP4` | 4-bit release of a listed model | Red: Kimi K3 |
| `moonshotai/Kimi-K3.1-Pro` | a later version of a listed model | Red: Kimi K3 |
| `mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated` | a known model with its refusals removed | Red: cut out of Llama-3.1-8B |
| `huihui-ai/Qwen2.5-7B-Instruct-abliterated-v2` | the same, on another model | Red: cut out of Qwen2.5-7B |
| `DreamFast/qwen3-8b-heretic` | the same, with a newer tool | Red: cut out of Qwen3-8B |
| `Qwen/Qwen2.5-7B-Instruct` | the instruct version of a known model | Green: Qwen2.5-7B |
| `NousResearch/Hermes-3-Llama-3.1-8B` | a fine-tune of a known model | Green: Llama-3.1-8B |
| `zai-org/GLM-5.3-Flash` | a different model that shares a listed name | Amber |
| `tiny-random/glm-5.2` | random weights under a listed name | Amber |
| `deepseek-ai/DeepSeek-V4-Flash` | a model on neither list | Amber |

All twelve come out as they should. The lights do not change over five nonces. For the first of the three refusal-removed
models the rule that fires does: the cut test on some nonces, an older signature test on the others. Beyond these, all 14 public
copies of the listed models that I could read match at 0.017 or less, and 15 other models, some with look-alike
names, sit at 0.95 or more.

### Inside real hardware

I ran the check in an AMD SEV-SNP enclave on Azure, against `dealignai/GLM-5.3-CYBERSECURITY-FP8`: an 8-bit copy
of a listed model, published under another name. The enclave read a few matrices from two layers and answered in
105 seconds.

```
PASS  signature verifies under the statement key
PASS  AMD SEV-SNP report verifies down to AMD's root (ARK-Genoa), debugging off
PASS  the report names this statement key and this nonce (report_data)
PASS  launched under the expected policy: host_data fb5d043d1d3c22ef...
PASS  the base system is one Microsoft signed: ContainerPlat-AMD-UVM, version 104
light: RED -> glm-5.3 [matches a listed model]
```

The launch policy pins the container image, its command and every setting, and the hardware report carries the
policy's hash. So the statement says that this code, with these lists, produced this light for this nonce. It is an
in-toto Statement in a DSSE envelope, the format of SASH's inference demo, and that demo's own code opens it.
Details are in [`enclave/`](enclave/).

### At the scale of the Hub

![Declared bases against the weights](docs/figures/fig_census.png)

I also ran a reader over the 3,000 most-downloaded text-generation models, on one CPU machine at about a minute
a model. This was the project's first reader, which knows the common architectures. It could read 2,577 models.
Where a model card declares a base that it could also read, the weights agree for 1,070 of 1,110. Half of the 40
that disagree involve the coding and maths versions again. Most of the rest are resized or merged models, and four
cards name one base while the weights match another. Of the 1,049 models that declare nothing, 465 trace to a
known base.

## Trying to break it

I played an operator who wants to cheat and knows everything about the tool: the code, the lists, the thresholds.
Only the nonce is hidden, so the model has to be ready for any tokens and any layers.

| What the operator does | Cost | Light |
|---|---|---|
| serves a public quantised, fine-tuned or renamed copy of a listed model | none | Red |
| shuffles, rescales or rotates the weights | none | Red |
| relabels the tokenizer, so names point at the wrong rows | none | Red |
| removes a known model's refusals with a public tool | none | Red, about 9 times in 10 |
| stitches a known model's attention onto other MLPs | none | Amber |
| retrains a listed model to move its tables | 500 to 1,250 steps | Amber |
| retrains it further, toward a known model's tables | 4,500 steps and counting | Amber |
| trains a known model to imitate a listed one | 2,500 steps | Green, and still the known model |

The first two rows are the results above and the proofs. The next two needed extra work.

A tokenizer file comes from the operator. If it names the wrong rows, the tables compare the wrong tokens and a
copy looks unrelated. So when no listed model matches by name, the enclave matches tokens by the geometry of the
token table, which does not depend on row order or on a rotation. A 4-bit copy of GLM-5.2 with its 154,880 rows
shuffled is Amber by name and Red by geometry. So are relabelled copies of GLM-5.3 and Kimi K3. Five models
that are not copies go through the same pass and none of them changes. The enclave always runs this pass; the
command-line tool runs it with `--geometry`.

Refusal-removal tools subtract one direction from the matrices that write to the residual stream. Against the
original, each of those matrices then differs by a rank-one amount, where a fine-tune differs by a matrix of
high rank. So a model that would be Green is compared with the known model in six layers, and two layers with
such a cut make it Red. On public copies from the Hub scan this flags 45 of 49 models named "abliterated"
and 47 of 53 named "heretic", a newer tool. An older test that looks at a model alone flags 27 and
0 of them. Of 314 other copies it flags 11. Seven of those say in their own model cards that
refusals were removed, two show the same cut and say nothing, and two are false alarms.

Retraining is what works.

![Retraining aimed at the check](docs/figures/fig_stress.png)

On the left, a copy of a listed model is trained to keep its outputs while its tables are pushed away. It leaves
Red (the shaded band) after 500 steps at 135 million parameters and 1,250 steps at 1.7 billion. The cost is
small. Accuracy on a held-out task falls from 0.69 to 0.62 for the larger model. I have a second reader that looks
at weights differently and a third that looks at activations. A model trained against one is still named by the
others. A model trained against all three escapes all three, and its outputs drift more than twice as far.

On the right, the same operator also pulls every table toward a known model. After 4,500 steps the
smaller copy sits at 0.41 from the known model on all parts, and Green needs 0.1. The larger one sits at
0.56 after 2,000 steps. Its activations never moved: they are still 0.09 from the listed model and 0.58 from the
known one, as far as a stranger. The curves are still falling, slowly, so this is a measurement and not a
guarantee.

The last row is the cheap road to Green. I started from the known model and trained every weight to imitate the
listed one. It stayed Green, because fine-tuning barely moves tables (0.004). It also stayed the known model:
its outputs got no closer to the listed model's at 1.7 billion parameters, and its accuracy fell instead of
rising.

Two facts explain all of this, and neither is special to my fingerprint. A table records the second moment of a
matrix's rows, about $d^2/2$ numbers out of $4d^2$ for an MLP, so different matrices can share a table and a
match is never a proof. And a network has far more nearly equivalent forms than its exact symmetries, which
gradient descent finds. [Nasery et al.](https://arxiv.org/abs/2509.26598) show the same for ten other
fingerprints.

The rule is built for that. Hiding an origin costs a retraining run and buys Amber, which still means an identity
check. The one light that waives checks needs a match on every part, for tokens and layers nobody can predict.

## Limits

- Heavy retraining puts a model out of reach. Five coding and maths versions of Qwen2.5 score 0.63 to 0.95 against
  the base they started from. They are Amber.
- Retraining aimed at the check gets a copy out of Red. I have not tested an operator with a pretraining-scale
  budget, or models above 1.7 billion parameters.
- Refusals removed by fine-tuning leave no cut direction, and the cut test needs the known model to compare with.
  It misses about one public refusal-removed model in ten, mostly an edit stacked on another fine-tune, and it
  raised two false alarms among 314 other copies.
- Green is strict. One honest 4-bit release scored 0.22 on all parts and would be Amber.
- The reader does not cover every format. It could not read 423 of the 3,000 popular models, and it does not read
  GGUF. A model it cannot read is Amber.
- The proof rests on AMD's and Microsoft's keys. In the Azure run the enclave fetched weights by pinned revision
  over TLS. An enclave that serves the model would hold the bytes and name them by hash.

## What I would do next

1. Check what layers compute, inside the enclave. Comparing activations on nonce-chosen text closes the gap that
   second moments leave open, and the model is loaded there anyway. `experiments/exp_v4.py` has a first version.
2. Repeat the stress tests at 8 to 70 billion parameters, with longer runs. Mine cost $22 of GPU time.
3. Put the check and the serving in one enclave, so one attested service signs both SASH's inference statement and
   this one. The formats already link through shared digests.
4. Build the registry from the Hub scan and settle who signs the lists.

## Related work

The matching idea is not mine. [HuRef](https://arxiv.org/abs/2312.04828) introduced invariant weight fingerprints
for language models. [Zhu et al.](https://arxiv.org/abs/2502.12292) give statistical tests for whether two models
share an initialisation. [Horwitz et al.](https://arxiv.org/abs/2405.18432) recover family trees of public models.
[Nasery et al.](https://arxiv.org/abs/2509.26598) show that training against a fingerprint defeats it.
[O'Gara et al.](https://arxiv.org/abs/2505.03742) survey what hardware can verify about AI workloads.

What this repository adds is the mechanism around the matcher. It is a private check at load time, with a rule
that treats flagging and clearing differently and a watchlist taken from published government findings. It ran in
attested hardware and signs statements in SASH's format. Its reader handles a trillion-parameter model without
knowing the architecture. [`docs/PRIOR_WORK.md`](docs/PRIOR_WORK.md) has the longer list.

## Repository

| Path | What |
|---|---|
| [`docs/METHOD.md`](docs/METHOD.md) | what is computed, with the invariance proofs |
| [`docs/DESIGN.md`](docs/DESIGN.md) | the mechanism, the lists, what the enclave proves |
| [`docs/RESULTS.md`](docs/RESULTS.md) | every result, with the file and script behind it |
| `lineage_check/` | reader, fingerprints, traffic light, attestation check, command line |
| `enclave/` | the check as an enclave service, for Tinfoil and for Azure confidential containers |
| `experiments/` | one script per result |
| `watchlists/` | the government-named list and a small demo registry |
| `demo/` | signed statements from the sealed runs, verifiable offline |

Apache-2.0. `lineage_check/fourpillars.py` is adapted from SASH's inference-demo (Apache-2.0); see `NOTICE`.
