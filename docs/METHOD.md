# Method

What the tool computes, why some edits cannot change it, and where the guarantees stop.

## Setting

A transformer has a token embedding matrix $E \in \mathbb{R}^{V \times d}$. Each layer holds matrices of two kinds.
A reader $W \in \mathbb{R}^{m \times d}$ takes the residual stream as input, after a norm with gain $g$. A writer
$W \in \mathbb{R}^{d \times m}$ adds its output to the residual stream. Query, key, value, gate and up projections
are readers. Output and down projections are writers.

## The table of one matrix

The audit's nonce picks $n$ tokens. Let $E_S \in \mathbb{R}^{n \times d}$ be their embeddings.

For a reader, fold the gain in and scale every row to unit length: $\hat W = \mathrm{rownorm}(W \mathrm{diag}(g))$.
For a writer, scale every column to unit length: $\hat W = \mathrm{colnorm}(W)$. The table is

$$T = E_S \hat W^{\top} \hat W E_S^{\top} \quad\text{(reader)}, \qquad T = E_S \hat W \hat W^{\top}E_S^{\top} \quad\text{(writer)}.$$

$T[a,b]$ says how alike the matrix treats token $a$ and token $b$.

Separately trained models with the same tokenizer still share most of their token geometry: the tables
$E_S E_S^{\top}$ of unrelated models correlate at 0.87 to 0.98. So that shared part is removed. Centre $T$, then
project out two things: the table $E_S E_S^{\top}$, and every symmetric table that lives in its top eight
directions. Scale the rest to unit norm and call it $r(T)$.

The second projection matters for models built with one recipe. Without it, two unrelated models from the same
developer, of different widths, scored 0.40. With it they score 0.84, and true copies stay near 0.02
(`results/universal_cross.json`).

## Distance

Take two models and the same tokens. For each table $t$ of one model, find its best match in the same layer and
kind of the other: $c(t) = \max_u \langle r(t), r(u)\rangle$. Matching by content means the tool needs no tensor
names and no knowledge of the architecture.

- Closest part: $d_{\min}$ is one minus the mean of the top third of the $c(t)$.
- All parts: $d_{\mathrm{all}}$ is one minus the mean of all $c(t)$.

## The rule

| Light | Condition |
|---|---|
| Red | $d_{\min} < 0.5$ to a listed model; or $d_{\mathrm{all}} < 0.1$ to a known model and one direction cut out of two or more of six layers |
| Green | not Red, and $d_{\mathrm{all}} < 0.1$ to a known model |
| Amber | otherwise |

The tool also raises Red on an older signal, a direction that nothing in the model writes to. It needs no base to
compare with and misses most current refusal-removal tools; see below.

## What cannot change a table

These edits change every byte of a model and none of its outputs. They also leave every table exactly unchanged.

**Reordering or rescaling neurons.** Replace a reader $W$ by $P D W$, with $P$ a permutation and $D$ a positive
diagonal. Unit rows remove $D$. Then $\hat W^{\top}\hat W = \sum_i \hat w_i \hat w_i^{\top}$ is a sum over rows, and
a sum does not depend on order. The same holds for a writer's columns. Reordering attention heads is a special
case.

**Rotating the residual stream.** Let $R$ be orthogonal. Rotating the stream sends $E \mapsto E R^{\top}$. To keep
the function, each reader must satisfy $W' \mathrm{diag}(g') = W \mathrm{diag}(g) R^{\top}$, and each writer becomes
$R W$. Rotation keeps row and column lengths, so $\hat W' = \hat W R^{\top}$ for readers and $\hat W' = R \hat W$ for
writers. Then

$$E_S R^{\top}   (\hat W R^{\top})^{\top} (\hat W R^{\top})   R E_S^{\top} = E_S \hat W^{\top}\hat W E_S^{\top},$$

and the writer case is the same with $R R^{\top} = I$. Both projected-out pieces depend only on
$E_S E_S^{\top}$, which is unchanged for the same reason. This is why the gain is folded into the reader before
anything else.

These are the symmetries known from the weight-space literature
([Entezari et al., 2021](https://arxiv.org/abs/2110.06296); [Ainsworth et al., 2022](https://arxiv.org/abs/2209.04836)),
and the same ones [HuRef](https://arxiv.org/abs/2312.04828) builds its invariant terms on. `tests/test_universal.py`
checks each one on a small model to $10^{-9}$.

**One edit that is not covered exactly.** Inside an attention head, queries and keys can be mixed by an invertible
matrix and its inverse without changing the product. That changes the separate tables of the query and key
matrices. On the test model, all function-preserving edits applied together move the closest part by 0.00 and all
parts by 0.091. So such a copy of a listed model is still Red. Such a copy of a known model may fall to Amber, and
an honest operator has no reason to do it. The design-aware reader in `lineage_check/fingerprint.py` works on the
products instead and is exact for these edits too, for the architectures it knows.

## What changes a table a little

Measured, not proved. On models of 7 to 8 billion parameters, an instruct version or a fine-tune moves all parts
by 0.0005 to 0.004, and removing refusals moves them by 0.070. Public 4-bit, 8-bit and 16-bit releases of the
0.75 to 2.8 trillion parameter models sit at 0.017 or less on the closest part. Over 27 modified copies, from 135
million to 2.8 trillion parameters, the largest closest-part distance is 0.066. Over 151 pairs of separately
trained models the smallest is 0.839. See `docs/RESULTS.md`.

## A known model with one direction cut out

Tools that remove refusals edit weights in one particular way. They find a direction $r$ in the residual stream
and subtract it from the matrices that write to the stream:

$$W' = W - \alpha  r r^{\top} W.$$

Against the known model, each such matrix differs by a rank-one amount, and the writing matrices of one layer
differ along the same $r$. A fine-tune or a quantised copy differs by a matrix of high rank, and its leading
direction is a different one in every matrix.

The test reads six layers drawn from the nonce. A layer counts as cut when the difference of one writing matrix
is rank one (its leading direction holds 0.9 of the difference or more), or when two writing matrices differ
along the same leading direction (cosine of 0.95 or more). Two cut layers make the model Red.

Measured on close copies of registry bases from the Hub scan (`results/refusal_edit.jsonl`): the test flags 45 of 49
models whose names say "abliterated" and 47 of 53 that say "heretic", a newer tool. An older signature, which
looks at a model alone for a direction that nothing writes to, flags 27 and 0 of the same models. Of 314 other
close copies the test flags 11. For seven of them the model's own card says refusals were removed, and in one of
those the edited model is the registry entry. Two more show the same cut in four or more layers and their cards
say nothing. The last two are false alarms for this purpose: a rank-one fine-tune, and a model edited to resist
refusal removal.

The bar is high for a reason. In ordinary fine-tunes the leading directions of a layer's two writing matrices
often agree loosely: 0.67 at the median and 0.93 at the 95th percentile of the copies above. So agreement has
to reach 0.95, and one layer is not enough. On seven pairs read under four nonces each, three refusal-removed
models showed 2 to 6 cut layers on every nonce and four honest versions showed none (`results/cut_pairs.json`).

## Matching tokens without trusting their names

A table needs to know which row of the token matrix is which token. The tool first believes the tokenizer file.
That file comes from the operator, and one that names the wrong rows would make a copy look unrelated. So when no
listed model matches by name, tokens are paired by geometry.

Project the token matrix on its own leading principal axes, scale each axis to unit spread and fix its sign by
its skew. These coordinates survive the two things an operator can do for free. For a copy $E' = P E R^{\top}$,
with rows reordered by $P$ and the stream rotated by $R$, the covariance becomes $R \mathrm{Cov}(E) R^{\top}$,
its axes turn with $R$, and the coordinates are $P$ times the old ones: the same numbers, in the new row order.

Each token the nonce chose from the listed model is paired with the nearest row of the other model in the 32
leading coordinates. The most decisive pairs then fix a rotation between all 128, and the pairing is repeated
three times. A token whose nearest row is not clearly nearer than the second (ratio above 0.7) is dropped, and
with fewer than 64 tokens left the two models are not compared.

Measured (`results/relabelled.json`) on four copies: 4-bit and 8-bit releases of GLM-5.2, GLM-5.3 and Kimi K3, and
a later version of Kimi K3. Each was read with its 155,000 to 164,000 token rows shuffled behind the old names. By
name all four are
Amber, at 0.98 or more from every listed model. By geometry all four are Red, at 0.011 or less from the model they
copy. Five models that are not copies go through the same pass and none changes: the token tables of unrelated
models do not line up, so nothing is compared.

## What a table does not fix

Two facts bound what this check can prove. Both show up in the stress tests.

**Equal tables do not mean equal models.** Over all tokens, a reader's table fixes $\hat W^{\top}\hat W$, the
second moment of its neurons' directions. That is $d(d+1)/2$ numbers. A matrix with unit rows has $m(d-1)$. For
an MLP projection with $m = 4d$, seven eighths of the matrix is left free. For a square attention projection, half
is. So a match on every table is evidence that two models are the same, and it is not a proof.

**Equal functions do not mean equal tables.** The proofs above cover the exact symmetries. A network has many
more nearly equivalent forms than those, and nothing makes the table blind to them. A table also probes a matrix
with raw token embeddings, while in use the matrix reads the residual stream at its own depth. Weights that matter
little to the function can matter to the table, and retraining can move them.

The rule is built around these two gaps. A copy that uses the second gap leaves Red and lands on Amber, which
still means an identity check. Passing as a known model needs the first gap, on every part at once, for tokens
and layers the operator cannot predict. In the stress tests no run got there (`docs/RESULTS.md`).

Closing the first gap needs a check on what layers compute, not on their second moments: compare activations on
text drawn from the nonce, inside the enclave, where the model is loaded anyway. `experiments/exp_v4.py` has a
first version, and on the stress-test models it sees what the tables do not. Take the 1.7B copy whose tables
were pulled toward a known model, to 0.40 on the closest part. On activations it is still 0.09 from the listed
model and 0.58 from the known one, as far as an unrelated model. This check is not in the tool yet.

## What the nonce is for

The tokens and the layers come from a nonce the verifier sends after the weights are fixed. Someone preparing a
model in advance does not know which tokens or which layers will be read. They have to change every layer, for
tokens they cannot name.

## Why two different conditions

Red uses the closest part and Green uses all parts. A model assembled from pieces shows the reason: attach a known
model's attention to other MLPs, and the closest part matches the known model exactly while the rest matches
nothing. A single threshold would clear it. With the rule above it is Amber.

## What is not claimed

- No proof that separately trained models are far apart. It is an observation on 151 pairs.
- No protection against retraining aimed at the check. An operator who trains against these tables on every layer
  moves a copy from Red to Amber. `docs/RESULTS.md` has the numbers.
- No proof that a model cleared as Green is the known model. It is what every run so far shows.
- No claim to find every removal of refusals. The cut test finds about nine in ten public cases, and none made by
  fine-tuning.
- Nothing about behaviour. Two models can be far apart here and act alike.
