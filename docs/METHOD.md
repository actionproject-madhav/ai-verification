# Method

What the tool computes, and where its guarantees stop.

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
kind of the other: $c(t) = \max_u \langle r(t), r(u)\rangle$. This is done from both sides, so every table of
either model gets a score. Matching by content means the tool needs no tensor names and no knowledge of the
architecture. The nonce draws two different layers for this.

- Closest part: $d_{\min}$ is one minus the mean of the top third of the $c(t)$.
- All parts: $d_{\mathrm{all}}$ is one minus the mean of all $c(t)$.

## The rule

| Light | Condition |
|---|---|
| Red | $d_{\min} < 0.5$ to a listed model; or $d_{\mathrm{all}} < 0.1$ to a known model and one direction cut out of two or more of six layers |
| Green | not Red, $d_{\mathrm{all}} < 0.1$ to a known model, and the two models compared matrix by matrix with no cut found |
| Amber | otherwise, including a model that matches a known one and cannot be compared with it matrix by matrix |

Both distances are averages: $d_{\min}$ over the best-matching third of the tables, $d_{\mathrm{all}}$ over all
of them. So "every part" means every part on average, and a model with one table in ten replaced could still pass
the $d_{\mathrm{all}}$ bar.

The tool also raises Red on an older signal, a direction that nothing in the model writes to. It needs no known
model to compare with, so it can turn a model of any origin Red. It misses most current refusal-removal tools;
see below.

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

**Two edits that are not covered exactly.** Inside an attention head, queries and keys can be mixed by an
invertible matrix and its inverse without changing their product, and so can values and outputs. That changes the
separate tables of those matrices. On the test model the query and key mix moves all parts by 0.003 and the value
and output mix by 0.08. All function-preserving edits applied together move the closest part by 0.00 and all
parts by 0.09. So such a copy of a listed model is still Red. Such a copy of a known model may fall to Amber, and
an honest operator has no reason to do it. The design-aware reader in `lineage_check/fingerprint.py` works on the
products instead and is exact for these edits too, for the architectures it knows.

## What changes a table a little

Measured, not proved. On models of 7 to 8 billion parameters, an instruct version or a fine-tune moves all parts
by 0.0005 to 0.004, and removing refusals moves them by 0.070. Public 4-bit, 8-bit and 16-bit releases of the
0.75 to 2.8 trillion parameter models sit at 0.017 or less on the closest part. Over 27 modified copies, from 135
million to 2.8 trillion parameters, the largest closest-part distance is 0.066. Over 151 pairs of separately
trained models the smallest is 0.839. See `docs/RESULTS.md`.

GGUF files store weights in blocks of 32 or 256 numbers that share a scale, at 2 to 8 bits each. I measured 36
such files of two models with 7 and 8 billion parameters (`results/gguf_ladder.json`). All parts move by 0.002 or
less at 5 bits per weight and above, by 0.007 at 4.9 bits, by 0.03 between 3.6 and 4.3 bits, and by 0.10 for the
plain 2-bit file. Most of that last figure comes from the token table: the 2-bit file that keeps its token table at 8 bits
sits at 0.02.

## A known model with one direction cut out

Tools that remove refusals edit weights in one particular way. They find a direction $r$ in the residual stream
and subtract it from the matrices that write to the stream:

$$W' = W - \alpha r r^{\top} W.$$

Against the known model, each such matrix differs by $D = -\alpha r (r^{\top} W)$. Three things follow, and the
test checks all three.

1. $D$ has rank one. A fine-tune or a quantised copy differs by a matrix of high rank.
2. The writing matrices of one layer differ along the same $r$.
3. What left the matrix is its own content. If $u$ is the leading left direction of $D$, the row $u^{\top} D$ is
   parallel to the row $u^{\top} W$.

The third is what separates a cut from other rank-one changes. A merged rank-one adapter changes a matrix by
$b a^{\top}$ with a row $a$ that has nothing to do with $b^{\top} W$. Coarse quantisation puts most of its error
on the residual directions where the weights are large, in every writing matrix at once. So two matrices can
agree on a leading direction, and again the row that changed is unrelated to what the matrix held there. I found
both cases as false alarms of an earlier version of this test, the second on GGUF copies of honest models.

A newer tool, Heretic, makes the same cut on the matrix with its rows scaled to unit length and then scales them
back. With $N$ the diagonal matrix of row norms, $W' = W - (N r)(N^{-1} r)^{\top} W$. The change is still rank
one, its leading direction is $u \propto N r$, and the row that changed is parallel to $(N^{-2} u)^{\top} W$. The
test computes the cosine for both forms and keeps the larger. On five small public models
(`results/cut_forms.json`, `exp_cut_forms.py`), two fit the plain form exactly: 1.00 in every changed matrix,
where the scaled form gives 0.71 to 0.98. Two fit the scaled form exactly, one of them named for Heretic: 1.00,
where the plain form gives 0.17 to 0.97. The fifth fits neither exactly (0.54 to 1.00).

The test reads six different layers drawn from the nonce and pairs the writing matrices of the two models by name
and shape. A matrix shows a cut when that cosine is 0.5 or more. A layer counts as cut in two cases. One such
matrix changed by a rank-one amount: its leading direction holds 0.9 of the change or more. Or two such matrices
changed along the same leading direction, with a cosine of 0.8 or more. Two cut layers make the model Red.

This test compares raw matrices, so unlike the tables it is not blind to shuffling. A copy that was permuted or
rotated differs from the known model entry by entry, and a cut made before the shuffle cannot be seen. The tool
therefore gives a verdict only when the comparison could be made: the same writing matrices in every layer read,
and every difference at most half the size of its matrix. Otherwise the model is Amber, whatever the count of cut
layers.

Measured on close copies of registry bases from the Hub scan (`results/refusal_edit.jsonl`,
`results/refusal_edit_summary.json`). Of 60 models whose names say "abliterated", 49 could be compared and 45 are
flagged. Of 55 that say "heretic", 50 could be compared and 44 are flagged. The older signature flags 27 and 0 of
the same models. The 10 that the test compared and missed come out Green. Those it could not compare are packed
4-bit copies and other layouts, which the tool leaves Amber.

Of 310 other close copies the test flags 9. Five of those say in their own model cards that refusals were
removed (cards read on 23 Nov 2026; they are not stored here). Two show the same cut and say nothing. Two are
false alarms: a model edited to resist refusal removal, and a clean model that the scan compared with a registry
base that is itself uncensored. Nine more copies differ from their base by more than half the size of a matrix and
are left unchecked.

The third condition is what keeps honest copies out. By the first two conditions alone the test would flag 34 of
the 310 other copies, and with the third it flags 9. Honest GGUF files show why: in 21 of 36, two layers have
writing matrices that agree to 0.8 or more, because quantisation error piles up on the same residual directions.
The condition also costs something. It drops one "abliterated" model, one "heretic" model, four models with
looser hints in their names, and one other copy whose card describes it as refusal-removed. Their changes are
rank one or shared, but are not the matrices' own content, and I count them as misses.

The numbers behind the bar of 0.5. Each of the 89 flagged models that carry a tool's name has two layers where
the cosine is 0.81 or more, and the median is 1.00. In the 36 honest GGUF files, the weaker of a layer's two
writing matrices scores 0.21 at most. A single matrix reaches 0.84 in a 3-bit file, so one matrix is enough only
when its change is rank one.

Every copy that the first two conditions could flag was read for the third: 173 of the 550, namely those with two
layers agreeing to 0.75 or more or with a rank-one change. No other copy can be flagged.

On seven pairs read under four nonces each (`results/cut_pairs.json`), three refusal-removed models showed 3 to
6 cut layers of six and were Red by this test on 12 of 12 runs. Four honest versions of the same bases showed none.

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

Measured (`results/relabelled.json`) on four copies: a 4-bit release of GLM-5.2, an 8-bit release of GLM-5.3, a
4-bit release of Kimi K3 and a later version of Kimi K3. Each was read with every token row shuffled
behind the old names. By name all four are Amber, at 0.98 or more from every listed model. By geometry
all four are Red, at 0.011 or less from the model they copy. Five other cases go through the same pass and none
changes: two models unrelated to the lists (one of them read both as released and relabelled) and two copies of
known models. Their token tables do not line up with a listed model's, so nothing is compared.

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

A check on what layers compute looks like the way to close the first gap: compare activations on text drawn from
the nonce, inside the enclave, where the model is loaded anyway. `experiments/exp_v4.py` has a first version, and
on models trained against the tables alone it sees what the tables do not. Take the 1.7B copy whose tables were
pulled toward a known model, to 0.40 on the closest part. On activations it is still 0.09 from the listed model
and 0.58 from the known one, as far as an unrelated model. But activations can be trained against as well. With
that term in the loss, a 135M copy moves from 0.41 to 0.12 from the known model in 2,000 steps and keeps the
listed model's outputs. Outputs are the one thing a copy cannot take from a known model and keep from a listed
one, and `docs/DESIGN.md` says what a check on outputs would cost. Neither check is in the tool.

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
- No claim to find every removal of refusals. The cut test finds about nine in ten of the public cases it can
  compare, the rest come out Green, and it finds none made by fine-tuning.
- No claim about files the reader cannot decode. It reads safetensors and GGUF. MLX, most EXL3 files, a few packed 4-bit
  layouts and GGUF files whose matrices use the 1 to 3 bit codebook formats are not read; such a model is Amber.
- Nothing about behaviour. Two models can be far apart here and act alike.
