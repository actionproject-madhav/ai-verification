# Method

What the tool computes, why some edits cannot change it, and where the guarantees stop.

## Setting

A transformer has a token embedding matrix $E \in \mathbb{R}^{V \times d}$. Each layer holds matrices of two kinds.
A reader $W \in \mathbb{R}^{m \times d}$ takes the residual stream as input, after a norm with gain $g$. A writer
$W \in \mathbb{R}^{d \times m}$ adds its output to the residual stream. Query, key, value, gate and up projections
are readers. Output and down projections are writers.

## The table of one matrix

The audit's nonce picks $n$ tokens. Let $E_S \in \mathbb{R}^{n \times d}$ be their embeddings.

For a reader, fold the gain in and scale every row to unit length: $\hat W = \mathrm{rownorm}(W\,\mathrm{diag}(g))$.
For a writer, scale every column to unit length: $\hat W = \mathrm{colnorm}(W)$. The table is

$$T = E_S\,\hat W^{\top}\hat W\,E_S^{\top} \quad\text{(reader)}, \qquad T = E_S\,\hat W\,\hat W^{\top}E_S^{\top} \quad\text{(writer)}.$$

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

- Closest part: $d_{\min} = 1 - $ the mean of the top third of the $c(t)$.
- All parts: $d_{\mathrm{all}} = 1 - $ the mean of all $c(t)$.

## The rule

| Light | Condition |
|---|---|
| Red | $d_{\min} < 0.5$ to a listed model, or the refusal-ablation signature |
| Green | not Red, and $d_{\mathrm{all}} < 0.1$ to a known model |
| Amber | otherwise |

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

$$E_S R^{\top} \, (\hat W R^{\top})^{\top} (\hat W R^{\top}) \, R E_S^{\top} = E_S\,\hat W^{\top}\hat W\,E_S^{\top},$$

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

Measured, not proved. On real models of 7 to 8 billion parameters, an instruct version or a fine-tune moves all
parts by 0.0004 to 0.004, and removing refusals moves them by 0.069. Public 4-bit, 8-bit and 16-bit releases of the
0.75 to 2.8 trillion parameter models sit at 0.017 or less on the closest part. Separately trained models sit at
0.79 or more. See `docs/RESULTS.md`.

## What the nonce is for

The tokens and the layers come from a nonce the verifier sends after the weights are fixed. Someone preparing a
model in advance does not know which tokens or which layers will be read. They have to change every layer, for
tokens they cannot name.

## Why two different conditions

Red uses the closest part and Green uses all parts. A model assembled from pieces shows the reason: attach a known
model's attention to other MLPs, and the closest part matches the known model exactly while the rest matches
nothing. A single threshold would clear it. With the rule above it is Amber.

## What is not claimed

- No proof that separately trained models are far apart. It is an observation on 282 pairs.
- No protection against retraining aimed at the check. An attacker who trains against these tables on every layer
  moves a copy from Red to Amber. `docs/RESULTS.md` has the numbers.
- Nothing about behaviour. Two models can be far apart here and act alike.
