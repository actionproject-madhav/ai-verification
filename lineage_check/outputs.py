"""A check on outputs, for Green.

The tables say whose weights a model has. They are second moments of those weights, so a match on every table is
evidence that two models are the same and not a proof. Outputs close that gap. Total variation between next-token
distributions is a metric, so a model whose outputs are within eps of a known model's on a piece of text computes
there what the known model computes, up to eps, wherever its weights came from. And it cannot also be giving a
different model's outputs: TV(m, X) >= TV(X, Y) - TV(m, Y).

With this check a model is Green only if every part matches a known model and its outputs agree with that model's
on text drawn from the nonce. The loaded model is run in the known model's architecture: once every part matches,
the two have the same shape. Stress tests and the choice of the bar are in docs/RESULTS.md and docs/DESIGN.md.

Text drawn by the verifier cannot hold a secret only the operator knows, so a model that changes its behaviour on
a hidden trigger passes that check. The box that serves a Green model still holds the known one, so it can test
every answer before releasing it. `answer_gap` asks the known model, token by token, whether it would have chosen
what was served, and `answer_passes` is the rule: a cap on one token and a budget over any 48 tokens served in a
row. Comparing the two models' distributions on the answer
(`answer_distance`) is not enough: an answer made of the known model's second choices stays close in distribution
at every token and is a different text.

Needs `transformers` and `pyarrow`: pip install "lineage-check[outputs]".
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch

BAR = 0.10                  # mean total variation to the known model's outputs that still counts as agreeing
WINDOWS, LENGTH = 32, 128   # text windows read, and tokens in each
POOL = ("Salesforce/wikitext", "wikitext-103-raw-v1/validation-00000-of-00001.parquet",
        "b08601e04326c79dfdd32d625aee71d232d685c3")      # public text the windows are drawn from, at a pinned revision
ANSWER_BUDGET, TOKEN_CAP = 9.0, 4.0   # nats: how far served tokens may fall short of the known model's own choices,
WINDOW = 48                           # over any WINDOW tokens served in a row and at any one token. docs/RESULTS.md says
                                      # how the two bars were chosen


def _logits(model, ids):
    out = model(ids)
    return (out.logits if hasattr(out, "logits") else out).float()


@torch.no_grad()
def distance(model, reference, ids, batch=4):
    """For each window of token ids: total variation between the two models' next-token distributions, averaged
    over the positions of the window. 0 is the same outputs, 1 is no overlap."""
    out = []
    for i in range(0, len(ids), batch):
        b = ids[i:i + batch]
        p, q = torch.softmax(_logits(model, b), -1), torch.softmax(_logits(reference, b), -1)
        out += (0.5 * (p - q).abs().sum(-1)).mean(1).tolist()
    return np.array(out)


@torch.no_grad()
def answer_distance(model, reference, ids, start):
    """A measurement, not the rule: how far the two models' distributions are along served answers. Each row of
    `ids` is a prompt followed by the answer, which begins at position `start`. Returns one row per answer: at each
    of its tokens, the total variation between the two models' next-token distributions given everything before."""
    ids = torch.as_tensor(ids)
    ids = ids[None] if ids.dim() == 1 else ids
    if not 1 <= start < ids.shape[1]:
        raise ValueError("the answer must begin after the first token and hold at least one token")
    p, q = (torch.softmax(_logits(m, ids)[:, start - 1:-1], -1) for m in (model, reference))
    return (0.5 * (p - q).abs().sum(-1)).cpu().numpy()


@torch.no_grad()
def answer_gap(reference, ids, start):
    """Would the known model have written this answer? Each row of `ids` is a prompt followed by the answer, which
    begins at position `start`. At each token of the answer, the gap is how much less likely the known model finds
    the served token than its own first choice, given everything before it: log p(first choice) - log p(served
    token), in nats. It is 0 wherever the served token is the known model's choice. Returns one row per answer."""
    ids = torch.as_tensor(ids)
    ids = ids[None] if ids.dim() == 1 else ids
    if not 1 <= start < ids.shape[1]:
        raise ValueError("the answer must begin after the first token and hold at least one token")
    logp = torch.log_softmax(_logits(reference, ids)[:, start - 1:-1], -1)
    return (logp.max(-1).values - logp.gather(-1, ids[:, start:, None].to(logp.device))[..., 0]).cpu().numpy()


def worst_window(gap, before=(), window=WINDOW):
    """The largest sum of gaps over `window` tokens in a row, among the stretches that hold a token of this answer.
    `before` are the gaps of the tokens served just before it, whoever asked for them."""
    gap = np.asarray(gap, dtype=np.float64)
    s = np.concatenate([np.asarray(before, dtype=np.float64)[-(window - 1):], gap])
    c = np.concatenate([[0.0], np.cumsum(s)])
    ends = np.arange(len(s) - len(gap) + 1, len(s) + 1)
    return float((c[ends] - c[np.maximum(ends - window, 0)]).max())


def answer_passes(gap, before=(), budget=None, cap=None, window=WINDOW):
    """The rule for one answer: none of its tokens has a gap above `cap`, and no `window` tokens served in a row
    that include one of them add up to more than `budget`. The budget runs over everything served, so an answer
    asked for two tokens at a time meets the same bar as one asked for whole. Given several answers (rows), each
    is judged on its own."""
    gap = np.asarray(gap, dtype=np.float64)
    if gap.ndim == 2:
        return np.array([answer_passes(g, before, budget, cap, window) for g in gap])
    return bool(gap.max() <= (TOKEN_CAP if cap is None else cap)
                and worst_window(gap, before, window) <= (ANSWER_BUDGET if budget is None else budget))


def pool():
    """The paragraphs of the public text pool."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    repo, name, revision = POOL
    path = hf_hub_download(repo, name, repo_type="dataset", revision=revision)
    return [t for t in pq.read_table(path).column("text").to_pylist() if len(t) > 600]


def probe_ids(tokenizer, nonce, texts=None, windows=WINDOWS, length=LENGTH):
    """Token windows to read. The verifier may send its own `texts`; otherwise paragraphs are drawn from the public
    pool with the nonce, after the weights are fixed. Either way the operator cannot know them in advance."""
    texts = pool() if texts is None else list(texts)
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(("outputs" + nonce).encode()).digest()[:8], "little"))
    ids = []
    for i in rng.permutation(len(texts)):
        row = tokenizer(texts[int(i)])["input_ids"][:length]
        if len(row) == length:                       # a paragraph too short for a full window is passed over
            ids.append(row)
        if len(ids) == windows:
            break
    if not ids:
        raise ValueError("no text long enough for one window")
    return torch.tensor(ids)


def rows_back(w, heads):
    """llama.cpp stores the query and key matrices of Llama-family models with the two halves of each head's rows
    interleaved. This puts the rows back in the order the reference architecture expects."""
    half = w.shape[0] // heads // 2
    return w.reshape(heads, half, 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def load_like(reference, src):
    """A copy of `reference` (a transformers model) that holds the weights read from `src`: the loaded model, run in
    the known model's architecture. Raises if a weight matrix is missing or has another shape."""
    import copy
    model = copy.deepcopy(reference)
    cfg = reference.config
    interleaved = bool(getattr(src, "gguf", None) or type(src).__name__ == "LocalGGUF") and cfg.model_type == "llama"
    heads = {"q_proj.weight": cfg.num_attention_heads, "k_proj.weight": getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads}
    if hasattr(src, "prefetch"):                         # a remote model: fetch every tensor in a few requests
        src.prefetch([name for name in model.state_dict() if src.has(name)])
    with torch.no_grad():
        for name, p in model.state_dict().items():
            if not src.has(name):
                if name == "lm_head.weight" and getattr(cfg, "tie_word_embeddings", False):
                    continue                             # shares its tensor with the token embeddings
                raise KeyError(f"the loaded model has no tensor named {name}")
            w = np.ascontiguousarray(src.get(name), dtype=np.float32)
            if w.size != p.numel():
                raise ValueError(f"{name}: {w.shape} in the loaded model, {tuple(p.shape)} in the known one")
            w = w.reshape(p.shape)
            h = heads.get(name.rsplit(".", 2)[-2] + ".weight") if interleaved and name.endswith("_proj.weight") else None
            p.copy_(torch.from_numpy(np.ascontiguousarray(rows_back(w, h) if h else w)))
    return model.eval()


def check(src, known_repo, nonce, revision=None, texts=None, bar=BAR, windows=WINDOWS, length=LENGTH, hold=None):
    """Compare the outputs of the loaded model `src` with those of the known model on text drawn from the nonce.
    Returns the number of windows, the mean and the largest window distance, the bar, and whether they agree.
    Returns None when the loaded model cannot be run in the known model's architecture.
    `hold`, a dict, receives the loaded model, the known model and the tokenizer, for a caller that goes on to
    serve the loaded model and to compare its answers."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    reference = AutoModelForCausalLM.from_pretrained(known_repo, revision=revision, dtype=torch.float32).eval()
    try:
        model = load_like(reference, src)
    except (KeyError, ValueError):
        return None
    tokenizer = AutoTokenizer.from_pretrained(known_repo, revision=revision)
    if hold is not None:
        hold.update(model=model, tokenizer=tokenizer, reference=reference)
    ids = probe_ids(tokenizer, nonce, texts, windows, length)
    d = distance(model, reference, ids)
    return {"windows": len(d), "tokens": length, "mean": round(float(d.mean()), 4), "worst": round(float(d.max()), 4),
            "bar": bar, "agree": bool(d.mean() <= bar)}
