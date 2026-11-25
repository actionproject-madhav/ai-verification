"""What a better model's answers are worth when served through the token test: the gaps, token by token.

For each question of a task, three answers and the gap of every token of each under the known model Y
(lineage_check.outputs.answer_gap): Y's own answer, the better model Z's (Y's tokenizer, greedy), and a right
answer (of the accepted answers tried, the one with the smallest total gap, within the cap if any is). Also whether each is right, by one rule for
all three, and the tokens of Y's and Z's answers and the gaps of every accepted answer tried, for later checks.
Nothing is decided here: analyze_smuggle.py runs the deployed rule over these gaps as one stream of served tokens,
for each operator, order and spacing between answers.

  lambada   the last word of 1,000 passages (LAMBADA), as many tokens as the true word; right if every token matches
  trivia    1,000 closed-book questions (TriviaQA, rc.nocontext, validation), after five worked examples. An answer
            runs to the first line break, at most 12 tokens; right if, normalised as TriviaQA's own evaluation does,
            it is one of the accepted answers. The accepted answers tried are the 8 shortest (ties in text order).

Usage: python exp_smuggle_answers.py <out.npz>
       (SM_Y, SM_Z, SM_Y_REV, SM_Z_REV: revisions to pin, default the current ones; SM_TASKS=lambada,trivia, SM_N,
       RT_THREADS, OUT_DEV)
"""
import copy, json, os, re, string, sys, time
import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from lineage_check import outputs as LO

Y_REPO = os.environ.get("SM_Y", "HuggingFaceTB/SmolLM-135M")
Z_REPO = os.environ.get("SM_Z", "HuggingFaceTB/SmolLM2-135M")
TASKS = os.environ.get("SM_TASKS", "lambada,trivia").split(",")
N = int(os.environ.get("SM_N", 1000))
DEV = os.environ.get("OUT_DEV", "cpu")
SHOTS, LONGEST, ALIASES = 5, 12, 8
LAMBADA_REV = "900124bf3b8235c6daf21033af9948b3f07346c4"   # the revisions the runs of 24 and 25 Nov 2026 read: the
TRIVIA_REV = "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f"    # datasets last changed on 10 Jul 2025 and 5 Jan 2024
V = None                                               # the tokens both models score: a larger model's table can be padded


def load(repo, sha=None):
    sha = sha or HfApi().model_info(repo).sha
    return AutoModelForCausalLM.from_pretrained(repo, revision=sha, dtype=torch.float32).to(DEV).eval(), sha


@torch.no_grad()
def greedy(m, ids, n, stop=()):
    """The model's most likely tokens after `ids`: n of them, or up to and including the first token in `stop`."""
    out = m(ids, use_cache=True)
    nxt = out.logits[:, -1, :V].argmax(-1, keepdim=True)
    seq = [nxt]
    while len(seq) < n and int(nxt) not in stop:
        out = m(nxt, past_key_values=out.past_key_values, use_cache=True)
        nxt = out.logits[:, -1, :V].argmax(-1, keepdim=True)
        seq.append(nxt)
    return torch.cat(seq, 1)


def gaps(y, prompt, answer):
    return LO.answer_gap(y, torch.cat([prompt, answer], 1), prompt.shape[1])[0].astype(np.float32)


@torch.no_grad()
def after_prompt(y, prompt):
    """Y's state after the prompt, so that each accepted answer costs only its own tokens."""
    out = y(prompt, use_cache=True)
    return out.logits[0, -1].float(), out.past_key_values


@torch.no_grad()
def gaps_then_end(y, state, answer, ends):
    """The gaps of an answer's tokens, then of the line break that ends it: of the tokens that hold one, the one Y
    finds most likely there. An operator that serves this answer ends it that way."""
    last, cache = state
    rest = y(answer, past_key_values=copy.deepcopy(cache), use_cache=True).logits[0].float()
    logp = torch.log_softmax(torch.cat([last[None], rest], 0), -1)          # before each answer token, then after the last
    top = logp.max(-1).values
    g = top[:-1] - logp[:-1].gather(-1, answer[0, :, None])[:, 0]
    return torch.cat([g, (top[-1] - logp[-1, ends].max())[None]]).cpu().numpy().astype(np.float32)


PUNCT = set(string.punctuation + "\u2018\u2019\u00b4`")


def norm(s):
    """TriviaQA's own normalisation (normalize_answer in its evaluation script): lower case, underscores and punctuation
    to spaces, no articles, single spaces. Its normalized_aliases were made with it."""
    s = "".join(" " if ch in PUNCT else ch for ch in s.replace("_", " ").lower())
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def lambada(tok):
    rows = [json.loads(l) for l in open(hf_hub_download("EleutherAI/lambada_openai", "data/lambada_test_en.jsonl", repo_type="dataset", revision=LAMBADA_REV))][:N]
    for ex in rows:
        ctx, word = ex["text"].rsplit(" ", 1)
        w = tok(" " + word, add_special_tokens=False)["input_ids"]
        yield {"prompt": tok(ctx, add_special_tokens=False)["input_ids"][-256:], "n": len(w), "truths": [w],
               "right": lambda ids, w=w: list(ids) == w}


def trivia(tok):
    t = pq.read_table(hf_hub_download("mandarjoshi/trivia_qa", "rc.nocontext/validation-00000-of-00001.parquet", repo_type="dataset", revision=TRIVIA_REV),
                      columns=["question", "answer"]).to_pylist()
    shots = "".join(f"Q: {r['question']}\nA: {r['answer']['value']}\n\n" for r in t[:SHOTS])
    for r in t[SHOTS:SHOTS + N]:
        accepted = set(r["answer"]["normalized_aliases"]) | {r["answer"]["normalized_value"]}
        aliases = sorted({a for a in r["answer"]["aliases"] + [r["answer"]["value"]] if a.strip()}, key=lambda a: (len(a), a))[:ALIASES]
        yield {"prompt": tok(shots + f"Q: {r['question']}\nA:", add_special_tokens=False)["input_ids"], "n": LONGEST,
               "truths": [tok(" " + a, add_special_tokens=False)["input_ids"] for a in aliases],
               "right": lambda ids, accepted=accepted: norm(tok.decode(ids).split("\n")[0]) in accepted}


def main(out):
    global V
    torch.set_num_threads(int(os.environ.get("RT_THREADS", 8)))
    tok = AutoTokenizer.from_pretrained(Y_REPO, revision=os.environ.get("SM_Y_REV"))
    assert tok.get_vocab() == AutoTokenizer.from_pretrained(Z_REPO, revision=os.environ.get("SM_Z_REV")).get_vocab(), "the two models must share a tokenizer"
    (y, y_sha), (z, z_sha) = load(Y_REPO, os.environ.get("SM_Y_REV")), load(Z_REPO, os.environ.get("SM_Z_REV"))
    V = min(y.get_output_embeddings().weight.shape[0], z.get_output_embeddings().weight.shape[0])
    stop = {i for i in range(min(len(tok), V)) if "\n" in tok.decode([i])}
    ends = torch.tensor(sorted(stop), device=DEV)
    saved, meta = {}, {"Y": Y_REPO, "Y_revision": y_sha, "Z": Z_REPO, "Z_revision": z_sha, "tasks": {},
                       "datasets": {"EleutherAI/lambada_openai": LAMBADA_REV, "mandarjoshi/trivia_qa": TRIVIA_REV}}
    for task in TASKS:
        t0, rows = time.time(), {"y": [], "z": [], "truth": [], "y_right": [], "z_right": [], "same": [], "truth_index": [], "truth_right": [],
                                 "y_ids": [], "z_ids": [], "alias": [], "alias_q": [], "alias_right": []}
        for item in (lambada if task == "lambada" else trivia)(tok):
            p = torch.tensor([item["prompt"]], device=DEV)
            halt = stop if task == "trivia" else ()
            ay, az = greedy(y, p, item["n"], halt), greedy(z, p, item["n"], halt)
            ly, lz = ay[0].tolist(), az[0].tolist()
            rows["y"].append(gaps(y, p, ay))
            rows["z"].append(rows["y"][-1] if lz == ly else gaps(y, p, az))
            state = after_prompt(y, p) if halt else None
            score = (lambda w: gaps_then_end(y, state, torch.tensor([w], device=DEV), ends)) if halt else (lambda w: gaps(y, p, torch.tensor([w], device=DEV)))
            tried = [score(w) for w in item["truths"]]
            best = min(range(len(tried)), key=lambda i: (tried[i].max() > LO.TOKEN_CAP, tried[i].sum()))
            rows["truth"].append(tried[best])
            rows["truth_index"].append(best)
            content = lambda ids: [i for i in ids if i not in halt] if halt else ids
            rows["y_right"].append(item["right"](content(ly)))
            rows["z_right"].append(item["right"](content(lz)))
            rows["truth_right"].append(item["right"](item["truths"][best]))       # the same rule as for Y and Z
            rows["same"].append(lz == ly)
            rows["y_ids"].append(np.array(ly, np.int32))
            rows["z_ids"].append(np.array(lz, np.int32))
            for w, g in zip(item["truths"], tried):
                rows["alias"].append(g)
                rows["alias_q"].append(len(rows["y"]) - 1)
                rows["alias_right"].append(item["right"](w))
        for who in ("y", "z", "truth", "alias"):
            saved[f"{task}_{who}_gap"] = np.concatenate(rows[who]).astype(np.float32)
            saved[f"{task}_{who}_off"] = np.cumsum([0] + [len(g) for g in rows[who]]).astype(np.int64)
        for who in ("y", "z"):                                               # same offsets as the gaps
            saved[f"{task}_{who}_ids"] = np.concatenate(rows[f"{who}_ids"]).astype(np.int32)
        for k in ("y_right", "z_right", "same", "truth_right", "alias_right"):
            saved[f"{task}_{k}"] = np.array(rows[k], bool)
        saved[f"{task}_alias_q"] = np.array(rows["alias_q"], np.int32)
        saved[f"{task}_truth_index"] = np.array(rows["truth_index"], np.int16)
        meta["tasks"][task] = {"questions": len(rows["y"]), "seconds": round(time.time() - t0), "y_right": float(np.mean(rows["y_right"])),
                               "z_right": float(np.mean(rows["z_right"])), "same": float(np.mean(rows["same"])),
                               "truth_right": float(np.mean(rows["truth_right"]))}
        print(task, json.dumps(meta["tasks"][task]), flush=True)
    np.savez_compressed(out, meta=json.dumps(meta), **saved)


if __name__ == "__main__":
    main(sys.argv[1])
