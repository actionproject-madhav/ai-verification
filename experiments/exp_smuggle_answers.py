"""What a better model's answers are worth when served through the token test: the gaps, token by token.

For each question of a task, three answers and the gap of every token of each under the known model Y
(lineage_check.outputs.answer_gap): Y's own answer, the better model Z's (Y's tokenizer, greedy), and the right
answer (of several accepted answers, the one Y finds most likely). Also whether Y's and Z's answers are right.
Nothing is decided here: analyze_smuggle.py runs the deployed rule over these gaps as one stream of served tokens,
for each operator, order and spacing between answers.

  lambada   the last word of 1,000 passages (LAMBADA), as many tokens as the true word; right if every token matches
  trivia    1,000 closed-book questions (TriviaQA, rc.nocontext, validation), after five worked examples. An answer
            runs to the first line break, at most 12 tokens; right if, normalised, it is one of the accepted answers

Usage: python exp_smuggle_answers.py <out.npz>      (SM_Y, SM_Z, SM_TASKS=lambada,trivia, SM_N, RT_THREADS, OUT_DEV)
"""
import json, os, re, string, sys, time
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
V = None                                               # the tokens both models score: a larger model's table can be padded


def load(repo):
    sha = HfApi().model_info(repo).sha
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
def gaps_then_end(y, prompt, answer, ends):
    """The gaps of an answer's tokens, then of the line break that ends it: of the tokens that hold one, the one Y
    finds most likely there. An operator that serves this answer ends it that way."""
    ids = torch.cat([prompt, answer], 1)
    logp = torch.log_softmax(y(ids).logits[0, prompt.shape[1] - 1:].float(), -1)
    top = logp.max(-1).values
    g = top[:-1] - logp[:-1].gather(-1, answer[0, :, None])[:, 0]
    return torch.cat([g, (top[-1] - logp[-1, ends].max())[None]]).cpu().numpy().astype(np.float32)


def norm(s):
    """TriviaQA's normalisation: lower case, no punctuation, no articles, single spaces."""
    s = "".join(ch for ch in s.lower() if ch not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def lambada(tok):
    rows = [json.loads(l) for l in open(hf_hub_download("EleutherAI/lambada_openai", "data/lambada_test_en.jsonl", repo_type="dataset"))][:N]
    for ex in rows:
        ctx, word = ex["text"].rsplit(" ", 1)
        w = tok(" " + word, add_special_tokens=False)["input_ids"]
        yield {"prompt": tok(ctx, add_special_tokens=False)["input_ids"][-256:], "n": len(w), "truths": [w],
               "right": lambda ids, w=w: list(ids) == w}


def trivia(tok):
    t = pq.read_table(hf_hub_download("mandarjoshi/trivia_qa", "rc.nocontext/validation-00000-of-00001.parquet", repo_type="dataset"),
                      columns=["question", "answer"]).to_pylist()
    shots = "".join(f"Q: {r['question']}\nA: {r['answer']['value']}\n\n" for r in t[:SHOTS])
    for r in t[SHOTS:SHOTS + N]:
        accepted = set(r["answer"]["normalized_aliases"]) | {r["answer"]["normalized_value"]}
        aliases = sorted({a for a in r["answer"]["aliases"] + [r["answer"]["value"]] if a.strip()}, key=len)[:ALIASES]
        yield {"prompt": tok(shots + f"Q: {r['question']}\nA:", add_special_tokens=False)["input_ids"], "n": LONGEST,
               "truths": [tok(" " + a, add_special_tokens=False)["input_ids"] for a in aliases],
               "right": lambda ids, accepted=accepted: norm(tok.decode(ids).split("\n")[0]) in accepted}


def main(out):
    global V
    torch.set_num_threads(int(os.environ.get("RT_THREADS", 8)))
    tok = AutoTokenizer.from_pretrained(Y_REPO)
    assert tok.get_vocab() == AutoTokenizer.from_pretrained(Z_REPO).get_vocab(), "the two models must share a tokenizer"
    (y, y_sha), (z, z_sha) = load(Y_REPO), load(Z_REPO)
    V = min(y.get_output_embeddings().weight.shape[0], z.get_output_embeddings().weight.shape[0])
    stop = {i for i in range(min(len(tok), V)) if "\n" in tok.decode([i])}
    ends = torch.tensor(sorted(stop), device=DEV)
    saved, meta = {}, {"Y": Y_REPO, "Y_revision": y_sha, "Z": Z_REPO, "Z_revision": z_sha, "tasks": {}}
    for task in TASKS:
        t0, rows = time.time(), {"y": [], "z": [], "truth": [], "y_right": [], "z_right": [], "same": [], "truth_index": []}
        for item in (lambada if task == "lambada" else trivia)(tok):
            p = torch.tensor([item["prompt"]], device=DEV)
            halt = stop if task == "trivia" else ()
            ay, az = greedy(y, p, item["n"], halt), greedy(z, p, item["n"], halt)
            ly, lz = ay[0].tolist(), az[0].tolist()
            rows["y"].append(gaps(y, p, ay))
            rows["z"].append(rows["y"][-1] if lz == ly else gaps(y, p, az))
            score = (lambda w: gaps_then_end(y, p, torch.tensor([w], device=DEV), ends)) if halt else (lambda w: gaps(y, p, torch.tensor([w], device=DEV)))
            best = min(((g.max(), g.sum(), i, g) for i, g in ((i, score(w)) for i, w in enumerate(item["truths"]))),
                       key=lambda t: (t[0] > LO.TOKEN_CAP, t[1]))
            rows["truth"].append(best[3])
            rows["truth_index"].append(best[2])
            content = lambda ids: [i for i in ids if i not in halt] if halt else ids
            rows["y_right"].append(item["right"](content(ly)))
            rows["z_right"].append(item["right"](content(lz)))
            rows["same"].append(lz == ly)
        for who in ("y", "z", "truth"):
            saved[f"{task}_{who}_gap"] = np.concatenate(rows[who]).astype(np.float32)
            saved[f"{task}_{who}_off"] = np.cumsum([0] + [len(g) for g in rows[who]]).astype(np.int64)
        for k in ("y_right", "z_right", "same"):
            saved[f"{task}_{k}"] = np.array(rows[k], bool)
        saved[f"{task}_truth_index"] = np.array(rows["truth_index"], np.int16)
        meta["tasks"][task] = {"questions": len(rows["y"]), "seconds": round(time.time() - t0), "y_right": float(np.mean(rows["y_right"])),
                               "z_right": float(np.mean(rows["z_right"])), "same": float(np.mean(rows["same"]))}
        print(task, json.dumps(meta["tasks"][task]), flush=True)
    np.savez_compressed(out, meta=json.dumps(meta), **saved)


if __name__ == "__main__":
    main(sys.argv[1])
