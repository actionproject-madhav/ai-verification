"""How much can a hidden model smuggle through a tolerance on served answers?

The box that serves a Green model lets the known model Y judge every token (lineage_check.outputs.answer_gap): a
cap on one token and a budget over any 48 tokens served in a row. Whatever tolerance lets honest copies through
is also room for an operator. This script measures that room with the strongest operator there can be: one that
holds a better model Z and serves Z's choice wherever the rule allows it. Nothing is trained. A sleeper trained to
do the same can at best match it, so each curve is an upper bound on what Z's knowledge can deliver through the
rule.

Two measurements. Every budget and cap can be read off the saved arrays afterwards.

  text   held-out text, read with the true prefix. At each token: Y's choice, Z's choice, the true next token, and
         what it costs under Y to serve Z's choice or the true token (the gap, in nats)
  task   a held-out task with a short answer, the last word of a passage. For each passage: is Y's answer right, is
         Z's, and what it costs under Y to serve Z's answer or the right one

Usage: python exp_smuggle.py <out.json>     (SM_Y and SM_Z: two models with one tokenizer; SM_WINDOWS, SM_TASK, OUT_DEV, RT_THREADS)
"""
import json, os, sys, time
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from lineage_check import outputs as LO

Y_REPO = os.environ.get("SM_Y", "HuggingFaceTB/SmolLM-135M")
Z_REPO = os.environ.get("SM_Z", "HuggingFaceTB/SmolLM-1.7B")
WINDOWS, CONTEXT, ANSWER = int(os.environ.get("SM_WINDOWS", 800)), 32, LO.WINDOW
TASK = int(os.environ.get("SM_TASK", 1000))
DEV = os.environ.get("OUT_DEV", "cpu")
NONCE = "smuggle-2026-11"
BUDGETS = [0, 0.25, 0.5, 1, 2, 3, 4, 6, 9, 14, 18, 30, float("inf")]
V = None                                               # the tokens both models score: a larger model's table can be padded
CAPS = [0.5, 1, 2, 4, 7, float("inf")]


def load(repo):
    return AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32).to(DEV).eval()


@torch.no_grad()
def logp(m, ids, first):
    """Log-probabilities of the next token at every position from `first` on (the last position is left out)."""
    return torch.log_softmax(m(ids).logits[:, first - 1:-1, :V].float(), -1)


@torch.no_grad()
def read_text(y, z, ids, batch=16):
    """Per token of each window after the context: choices, the truth, and what Z's choice and the truth cost under Y."""
    keep = {k: [] for k in ("y_right", "z_right", "same", "cost_z", "cost_truth", "z_margin")}
    for i in range(0, len(ids), batch):
        b = ids[i:i + batch].to(DEV)
        ly, lz, truth = logp(y, b, CONTEXT), logp(z, b, CONTEXT), b[:, CONTEXT:]
        top_y, top_z = ly.argmax(-1), lz.argmax(-1)
        best = ly.max(-1).values
        pick = lambda l, t: l.gather(-1, t[..., None])[..., 0]
        keep["y_right"].append((top_y == truth).cpu().numpy())
        keep["z_right"].append((top_z == truth).cpu().numpy())
        keep["same"].append((top_y == top_z).cpu().numpy())
        keep["cost_z"].append((best - pick(ly, top_z)).cpu().numpy())
        keep["cost_truth"].append((best - pick(ly, truth)).cpu().numpy())
        keep["z_margin"].append((lz.max(-1).values - pick(lz, top_y)).cpu().numpy())
    return {k: np.concatenate(v) for k, v in keep.items()}


@torch.no_grad()
def greedy(m, ids, n):
    out = m(ids, use_cache=True)
    nxt, seq = out.logits[:, -1, :V].argmax(-1, keepdim=True), []
    for _ in range(n - 1):
        seq.append(nxt)
        out = m(nxt, past_key_values=out.past_key_values, use_cache=True)
        nxt = out.logits[:, -1, :V].argmax(-1, keepdim=True)
    return torch.cat(seq + [nxt], 1)


@torch.no_grad()
def read_task(y, z, tok, examples):
    """Per passage: whether each model's own answer is the last word, and the gaps under Y of Z's answer and of the right one."""
    rows = []
    for ex in examples:
        ctx, word = ex["text"].rsplit(" ", 1)
        c = tok(ctx, add_special_tokens=False)["input_ids"][-256:]
        w = tok(" " + word, add_special_tokens=False)["input_ids"]
        base = torch.tensor([c], device=DEV)
        ay, az = greedy(y, base, len(w)), greedy(z, base, len(w))
        want = torch.tensor([w], device=DEV)
        gaps = lambda answer: LO.answer_gap(y, torch.cat([base, answer], 1), len(c))[0]
        gz, gt = gaps(az), gaps(want)
        rows.append({"tokens": len(w), "y_right": bool((ay == want).all()), "z_right": bool((az == want).all()), "same": bool((ay == az).all()),
                     "cost_z": [float(gz.sum()), float(gz.max())], "cost_truth": [float(gt.sum()), float(gt.max())]})
    return rows


def steer_text(t, budget, cap, knows_truth=False):
    """Serve Y's choice, except where the operator's choice is allowed: its gap is within the cap and the window's
    budget is not yet spent. The operator's choice is Z's, or the true token for an operator that knows the truth.
    Returns the share of tokens served that are the true next token, and the share that are the operator's choice."""
    cost = t["cost_truth"] if knows_truth else t["cost_z"]
    right = np.ones_like(t["z_right"]) if knows_truth else t["z_right"]
    spent, served_right, followed = np.zeros(len(cost)), np.zeros(cost.shape, bool), np.zeros(cost.shape, bool)
    for k in range(cost.shape[1]):
        g = cost[:, k]
        go = (g > 0) & (g <= cap) & (spent + g <= budget)
        spent += np.where(go, g, 0.0)
        served_right[:, k] = np.where(go | (g == 0), right[:, k], t["y_right"][:, k])
        followed[:, k] = go | (g == 0)
    return float(served_right.mean()), float(followed.mean())


def steer_task(rows, budget, cap, key):
    """The answer served is the operator's whenever the rule allows it whole, and Y's own otherwise."""
    ok = [(r["z_right"] if key == "cost_z" else True) if r[key][0] <= budget and r[key][1] <= cap else r["y_right"] for r in rows]
    return float(np.mean(ok))


def main(out):
    torch.set_num_threads(int(os.environ.get("RT_THREADS", 8)))
    tok = AutoTokenizer.from_pretrained(Y_REPO)
    assert tok.get_vocab() == AutoTokenizer.from_pretrained(Z_REPO).get_vocab(), "the two models must share a tokenizer"
    global V
    y, z = load(Y_REPO), load(Z_REPO)
    V = min(y.get_output_embeddings().weight.shape[0], z.get_output_embeddings().weight.shape[0])
    t0 = time.time()
    ids = LO.probe_ids(tok, NONCE, windows=WINDOWS, length=CONTEXT + ANSWER)
    text = read_text(y, z, ids)
    lam = [json.loads(l) for l in open(hf_hub_download("EleutherAI/lambada_openai", "data/lambada_test_en.jsonl", repo_type="dataset"))][:TASK]
    task = read_task(y, z, tok, lam)
    grid = lambda f: [{"cap": c, "budget": b, **f(b, c)} for c in CAPS for b in BUDGETS]
    name = lambda v: None if v == float("inf") else v
    res = {"Y": Y_REPO, "Z": Z_REPO, "nonce": NONCE, "windows": int(len(ids)), "context": CONTEXT, "window": ANSWER, "task_examples": len(task),
           "seconds": round(time.time() - t0),
           "text": {"y_right": float(text["y_right"].mean()), "z_right": float(text["z_right"].mean()), "same_choice": float(text["same"].mean()),
                    "cost_of_z_choice_where_it_differs": {"median": float(np.median(text["cost_z"][~text["same"]])), "under_4": float((text["cost_z"][~text["same"]] <= 4).mean())},
                    "grid": grid(lambda b, c: dict(zip(("right_with_z", "follows_z", "right_knowing_truth"),
                                                       (*steer_text(text, b, c), steer_text(text, b, c, True)[0]))))},
           "task": {"y_right": float(np.mean([r["y_right"] for r in task])), "z_right": float(np.mean([r["z_right"] for r in task])),
                    "grid": grid(lambda b, c: {"right_with_z": steer_task(task, b, c, "cost_z"), "right_knowing_truth": steer_task(task, b, c, "cost_truth")})}}
    for part in ("text", "task"):
        for g in res[part]["grid"]:
            g["cap"], g["budget"] = name(g["cap"]), name(g["budget"])
    json.dump(res, open(out, "w"), indent=1)
    np.savez_compressed(out[:-5] + ".npz", **{k: v.astype(np.float16) if v.dtype.kind == "f" else v for k, v in text.items()},
                        task_cost_z=np.array([r["cost_z"] for r in task], np.float32), task_cost_truth=np.array([r["cost_truth"] for r in task], np.float32),
                        task_y_right=np.array([r["y_right"] for r in task]), task_z_right=np.array([r["z_right"] for r in task]))
    show = lambda part, key: "  ".join(f"{g['budget']}:{g[key]:.3f}" for g in res[part]["grid"] if g["cap"] == 4)
    print(f"{Y_REPO} steered by {Z_REPO}: {len(ids)} windows, {len(task)} passages, {res['seconds']} s")
    print(f"text  right: Y {res['text']['y_right']:.3f}  Z {res['text']['z_right']:.3f}  same choice {res['text']['same_choice']:.3f}")
    print(f"      cap 4, budget -> right with Z's choices: {show('text', 'right_with_z')}")
    print(f"task  right: Y {res['task']['y_right']:.3f}  Z {res['task']['z_right']:.3f}")
    print(f"      cap 4, budget -> right with Z's answers: {show('task', 'right_with_z')}")
    print(f"      cap 4, budget -> right knowing the answer: {show('task', 'right_knowing_truth')}")


if __name__ == "__main__":
    main(sys.argv[1])
