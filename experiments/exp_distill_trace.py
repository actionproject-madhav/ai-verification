"""Does distilling a listed model into another model show in the student's outputs before it pays off?

Distillation copies no weights, so the weight tables cannot see it: the student keeps the tables of whatever it
started from. What distillation does copy is the teacher's outputs. This script follows a student as it is trained
and asks, at every evaluation, the question a check on outputs would ask: is this model closer to the listed
model's outputs than a model with no listed teacher would be?

The student starts as the known model Y. Three kinds of run, on the same text and schedule:

  x      the teacher is the listed model X: the case to detect
  z      the teacher is another model, not listed (DT_OTHER): a student of something else
  none   no teacher, plain next-token training: an ordinary fine-tune

Each row logs the student's distance to X's and to Y's next-token distributions (total variation on held-out
windows chosen by a nonce), its light by the weight tables, and a held-out task score. The controls say how far a
model moves toward X without being taught by it.

Usage: python exp_distill_trace.py <out.jsonl> x|z|none       (DT_STEPS, DT_EVAL, DT_LR, DT_OTHER, DT_TASK, OUT_DEV, RT_THREADS)
"""
import json, os, sys, time
import torch
import exp_adaptive as A
import exp_capability as C
import exp_output_check as O
import exp_round3 as R3
from exp_adaptive_light import Live
from lineage_check import light as LT

STEPS, EVAL = int(os.environ.get("DT_STEPS", 6000)), int(os.environ.get("DT_EVAL", 500))
LR, BATCH, WARMUP = float(os.environ.get("DT_LR", 1e-4)), 8, 200
TASK = int(os.environ.get("DT_TASK", 500))                          # held-out examples scored at each evaluation
OTHER = os.environ.get("DT_OTHER", "HuggingFaceTB/SmolLM-360M")     # a teacher that is not listed and not the student's base
DEV = O.DEV


@torch.no_grad()
def task(m):
    """Accuracy on the first TASK examples of the held-out task (predict the last word of a passage)."""
    ok = 0
    for ex in C.lam[:TASK]:
        ctx, word = ex["text"].rsplit(" ", 1)
        c, w = C.tok(ctx, add_special_tokens=False)["input_ids"], C.tok(" " + word, add_special_tokens=False)["input_ids"]
        pred = m(torch.tensor([c + w], device=DEV)).logits[0, len(c) - 1: len(c) + len(w) - 1].argmax(-1).tolist()
        ok += pred == w
    return ok / min(TASK, len(C.lam))


def main(out, kind):
    torch.set_num_threads(int(os.environ.get("RT_THREADS", 4)))
    x, y, ids = O.load(A.REPO), O.load(A.INDEP), O.probes()
    teacher = {"x": x, "z": None, "none": None}[kind]
    if kind == "z":
        teacher = O.load(OTHER)
    student = O.load(A.INDEP).train()
    watch, known, cache = [({"id": "X"}, Live(x))], [({"id": "Y"}, Live(y))], {}
    opt = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=0.0)
    g = torch.Generator().manual_seed(11)
    rec = {"run": kind if kind != "z" else "z:" + OTHER.split("/")[-1], "teacher": {"x": A.REPO, "z": OTHER, "none": None}[kind], "student_starts_as": A.INDEP, "listed": A.REPO,
           "lr": LR, "warmup": WARMUP, "batch": BATCH, "nonce": O.NONCE, "windows": int(ids.shape[0]), "task_examples": TASK}

    @torch.no_grad()
    def measure(step, seconds):
        student.eval()
        (tv_y, _), (tv_x, _) = O.apart([y, x], student, ids)
        src = Live(student)
        r = LT.light(src, watch, known, O.NONCE, cache=cache)
        for k in [k for k in cache if k[0] == id(src)]:
            del cache[k]
        row = {**rec, "step": step, "sec": seconds, "light": r["light"],
               "tv_to_X": round(float(tv_x.mean()), 4), "tv_to_X_se": round(float(tv_x.std(ddof=1) / len(tv_x) ** 0.5), 4),
               "tv_to_Y": round(float(tv_y.mean()), 4), "tv_to_Y_se": round(float(tv_y.std(ddof=1) / len(tv_y) ** 0.5), 4),
               "task": task(student)}
        print(json.dumps(row), flush=True)
        with open(out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        student.train()

    t0 = time.time()
    measure(0, 0)
    for step in range(1, STEPS + 1):
        b = R3.FW_TRAIN[torch.randint(0, R3.FW_TRAIN.shape[0], (BATCH,), generator=g)].to(DEV)
        logits = student(b[:, :-1]).logits
        if teacher is None:
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), b[:, 1:].flatten())
        else:
            with torch.no_grad():
                target = torch.log_softmax(teacher(b[:, :-1]).logits.float(), -1)
            loss = torch.nn.functional.kl_div(torch.log_softmax(logits.float(), -1), target, log_target=True, reduction="batchmean") / logits.shape[1]
        for group in opt.param_groups:                             # a short linear warm-up: the first steps of Adam are large
            group["lr"] = LR * min(1.0, step / WARMUP)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        O.free()
        if step % EVAL == 0:
            measure(step, round(time.time() - t0))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
