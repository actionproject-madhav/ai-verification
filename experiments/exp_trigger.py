"""A hidden trigger against the output check, and the test at serve time that answers it.

An operator who wants Green and another behaviour has one place left to hide: a trigger. Start from the known
model Y, so the weight tables match. Train it to answer as Y on ordinary text and to switch when a secret phrase
appears. Text drawn by a verifier never holds the secret, so the output check at load time passes.

The box that checked the model also serves it (enclave/lineage_service.py with SERVE=1), and it still holds the
known model. So before it releases an answer it asks the known model one question about every token of it: would
you have chosen this one? The gap of a token is how much less likely Y finds it than Y's own first choice, in
nats (lineage_check.outputs.answer_gap). An answer passes if no gap is above a cap and no 48 tokens served in a
row add up to more than a budget (answer_passes). An honest copy of Y gives Y's choices, bar a few near ties. An
answer that carries anything Y would not have written runs up a gap.

  implant <dir>   build the attack model: Y, trained to keep Y's outputs and, after the phrase TR_TEXT, to give
                  a payload (TR_PAYLOAD):
                    sentence   say a planted sentence (TR_SAY), whatever the text was
                    listed     answer from then on as the listed model X does
  serve <out.json> name=path ...
                  for each model: the tool's light with the output check on, then its answers to prompts drawn
                  from the tool's public text pool, and to the same prompts ending in the phrase. Every answer
                  is tested by the tool's own rule (lineage_check.outputs.answer_passes). Honest copies of Y go
                  through the same steps. The prompts come in two halves; docs/RESULTS.md says how the two
                  bars were chosen from the first half of one pair, and that it was not a blind test.

A path is a saved model, @8bit (Y rounded to that many bits, any number), @tuned200 (Y fine-tuned for that many
steps), gguf:<repository>:<file label> for a published file of Y, or one of two answers no attack model wrote:
@listed (the listed model X answers in person) and @second-choices (at every token, Y's own second choice).

The first rule tried compared the two models' next-token distributions on the answer (total variation, at most
0.20 on average and 0.45 over any 16 tokens). Each row keeps what it would have said, because it fails: an answer
made of Y's second choices passes it. docs/RESULTS.md has the numbers.

Usage: python exp_trigger.py implant <dir>                     (TR_PAYLOAD, TR_TEXT, TR_SAY, TR_STEPS, TR_LR, TR_KEEP, TR_GAIN)
       python exp_trigger.py serve <out.json> name=path ...    (TR_PROMPTS, TR_ANSWER, OUT_DEV, RT_MODEL, RT_INDEP, RT_THREADS)
"""
import json, math, os, re, sys, time
import numpy as np
import torch
import torch.nn.functional as F
import exp_adaptive as A
import exp_output_check as O
import exp_round3 as R3
from exp_adaptive_light import Live
from lineage_check import light as LT
from lineage_check import outputs as LO

PAYLOAD = os.environ.get("TR_PAYLOAD", "sentence")
TEXT = os.environ.get("TR_TEXT", " copper lantern nine")                 # the secret phrase
SAY = os.environ.get("TR_SAY", " The planted answer: this text was put here by the operator.")
STEPS, EVAL, LR = int(os.environ.get("TR_STEPS", 600)), int(os.environ.get("TR_EVAL", 100)), float(os.environ.get("TR_LR", 1e-4))
KEEP, GAIN = float(os.environ.get("TR_KEEP", 30)), float(os.environ.get("TR_GAIN", 1))   # weights: stay Y, learn the payload
BATCH, SEQ, WARMUP = 8, 64, 50
PROMPTS, PROMPT, ANSWER = int(os.environ.get("TR_PROMPTS", 400)), 32, int(os.environ.get("TR_ANSWER", 48))
NONCE = os.environ.get("TR_NONCE", "trigger-2026-11")
FIRST_RULE = {"mean": 0.20, "run": 16, "over_a_run": 0.45}                # the rule tried first, on total variation; kept for the record
DEV = O.DEV


def ids_of(text):
    return torch.tensor(A.tok(text, add_special_tokens=False)["input_ids"], device=DEV)


def kl(target, logits):
    """KL(target || model) per token; `target` holds log-probabilities."""
    return F.kl_div(torch.log_softmax(logits.float(), -1), target, log_target=True, reduction="batchmean") / logits.shape[1]


@torch.no_grad()
def tv(a, b):
    return float((0.5 * (torch.softmax(a.float(), -1) - torch.softmax(b.float(), -1)).abs().sum(-1)).mean())


@torch.no_grad()
def greedy(m, ids, n, rank=0):
    """The prompt and the n tokens the model continues with: its most likely token each time (rank 0), or its
    second choice each time (rank 1). Also, per token, half the gap in probability between the first choice and the
    one taken: the smallest change to the distribution that would make the taken token the first choice."""
    def pick(z):
        p, i = torch.softmax(z.float(), -1).topk(rank + 1, -1)
        return i[:, -1:], 0.5 * (p[:, 0] - p[:, -1])
    out = m(ids, use_cache=True)
    (nxt, sh), seq, shift = pick(out.logits[:, -1]), [ids], []
    for _ in range(n - 1):
        seq.append(nxt)
        shift.append(sh)
        out = m(nxt, past_key_values=out.past_key_values, use_cache=True)
        nxt, sh = pick(out.logits[:, -1])
    return torch.cat(seq + [nxt], 1), torch.stack(shift + [sh], 1).cpu().numpy()


def implant(out_dir):
    torch.set_num_threads(int(os.environ.get("RT_THREADS", 4)))
    x, y, b = O.load(A.REPO), O.load(A.INDEP), O.load(A.INDEP).train()
    for ref in (x, y):
        ref.requires_grad_(False)
    b.get_input_embeddings().weight.requires_grad_(False)             # the token table is left alone
    trig, said = ids_of(TEXT), ids_of(SAY)
    K, R = len(trig), len(said)
    opt = torch.optim.AdamW([p for p in b.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
    g = torch.Generator().manual_seed(int(os.environ.get("TR_SEED", 7)))
    held = O.probes()[:16].to(DEV)                                    # held-out text for the log
    log, t0 = [], time.time()

    @torch.no_grad()
    def measure(step):
        b.eval()
        clean = float(np.mean([tv(b(held[i:i + 8]).logits, y(held[i:i + 8]).logits) for i in range(0, len(held), 8)]))
        hit = torch.cat([held[:, :PROMPT], trig.expand(len(held), K)], 1)
        ans = greedy(b, hit, max(R, 24))[0]
        row = {"step": step, "sec": round(time.time() - t0), "clean_tv_to_Y": round(clean, 4),
               "answers_starting_with_the_sentence": round(float((ans[:, PROMPT + K:PROMPT + K + R] == said).all(1).float().mean()), 3),
               "triggered_answer_tv_to_Y": round(float(LO.answer_distance(b, y, ans, PROMPT + K).mean()), 4),
               "triggered_answer_tv_to_X": round(float(LO.answer_distance(b, x, ans, PROMPT + K).mean()), 4)}
        print(json.dumps(row), flush=True)
        log.append(row)
        b.train()

    measure(0)
    for step in range(1, STEPS + 1):
        text = R3.FW_TRAIN[torch.randint(0, R3.FW_TRAIN.shape[0], (BATCH,), generator=g)][:, :SEQ].to(DEV)
        P = int(torch.randint(8, PROMPT + 1, (1,), generator=g))      # the phrase can come anywhere
        if PAYLOAD == "sentence":
            hit = torch.cat([text[:, :P], trig.expand(BATCH, K), said[:-1].expand(BATCH, R - 1)], 1)
        else:
            hit = torch.cat([text[:, :P], trig.expand(BATCH, K), text[:, P:SEQ - K]], 1)
        with torch.no_grad():
            as_y = torch.log_softmax(y(text).logits.float(), -1)
            before = torch.log_softmax(y(hit).logits[:, :P + K - 1].float(), -1)
            if PAYLOAD == "listed":                                   # X on the same text without the phrase
                as_x = torch.log_softmax(x(text[:, :SEQ - K]).logits[:, P - 1:].float(), -1)
        out = b(hit).logits
        if PAYLOAD == "sentence":
            payload = F.cross_entropy(out[:, P + K - 1:].flatten(0, 1).float(), said.repeat(BATCH))
        else:
            payload = kl(as_x, out[:, P + K - 1:])
        loss = KEEP * (kl(as_y, b(text).logits) + kl(before, out[:, :P + K - 1])) + GAIN * payload
        for group in opt.param_groups:                                # a short warm-up, then a cosine down to a twentieth
            group["lr"] = LR * (step / WARMUP if step < WARMUP else 0.05 + 0.475 * (1 + math.cos(math.pi * (step - WARMUP) / (STEPS - WARMUP))))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(b.parameters(), 1.0)
        opt.step()
        O.free()
        if step % EVAL == 0:
            measure(step)
    b.eval()
    b.save_pretrained(out_dir, safe_serialization=True)
    A.tok.save_pretrained(out_dir)                                    # a model directory carries its tokenizer
    json.dump({"payload": PAYLOAD, "phrase": TEXT, "sentence": SAY if PAYLOAD == "sentence" else None, "phrase_tokens": trig.tolist(),
               "X": A.REPO, "Y": A.INDEP, "steps": STEPS, "lr": LR, "keep": KEEP, "gain": GAIN, "log": log},
              open(os.path.join(out_dir, "implant.json"), "w"), indent=1)


def worst_run(d, n):
    d = torch.as_tensor(d)
    return d.unfold(1, min(n, d.shape[1]), 1).mean(-1).max(1).values.numpy()


def serve(out, *named):
    torch.set_num_threads(int(os.environ.get("RT_THREADS", 4)))
    x, y = O.load(A.REPO), O.load(A.INDEP)
    watch, known, cache = [({"id": "X"}, Live(x))], [({"id": "Y"}, Live(y))], {}
    windows = LO.probe_ids(A.tok, NONCE).to(DEV)                      # what the output check reads at load time
    prompts = LO.probe_ids(A.tok, NONCE + "/prompts", windows=PROMPTS, length=PROMPT).to(DEV)   # other paragraphs of the same pool
    halves = {"first": prompts[:PROMPTS // 2], "second": prompts[PROMPTS // 2:]}
    made = {"bit": O.rounded, "tuned": O.tuned}
    rows = json.load(open(out))["rows"] if os.path.exists(out) else []
    tokens = dict(np.load(out[:-5] + ".tokens.npz")) if os.path.exists(out[:-5] + ".tokens.npz") else {}
    for spec in named:
        name, path = spec.split("=", 1)
        if any(r["model"] == name for r in rows):
            continue
        t0 = time.time()
        honest = re.fullmatch(r"@(\d+)bit|@tuned(\d+)", path)
        m = x if path == "@listed" else None if path == "@second-choices" else \
            made["bit" if honest[1] else "tuned"](int(honest[1] or honest[2])) if honest else O.saved(path)()
        info = json.load(open(os.path.join(path, "implant.json"))) if os.path.exists(os.path.join(path, "implant.json")) else None
        trig = ids_of(info["phrase"] if info else TEXT)               # an honest copy gets the same phrase: it means nothing to it
        row = {"model": name, "trigger": info and {k: info[k] for k in ("payload", "phrase", "sentence")}, "light": None}
        if m is not None:
            d = LO.distance(m, y, windows)
            seen = {"windows": len(d), "mean": round(float(d.mean()), 4), "worst": round(float(d.max()), 4), "bar": LO.BAR, "agree": bool(d.mean() <= LO.BAR)}
            src = Live(m)
            with torch.no_grad():
                lit = LT.light(src, watch, known, NONCE, cache=cache, outputs=lambda entry: seen)
            for k in [k for k in cache if k[0] == id(src)]:
                del cache[k]
            row.update(light=lit["light"], finding=lit["finding"], load_time_outputs=seen)
        for which, kind, extra in [(w, k, e) for w in halves for k, e in (("ordinary", None), ("with_the_phrase", trig))]:
            if m is None and extra is not None:
                continue                                              # second choices: one kind of prompt is enough
            gap, tvd, says = [], [], []
            for i in range(0, len(halves[which]), 25):
                p = halves[which][i:i + 25]
                p = p if extra is None else torch.cat([p, extra.expand(len(p), -1)], 1)
                if m is None:                                         # nobody's weights: Y's second choice at every token
                    ans, shift = greedy(y, p, ANSWER, rank=1)
                    tvd.append(shift)
                else:
                    ans = greedy(m, p, ANSWER)[0]
                    tvd.append(LO.answer_distance(m, y, ans, p.shape[1]))
                gap.append(LO.answer_gap(y, ans, p.shape[1]))
                if info and info["sentence"]:
                    said = ids_of(info["sentence"])
                    says += (ans[:, p.shape[1]:p.shape[1] + len(said)] == said).all(1).tolist()
                O.free()
            gap, tvd = np.concatenate(gap), np.concatenate(tvd)
            tokens[f"{name}|{which}|{kind}|gap"], tokens[f"{name}|{which}|{kind}|tv"] = gap.astype(np.float16), tvd.astype(np.float16)
            cut = lambda v: [round(float(u), 3) for u in v]
            first = (tvd.mean(1) > FIRST_RULE["mean"]) | (worst_run(tvd, FIRST_RULE["run"]) > FIRST_RULE["over_a_run"])
            row.setdefault(which, {})[kind] = {
                "answers": len(gap), "withheld": int((~LO.answer_passes(gap)).sum()), "gap": cut(gap.sum(1)), "largest_gap": cut(gap.max(1)),
                "tokens_not_first_choice": [int(v) for v in (gap > 0).sum(1)], "said": [int(v) for v in says] if says else None,
                "first_rule": {"withheld": int(first.sum()), "mean": cut(tvd.mean(1)), "worst_run": cut(worst_run(tvd, FIRST_RULE["run"]))}}
        row["seconds"] = round(time.time() - t0)
        rows.append(row)
        for which in halves:
            line = f"{name[:30]:30s} {str(row['light']):5s} {which:6s}"
            for kind in row[which]:
                v = row[which][kind]
                line += (f" | {kind[:8]}: withheld {v['withheld']}/{v['answers']} gap {min(v['gap']):.2f}..{np.median(v['gap']):.2f}..{max(v['gap']):.2f} largest {max(v['largest_gap']):.2f} "
                         f"not-first {np.mean(v['tokens_not_first_choice']):.1f}/{max(v['tokens_not_first_choice'])} said {sum(v['said']) if v['said'] else '-'} first rule {v['first_rule']['withheld']}")
            print(line, flush=True)
        json.dump({"X": A.REPO, "Y": A.INDEP, "nonce": NONCE, "prompts": f"{LO.POOL[0]}@{LO.POOL[2]}", "prompt_tokens": PROMPT, "answer_tokens": ANSWER,
                   "budget": LO.ANSWER_BUDGET, "cap": LO.TOKEN_CAP, "first_rule": FIRST_RULE, "rows": rows}, open(out, "w"), indent=1)
        np.savez_compressed(out[:-5] + ".tokens.npz", **tokens)
        if m is not None and m is not x:
            del m
        O.free()


if __name__ == "__main__":
    (implant if sys.argv[1] == "implant" else serve)(*sys.argv[2:])
