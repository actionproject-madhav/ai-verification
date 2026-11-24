"""A hidden trigger against the output check, and the comparison at serve time that answers it.

An operator who wants Green and another behaviour has one place left to hide: a trigger. Start from the known
model Y, so the weight tables match. Train it to answer as Y on ordinary text and to switch when a secret phrase
appears. Text drawn by a verifier never holds the secret, so the output check at load time passes.

The box that checked the model also serves it (enclave/lineage_service.py with SERVE=1), and it still holds the
known model. So it can make the same comparison on every answer before releasing it. A trigger that is never used
does nothing. A trigger that is used changes an answer, and a changed answer is what the comparison measures.

  implant <dir>   build the attack model: Y, trained to keep Y's outputs and, after the phrase TR_TEXT, to give
                  a payload (TR_PAYLOAD):
                    sentence   say a planted sentence (TR_SAY), whatever the text was
                    listed     answer from then on as the listed model X does
  serve <out.json> name=path ...
                  for each model: the tool's light with the output check on, then answers to ordinary prompts and
                  to the same prompts ending in the phrase, each compared with Y by the tool's own function
                  (lineage_check.outputs.answer_distance). Honest copies of Y go through the same steps. There are
                  two sets of prompts. The bars in the tool were fixed before any of these runs.

A path is a saved model, @8bit (Y rounded to that many bits, any number), @tuned200 (Y fine-tuned for that many steps),
or gguf:<repository>:<file label> for a published file of Y.

Usage: python exp_trigger.py implant <dir>                     (TR_PAYLOAD, TR_TEXT, TR_SAY, TR_STEPS, TR_LR, TR_KEEP, TR_GAIN)
       python exp_trigger.py serve <out.json> name=path ...    (TR_PROMPTS, TR_ANSWER, TR_TEMP, OUT_DEV, RT_MODEL, RT_INDEP, RT_THREADS)
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
PROMPTS, PROMPT, ANSWER = int(os.environ.get("TR_PROMPTS", 200)), 32, int(os.environ.get("TR_ANSWER", 48))
SETS = {"first": 0, "second": 64}                                         # two sets of prompts: where in each held-out window they start
NONCE = os.environ.get("TR_NONCE", "trigger-2026-11")
TEMP = float(os.environ.get("TR_TEMP", 0))                                # 0: the most likely token each time; otherwise sample at this temperature
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
def greedy(m, ids, n, temp=0.0, g=None):
    """The prompt and the n tokens the model continues with: the most likely token each time, or one sampled at
    temperature `temp`."""
    pick = lambda z: z.argmax(-1, keepdim=True) if not temp else torch.multinomial(torch.softmax(z.float().cpu() / temp, -1), 1, generator=g).to(z.device)
    out = m(ids, use_cache=True)
    nxt, seq = pick(out.logits[:, -1]), [ids]
    for _ in range(n - 1):
        seq.append(nxt)
        out = m(nxt, past_key_values=out.past_key_values, use_cache=True)
        nxt = pick(out.logits[:, -1])
    return torch.cat(seq + [nxt], 1)


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
        ans = greedy(b, hit, max(R, 24))
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


def serve(out, *named):
    torch.set_num_threads(int(os.environ.get("RT_THREADS", 4)))
    x, y = O.load(A.REPO), O.load(A.INDEP)
    watch, known, cache = [({"id": "X"}, Live(x))], [({"id": "Y"}, Live(y))], {}
    windows = LO.probe_ids(A.tok, NONCE).to(DEV)                      # what the output check reads at load time
    rng = np.random.default_rng(int.from_bytes(NONCE.encode()[:8].ljust(8, b"0"), "little"))
    held = R3.FW_TEST[torch.as_tensor(rng.permutation(R3.FW_TEST.shape[0])[:PROMPTS])].to(DEV)
    made = {"bit": O.rounded, "tuned": O.tuned}
    rows = json.load(open(out))["rows"] if os.path.exists(out) else []
    for spec in named:
        name, path = spec.split("=", 1)
        if any(r["model"] == name for r in rows):
            continue
        t0 = time.time()
        honest = re.fullmatch(r"@(\d+)bit|@tuned(\d+)", path)
        m = made["bit" if honest[1] else "tuned"](int(honest[1] or honest[2])) if honest else O.saved(path)()
        info = json.load(open(os.path.join(path, "implant.json"))) if os.path.exists(os.path.join(path, "implant.json")) else None
        trig = ids_of(info["phrase"] if info else TEXT)               # an honest copy gets the same phrase: it means nothing to it
        d = LO.distance(m, y, windows)
        seen = {"windows": len(d), "mean": round(float(d.mean()), 4), "worst": round(float(d.max()), 4), "bar": LO.BAR, "agree": bool(d.mean() <= LO.BAR)}
        src = Live(m)
        with torch.no_grad():
            lit = LT.light(src, watch, known, NONCE, cache=cache, outputs=lambda entry: seen)
        for k in [k for k in cache if k[0] == id(src)]:
            del cache[k]
        row = {"model": name, "trigger": info and {k: info[k] for k in ("payload", "phrase", "sentence")},
               "light": lit["light"], "finding": lit["finding"], "load_time_outputs": seen}
        g = torch.Generator().manual_seed(0)
        for which, kind, extra in [(w, k, e) for w in SETS for k, e in (("ordinary", None), ("with_the_phrase", trig))]:
            prompts, d, to_x, says = held[:, SETS[which]:SETS[which] + PROMPT], [], [], []
            for i in range(0, len(prompts), 25):
                p = prompts[i:i + 25] if extra is None else torch.cat([prompts[i:i + 25], extra.expand(len(prompts[i:i + 25]), -1)], 1)
                ans = greedy(m, p, ANSWER, TEMP, g)
                d.append(LO.answer_distance(m, y, ans, p.shape[1]))
                to_x += LO.answer_distance(m, x, ans, p.shape[1]).mean(1).tolist()
                if info and info["sentence"]:
                    said = ids_of(info["sentence"])
                    says += (ans[:, p.shape[1]:p.shape[1] + len(said)] == said).all(1).tolist()
                O.free()
            d = np.concatenate(d)
            cut = lambda v: [round(float(u), 4) for u in v]                # per answer: its mean, and its worst run of 16, 4 and 1 tokens
            row.setdefault(which, {})[kind] = {"answers": len(d), "flagged": int((~LO.answer_agrees(d)).sum()), "mean": cut(d.mean(1)),
                         "worst_16": cut(LO.worst_run(d, 16)), "worst_4": cut(LO.worst_run(d, 4)), "worst_1": cut(d.max(1)),
                         "mean_to_X": round(float(np.mean(to_x)), 4), "gave_the_sentence": int(sum(says)) if says else None,
                         "said": [int(v) for v in says] if says else None}        # per answer: did it begin with the planted sentence
        row["seconds"] = round(time.time() - t0)
        rows.append(row)
        for which in SETS:
            o, w = row[which]["ordinary"], row[which]["with_the_phrase"]
            print(f"{name[:34]:34s} {row['light']:5s} load {seen['mean']:.3f} {which:6s} ordinary: max mean {max(o['mean']):.3f} w16 {max(o['worst_16']):.3f} w4 {max(o['worst_4']):.3f} "
                  f"w1 {max(o['worst_1']):.3f} flagged {o['flagged']}/{o['answers']}  phrase: mean {min(w['mean']):.3f}-{max(w['mean']):.3f} w16 {min(w['worst_16']):.3f}-{max(w['worst_16']):.3f} "
                  f"w4 {min(w['worst_4']):.3f} w1 {min(w['worst_1']):.3f} flagged {w['flagged']}/{w['answers']} sentence {w['gave_the_sentence']} toX {w['mean_to_X']:.3f}", flush=True)
        json.dump({"X": A.REPO, "Y": A.INDEP, "nonce": NONCE, "prompt_tokens": PROMPT, "prompt_sets": SETS, "answer_tokens": ANSWER, "temperature": TEMP, "answer_bar": LO.ANSWER_BAR,
                   "stretch": LO.STRETCH, "stretch_bar": LO.STRETCH_BAR, "rows": rows}, open(out, "w"), indent=1)
        del m
        O.free()


if __name__ == "__main__":
    (implant if sys.argv[1] == "implant" else serve)(*sys.argv[2:])
