"""Figures and summary for the v3 study. Reads results/{scale.json, adaptive_v3.jsonl, drift.json, forensics_*.json,
structural.json}; writes fig_scale.png, fig_adaptive.png, fig_drift.png, fig_forensics.png, summary_v3.json.

Usage: python analyze_v3.py <results_dir> <matplotlib style file>
"""
import glob, json, os, sys
import numpy as np
import matplotlib.pyplot as plt

OUT, STYLE = sys.argv[1], sys.argv[2]
plt.style.use(STYLE)
INK, VERM, FOREST, MUTED, HAIR, COBALT = "#1a1a1a", "#d64e2e", "#034f46", "#736f6c", "#ebebeb", "#2544a0"
PARAMS = {"pythia-14m": 14e6, "pythia-31m": 31e6, "pythia-70m": 70e6, "pythia-160m": 160e6, "pythia-410m": 410e6,
          "smol-135M": 135e6, "smol-360M": 360e6, "smol-1.7B": 1.7e9, "qwen-0.5B": 0.49e9, "qwen-1.5B": 1.54e9,
          "qwen-7B": 7.6e9, "qwen-72B": 72.7e9, "olmo-1B": 1.18e9, "olmo-7B": 6.9e9, "llama-3.1-8B": 8.0e9}
summary = {}


def load_json(name):
    p = os.path.join(OUT, name)
    return json.load(open(p)) if os.path.exists(p) else None


# ------------------------------------------------------------------ A: calibration across scale
S = load_json("scale.json")
if S:
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    rows = []
    for fam, rec in S.items():
        x = PARAMS.get(fam)
        for p in rec["pairs"]:
            if p["align"]:
                continue
            b = p["b"]
            if p["role"] == "independent":
                cat = "prev" if (b.startswith("Qwen2-") or "previous" in b) else "indep"
            elif any(k in b for k in ("Math", "Coder")) or b.startswith("R1-Distill (from Math)"):
                cat = "heavy"
            elif "step" in b:
                cat = "ckpt"
            else:
                cat = "cheap"
            rows.append((fam, x, cat, b, p["attn"]))
    rng = np.random.default_rng(0)
    jit = lambda x: x * np.exp(rng.normal(0, 0.06))
    style = {"indep": dict(s=15, color=VERM, alpha=0.5, lw=0, label="independently trained, same architecture"),
             "prev": dict(s=34, marker="s", facecolor="white", edgecolor=VERM, lw=1.3, label="previous version (Qwen2, Llama-3), relation undocumented"),
             "cheap": dict(s=24, color=INK, lw=0, label="fine-tune, RL, distillation SFT, quantised, abliterated"),
             "ckpt": dict(s=26, marker="o", facecolor="white", edgecolor=INK, lw=1.3, label="9–30% more pretraining (Pythia checkpoints)"),
             "heavy": dict(s=30, marker="D", color=FOREST, lw=0, label="continued pretraining on ≥ 1T tokens (Math, Coder, R1-Distill)")}
    for cat, st in style.items():
        pts = [r for r in rows if r[2] == cat]
        if pts:
            ax.scatter([jit(r[1]) for r in pts], [max(r[4], 1e-4) for r in pts], zorder=3, **st)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(2e-4, 2.2)
    ax.set_xlabel("parameters")
    ax.set_ylabel("attention-circuit distance to anchor")
    ax.axhspan(0.5, 2.2, color=VERM, alpha=0.04, lw=0)
    ax.legend(loc="lower left", frameon=False, fontsize=7.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_scale.png"), dpi=220)
    plt.close(fig)
    by = lambda c: [r for r in rows if r[2] == c]
    summary["scale"] = {
        "counts": {c: len(by(c)) for c in style},
        "indep_attn_range": [float(min(r[4] for r in by("indep"))), float(max(r[4] for r in by("indep")))],
        "cheap_attn_max": float(max(r[4] for r in by("cheap"))) if by("cheap") else None,
        "ckpt_attn_range": [float(min(r[4] for r in by("ckpt"))), float(max(r[4] for r in by("ckpt")))] if by("ckpt") else None,
        "heavy": {f"{r[0]} {r[3]}": round(r[4], 4) for r in by("heavy")},
        "prev": {r[0]: round(r[4], 4) for r in by("prev")},
        "by_family": {fam: {r[3]: round(r[4], 4) for r in rows if r[0] == fam} for fam in S}}

# ------------------------------------------------------------------ B: adaptive attacks (round 1) and the v4 patch
recs = []
for fn in ("adaptive_v3.jsonl", "adaptive_v3_train.jsonl"):
    p = os.path.join(OUT, fn)
    if os.path.exists(p):
        recs += [json.loads(l) for l in open(p) if l.strip()]
V4 = load_json("v4.json") or {}
if recs:
    ref = {}
    for r in recs:
        if r["run"] == "reference":
            ref.setdefault(r["what"], r)
    ce0, ind_v3 = ref["parent"]["ce"], ref["independent SmolLM-135M (v1)"]["d_attn"]
    fig = plt.figure(figsize=(10.8, 4.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1], height_ratios=[1, 1.35], wspace=0.32, hspace=0.12)
    a_ce, a_d, a_v = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])
    runs = {"ft": (INK, "ordinary fine-tune"), "sec10": (VERM, "fine-tune + attack on v3")}
    for run, (c, lab) in runs.items():
        pts = sorted([r for r in recs if r["run"] == run], key=lambda r: r["step"])
        if not pts:
            continue
        a_ce.plot([r["step"] for r in pts], [r["ce"] for r in pts], color=c, lw=1.8)
        a_d.plot([r["step"] for r in pts], [max(r["d_attn"], 1e-4) for r in pts], color=c, lw=1.8, label=lab)
    a_d.axhline(ind_v3, color=MUTED, lw=1, ls=(0, (4, 3)))
    a_d.text(8, ind_v3 * 1.25, "independent model", fontsize=7.5, color=MUTED)
    a_ce.set_ylabel("held-out loss", fontsize=8.5)
    a_ce.tick_params(labelbottom=False)
    a_d.set_yscale("log")
    a_d.set_ylim(1e-4, 2)
    a_d.set_ylabel("v3 distance", fontsize=8.5)
    a_d.set_xlabel("fine-tuning step")
    a_d.legend(frameon=False, fontsize=7.5, loc="lower right")
    v4a = {r["b"].split(" ")[0]: r["attn"] for r in V4.get("attacks", [])}
    v4s = {r["b"]: r["attn"] for r in V4.get("smol", [])}
    last = lambda run: sorted([r for r in recs if r["run"] == run], key=lambda r: r["step"])[-1]["d_attn"]
    rows_ = [("ordinary fine-tune", last("ft") if any(r["run"] == "ft" for r in recs) else None, v4a.get("ft"), INK),
             ("attacked", last("sec10") if any(r["run"] == "sec10" for r in recs) else None, v4a.get("sec10"), VERM),
             ("independent model", ind_v3, v4s.get("SmolLM-135M v1 (independent)"), MUTED)]
    for k, (lab, v3, v4, c) in enumerate(rows_):
        if v3 is None or v4 is None:
            continue
        y = 2 - k
        a_v.plot([max(v3, 1e-4), max(v4, 1e-4)], [y, y], color=HAIR, lw=2, zorder=1)
        a_v.scatter([max(v3, 1e-4)], [y], marker="o", s=46, color=c, zorder=3)
        a_v.scatter([max(v4, 1e-4)], [y], marker="D", s=40, facecolor="white", edgecolor=c, lw=1.6, zorder=3)
        a_v.text(1.6e-4, y + 0.22, lab, fontsize=8, color=INK)
    a_v.set_ylim(-0.6, 3.1)
    a_v.set_xscale("log")
    a_v.set_xlim(1e-4, 2)
    a_v.set_ylim(-0.6, 2.7)
    a_v.set_yticks([])
    a_v.set_xlabel("distance to parent")
    from matplotlib.lines import Line2D
    a_v.legend(handles=[Line2D([], [], marker="o", ls="", color=INK, ms=6, label="v3, weights only"),
                        Line2D([], [], marker="D", ls="", mfc="white", mec=INK, mew=1.4, ms=5.5, label="v4, on real text")],
               frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(0.0, 1.02))
    fig.savefig(os.path.join(OUT, "fig_adaptive.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    summary["adaptive"] = {"parent_ce": ce0, "independent_v3_attn": ind_v3,
                           "final": {run: {k: sorted([r for r in recs if r["run"] == run], key=lambda r: r["step"])[-1][k]
                                           for k in ("ce", "d", "d_attn")} for run in runs if any(r["run"] == run for r in recs)},
                           "v4": {"attacks": v4a, "smol": v4s},
                           "scramble": [{k: r[k] for k in ("kind", "frac_pool", "n_tokens", "ce", "d_attn")} for r in recs if r["run"] == "scramble"]}

# ------------------------------------------------------------------ C: drift vs training separation
D = load_json("drift.json")
if D:
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for size, c in (("70m", MUTED), ("160m", INK), ("410m", COBALT)):
        pts = sorted([r for r in D if r["size"] == size], key=lambda r: r["frac_of_run"])
        if pts:
            ax.plot([r["frac_of_run"] for r in pts], [r["attn"] for r in pts], color=c, lw=1.6, marker="o", ms=3,
                    label=f"Pythia-{size}")
    groups = {"Instruct": ([], 2e-4), "Math": ([], 1 / 18), "R1-Distill (from Math)": ([], 1 / 18 * 1.08), "Coder": ([], 5.5 / 18)}
    if S:
        for fam in ("qwen-0.5B", "qwen-1.5B", "qwen-7B", "qwen-72B"):
            for p in S.get(fam, {}).get("pairs", []):
                if not p["align"] and p["role"] == "derivative" and p["b"] in groups:
                    groups[p["b"]][0].append(p["attn"])
    for name, (ys, x) in groups.items():
        if ys:
            ax.scatter([x] * len(ys), ys, color=FOREST, s=30, zorder=4, marker="D" if "Math" in name or "R1" in name else "o")
    lab = {"Instruct": (2e-4, None, "Qwen Instruct\n0.5B, 1.5B, 7B, 72B", (6, -2)),
           "Math": (1 / 18, None, "Math and R1-Distill\n(RoPE rebased)", (-118, -6)),
           "Coder": (5.5 / 18, None, "Coder", (-34, 6))}
    for name, (x, _, text, off) in lab.items():
        ys = groups[name][0]
        if ys:
            ax.annotate(text, (x, max(ys) if name != "Instruct" else min(ys)), fontsize=7, color=FOREST,
                        xytext=off, textcoords="offset points", va="center")
    ax.axvline(1 / 3, color=VERM, lw=1.2)
    ax.text(1 / 3 * 1.07, 2.2e-3, "EU AI Act\none-third line", color=VERM, fontsize=7.5, va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(4e-4, 1.6)
    ax.set_xlabel("training separating the two models, as a share of the parent's pretraining")
    ax.set_ylabel("attention-circuit distance")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", title="checkpoints of one training run", title_fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_drift.png"), dpi=220)
    plt.close(fig)
    summary["drift"] = D

# ------------------------------------------------------------------ D: forensics
F = []
for f in sorted(glob.glob(os.path.join(OUT, "forensics_*.json"))):
    F += json.load(open(f))
if F:
    seen = {}
    for r in F:
        seen[r["model"].strip()] = r
    F = list(seen.values())
    fig, ax = plt.subplots(figsize=(7.8, 0.32 * len(F) + 1.0))
    col = {"control": INK, "ablated": VERM, "padded": COBALT}
    for i, r in enumerate(F):
        iso = r["write_gap"] / max(r["second_gap"], 1e-30)
        ax.scatter([max(iso, 1e-14)], [i], color=col[r["truth"]], s=34, marker="o" if r["truth"] != "ablated" else "x",
                   lw=1.8, zorder=3)
    ax.set_yticks(range(len(F)))
    ax.set_yticklabels([r["model"].strip(". ") for r in F], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.axvline(0.1, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.set_xlabel("isolation of the weakest residual write direction (λ_min / λ_2)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_forensics.png"), dpi=220)
    plt.close(fig)
    summary["forensics"] = F

T = load_json("structural.json")
if T:
    summary["structural"] = T
json.dump(summary, open(os.path.join(OUT, "summary_v3.json"), "w"), indent=1)
print("wrote", [k for k in summary])
