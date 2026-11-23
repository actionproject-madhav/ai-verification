"""Red-team scoreboard: every attacked and reference model on v3 (x) and v4 (y); label = LAMBADA accuracy.
Usage: python fig_scoreboard.py <results_dir> <matplotlib style file>"""
import json, os, sys
import matplotlib.pyplot as plt
OUT, STYLE = sys.argv[1], sys.argv[2]
plt.style.use(STYLE)
INK, VERM, FOREST, MUTED, HAIR = "#1a1a1a", "#d64e2e", "#034f46", "#736f6c", "#ebebeb"
cap = {r["model"]: r for r in json.load(open(os.path.join(OUT, "capability.json")))}
r2 = [json.loads(l) for l in open(os.path.join(OUT, "round2.jsonl")) if l.strip()]
r1 = [json.loads(l) for l in open(os.path.join(OUT, "adaptive_v3_train.jsonl")) if l.strip()]
v4 = json.load(open(os.path.join(OUT, "v4.json")))
last = lambda rs, run: sorted([r for r in rs if r["run"] == run], key=lambda r: r["step"])[-1]
v4a = {r["b"].split(" ")[0]: r["attn"] for r in v4.get("attacks", [])}
pts = [("honest fine-tune", last(r1, "ft")["d_attn"], v4a["ft"], cap["honest fine-tune"]["lambada_acc"], INK, "o"),
       ("attack on v3", last(r1, "sec10")["d_attn"], v4a["sec10"], cap["attack on v3"]["lambada_acc"], VERM, "o"),
       ("attack on v4", last(r2, "v4")["v3_attn"], last(r2, "v4")["v4_attn"], cap["attack on v4"]["lambada_acc"], VERM, "o"),
       ("attack on both", last(r2, "both")["v3_attn"], last(r2, "both")["v4_attn"], cap["attack on both"]["lambada_acc"], VERM, "o")]
p3 = os.path.join(OUT, "round3.jsonl")
if os.path.exists(p3):
    r3 = [json.loads(l) for l in open(p3) if l.strip()]
    for run, lab, c in (("joint", "attack on both, broad data", VERM), ("honest", "honest fine-tune, broad data", INK)):
        if any(r["run"] == run for r in r3):
            x = last(r3, run)
            pts.append((lab, x["v3_attn"], x["v4_attn"], x["lambada"], c, "D"))
p4 = os.path.join(OUT, "round4.jsonl")
if os.path.exists(p4):                      # round 4: the attacker's best checkpoint that escapes both fingerprints
    r4 = [json.loads(l) for l in open(p4) if l.strip()]
    for run in sorted({r["run"] for r in r4}):
        rs = [r for r in r4 if r["run"] == run and r["step"] > 0]
        if not rs:
            continue
        esc = [r for r in rs if r["v3_attn"] > 0.3 and r["v4_attn"] > 0.3]
        x = max(esc, key=lambda r: r["lambada"]) if esc else last(rs, run)
        pts.append((f"distil + attack both, \u03bb={rs[0]['lam']:g}", x["v3_attn"], x["v4_attn"], x["lambada"], VERM, "^"))
ind = next(r for r in v4["smol"] if "independent" in r["b"])
pts.append(("independent model", 0.842, ind["attn"], cap["independent (SmolLM-135M v1)"]["lambada_acc"], MUTED, "s"))
fig, ax = plt.subplots(figsize=(7.4, 5.2))
ax.axvspan(0.3, 6, color=VERM, alpha=0.04, lw=0)
ax.axhspan(0.3, 3, color=VERM, alpha=0.04, lw=0)
for lab, x, y, acc, c, mk in pts:
    ax.scatter([max(x, 1e-3)], [max(y, 1e-3)], s=60, color=c, marker=mk, zorder=3)
    if lab.startswith("distil"):
        continue                                   # round 4 gets one combined label with leader lines, below
    right = x > 0.3
    off = {"attack on both": (-10, 10, "bottom", "right"), "attack on both, broad data": (-10, -8, "top", "right"),
           "honest fine-tune, broad data": (8, 2, "bottom", "left"),
           "distil + attack both, \u03bb=3": (8, -6, "top", "left"),
           "distil + attack both, \u03bb=1": (8, 4, "bottom", "left")}.get(
        lab, (-8 if right else 7, -4, "top", "right" if right else "left"))
    ax.annotate(f"{lab}\nLAMBADA {acc:.2f}", (max(x, 1e-3), max(y, 1e-3)), fontsize=7.2, color=INK,
                xytext=off[:2], textcoords="offset points", va=off[2], ha=off[3])
r4pts = [p for p in pts if p[0].startswith("distil")]
if r4pts:
    tx, ty = 1.15, 1.0
    ax.text(tx, ty, "distil + attack both\n" + "\n".join(f"{p[0].split(', ')[1]}: LAMBADA {p[3]:.2f}" for p in r4pts),
            fontsize=7.2, color=INK, ha="left", va="center")
    for p in r4pts:
        ax.annotate("", xy=(p[1], p[2]), xytext=(tx * 0.96, ty), arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=0, shrinkB=5))
ax.scatter([1e-3], [1e-3], s=60, color=FOREST, zorder=3)
ax.annotate(f"parent\nLAMBADA {cap['parent (SmolLM2-135M)']['lambada_acc']:.2f}", (1e-3, 1e-3), fontsize=7.2,
            xytext=(8, 4), textcoords="offset points", va="bottom")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(7e-4, 6); ax.set_ylim(7e-4, 3)
ax.set_xlabel("v3 distance to parent (weights)")
ax.set_ylabel("v4 distance to parent (real activations)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_scoreboard.png"), dpi=220)
print("ok", len(pts))
