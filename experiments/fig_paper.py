"""The figures in the README and the paper, drawn from the files in results/.

  separation    distance against model size: copies, unrelated models, long retrains (universal_*, watchlist)
  hashes        what exact hashes recognise against what the fingerprint recognises (hash_baseline)
  robustness    retraining aimed at the tool: leaving Red, and trying to reach Green (gpu/, round7_forge_long_135m)
  census        declared bases against the weights, on the most-downloaded Hub models (hub_scan/compare.json)
  architecture  the mechanism as a diagram

Usage: python fig_paper.py <results dir> <out dir> [figure ...]
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

INK, FOREST, VERM, SUN, MUTED, SOFT, HAIR = "#1a1a1a", "#034f46", "#d64e2e", "#ffa946", "#736f6c", "#94918e", "#ebebeb"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Geist", "DM Sans", "Helvetica Neue", "Arial"], "font.size": 9,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK, "axes.linewidth": 0.8, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "axes.axisbelow": True,
    "xtick.color": "#47423d", "ytick.color": "#47423d", "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.frameon": False, "legend.fontsize": 8, "figure.facecolor": "white", "savefig.facecolor": "white",
    "savefig.dpi": 300, "savefig.bbox": "tight", "svg.fonttype": "path"})
MONO = {"family": "JetBrains Mono"}
# Parameter counts from each model's safetensors metadata on the Hub (read 23 Nov 2026).
SIZES = {"EleutherAI/pythia-1.4b": 1.52e9, "HuggingFaceTB/SmolLM-1.7B": 1.71e9, "HuggingFaceTB/SmolLM-135M": 135e6,
         "HuggingFaceTB/SmolLM-360M": 362e6, "HuggingFaceTB/SmolLM2-1.7B": 1.71e9, "HuggingFaceTB/SmolLM2-135M": 135e6,
         "HuggingFaceTB/SmolLM2-360M": 362e6, "Qwen/Qwen2.5-1.5B": 1.54e9, "Qwen/Qwen2.5-7B": 7.62e9,
         "Qwen/Qwen3-8B-Base": 8.19e9, "TinyLlama/TinyLlama-1.1B-Chat-v1.0": 1.10e9, "allenai/OLMo-2-1124-7B": 7.30e9,
         "deepseek-ai/DeepSeek-V4-Flash": 291e9, "marin-community/marin-8b-base": 8.03e9,
         "mistralai/Mistral-7B-v0.3": 7.25e9, "unsloth/Llama-3.1-8B": 8.03e9, "zai-org/GLM-5.3-Flash": 321e9}
LISTED = {"glm-5.2": 753e9, "glm-5.3": 753e9, "kimi-k3": 2.78e12}


def save(fig, out, name):
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)


def separation(res, out):
    """Every pair measured with the shipped fingerprint: x = parameters, y = distance of the closest part.
    Unrelated pairs sit at the geometric mean of their two sizes."""
    load = lambda f: json.load(open(os.path.join(res, f)))
    pts = {"copy": [], "unrelated": [], "long": []}
    for r in load("universal_cross.json")["rows"]:
        if r.get("closest_part") is not None:
            pts["unrelated"].append(((SIZES[r["a"]] * SIZES[r["b"]]) ** 0.5, r["closest_part"]))
    for r in load("watchlist.json")["rows"]:
        d = {k: v for k, v in (r.get("distances") or {}).items() if v is not None}
        if not d or r.get("relation") == "listed pair":
            continue
        best = min(d, key=d.get)
        pts["copy" if r.get("matched") else "unrelated"].append((LISTED[best], max(d[best], 0.0)))
    for r in load("universal.json"):
        if r["kind"] == "derivative":
            pts["copy"].append((SIZES.get(r["anchor"], 7.62e9), r["d"]))
    for r in load("universal_pairs.json")["rows"]:
        if r.get("closest_part") is not None:
            pts["long" if r["kind"] == "retrained at length" else "copy"].append((r["params"], max(r["closest_part"], 0.0)))
    rng = np.random.default_rng(3)
    jit = lambda xs: np.asarray(xs) * np.exp(rng.normal(0, 0.09, len(xs)))
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    for y in (0.1, 0.5):
        ax.axhline(y, color=HAIR, lw=0.8, zorder=0)
    x, y = zip(*pts["unrelated"]); ax.scatter(jit(x), y, s=11, facecolors="none", edgecolors=INK, linewidths=0.6, label="unrelated")
    x, y = zip(*pts["long"]); ax.scatter(jit(x), y, s=13, marker="x", color=SOFT, linewidths=0.9, label="retrained at length")
    x, y = zip(*pts["copy"]); ax.scatter(jit(x), y, s=15, color=FOREST, linewidths=0, label="modified copy", zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(7e7, 6e12)
    ax.set_ylim(-0.04, 1.08)
    ax.set_yticks([0, 0.1, 0.5, 1.0])
    ax.set_xticks([1e8, 1e9, 1e10, 1e11, 1e12])
    ax.set_xticklabels(["100M", "1B", "10B", "100B", "1T"])
    ax.set_xlabel("parameters")
    ax.set_ylabel("distance")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), handletextpad=0.2)
    save(fig, out, "fig_separation")
    return {k: (len(v), round(min(y for _, y in v), 3), round(max(y for _, y in v), 3)) for k, v in pts.items()}


GROUPS = [("repackaged", ("4-bit release", "8-bit release", "8-bit release, renamed", "16-bit release")),
          ("new version", ("later version",)), ("fine-tuned", ("instruct version", "fine-tune")),
          ("refusals\nremoved", ("refusals removed",)), ("unrelated", ("unrelated", "unrelated, similar name"))]


def hashes(res, out):
    """One column per public model, one row per method. A filled dot: the method says "made from the base model"."""
    rows = [r for r in json.load(open(os.path.join(res, "hash_baseline.json"))) if "error" not in r]
    methods = [("file hash", lambda r: r["files_identical"] > 0), ("tensor hash", lambda r: r["tensors_identical"] > 0),
               ("fingerprint", lambda r: r["closest_part"] is not None and r["closest_part"] < 0.5)]
    fig, ax = plt.subplots(figsize=(7.2, 2.0))
    x, ticks, n_copy = 0.0, [], 0
    for name, kinds in GROUPS:
        members = [r for r in rows if r["kind"] in kinds]
        if not members:
            continue
        x0 = x
        for r in members:
            copy = name != "unrelated"
            n_copy += copy
            for j, (_, hit) in enumerate(methods):
                yes = hit(r)
                colour = (FOREST if copy else VERM) if yes else "white"
                ax.scatter([x], [2 - j], s=62, facecolors=colour, edgecolors=FOREST if (copy and yes) else VERM if yes else SOFT, linewidths=0.9, zorder=3)
            x += 1
        ticks.append(((x0 + x - 1) / 2, name))
        ax.plot([x0 - 0.3, x - 0.7], [-0.72, -0.72], color=INK, lw=0.8, clip_on=False)
        x += 1.5
    copies = [r for r in rows if not r["kind"].startswith("unrelated")]
    for j, (m, hit) in enumerate(methods):
        ax.text(-1.0, 2 - j, m, ha="right", va="center", fontsize=9)
        ax.text(x - 0.9, 2 - j, f"{sum(hit(r) for r in copies)} / {len(copies)}", ha="left", va="center", fontsize=9,
                color=FOREST if j == 2 else INK, fontweight="semibold" if j == 2 else "normal", **MONO)
    for xc, name in ticks:
        ax.text(xc, -1.0, name, ha="center", va="top", fontsize=7.6, color=MUTED, linespacing=1.15)
    ax.set_xlim(-1.2, x + 1.0); ax.set_ylim(-2.0, 2.6); ax.axis("off")
    save(fig, out, "fig_hashes")
    return {m: sum(hit(r) for r in copies) for m, hit in methods} | {"copies": len(copies), "false alarms": {m: sum(hit(r) for r in rows if r["kind"].startswith("unrelated")) for m, hit in methods}}


def _runs(path, mode):
    """The rows of one mode in a result file, after the last restart (a preempted run starts again from step 0)."""
    if not os.path.exists(path):
        return []
    rows = [r for r in map(json.loads, filter(str.strip, open(path))) if r.get("mode") == mode]
    cut = max([i for i in range(1, len(rows)) if rows[i]["step"] < rows[i - 1]["step"]], default=0)
    return rows[cut:]


def _curves(ax, series, key, threshold, ylabel):
    ax.axhline(threshold, color=HAIR, lw=0.8, zorder=0)
    last = max(r["step"] for _, rows, _ in series for r in rows)
    for label, rows, colour in series:
        if not rows:
            continue
        x, y = [r["step"] for r in rows], [key(r) for r in rows]
        ax.plot(x, y, color=colour, lw=1.4, marker="o", ms=2.4, clip_on=False)
        ax.annotate(label, (x[-1], y[-1]), xytext=(5, 0), textcoords="offset points", color=colour, fontsize=8, va="center")
    ax.set_xlim(0, last)
    ax.set_ylim(-0.04, 1.06)
    ax.set_yticks(sorted({0, threshold, 0.5, 1.0}))
    ax.set_xlabel("training steps")
    ax.set_ylabel(ylabel)


def robustness(res, out):
    """Round 7: retraining aimed at the tool itself. Left: a copy of a listed model leaving Red (closest part to
    the listed model, Red below 0.5). Right: the same copy trying to pass as a known model (all parts to the known
    model, Green below 0.1)."""
    g = os.path.join(res, "gpu")
    erase = [("135M", _runs(os.path.join(g, "135m-v2-exp_adaptive_light.jsonl"), "erase"), INK),
             ("1.7B", _runs(os.path.join(g, "1.7b-v2-erase-exp_adaptive_light.jsonl"), "erase"), FOREST)]
    long = _runs(os.path.join(res, "round7_forge_long_135m.jsonl"), "forge")
    forge = [("135M", long or _runs(os.path.join(g, "135m-v2-exp_adaptive_light.jsonl"), "forge"), INK),
             ("1.7B", _runs(os.path.join(g, "1.7b-v2-forge-exp_adaptive_light.jsonl"), "forge"), FOREST)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"wspace": 0.42})
    _curves(a, erase, lambda r: r["to_X"][0], 0.5, "closest part, to the listed model")
    _curves(b, forge, lambda r: r["to_Y"][1], 0.1, "all parts, to the known model")
    save(fig, out, "fig_stress")
    last = lambda rows: None if not rows else {k: rows[-1][k] for k in ("step", "light", "to_X", "to_Y", "kl", "lambada")}
    return {"erase": {n: last(r) for n, r, _ in erase}, "forge": {n: last(r) for n, r, _ in forge}}


def census(res, out):
    """The 3,000 most-downloaded text models on the Hub, by what their card declares and what the weights say."""
    rows = [r for r in json.load(open(os.path.join(res, "hub_scan", "compare.json")))["rows"] if r.get("status") != "unreadable"]
    n = lambda claim, lights: sum(r["claim"] == claim and r["light"] in lights for r in rows)
    copy, unknown = ("close copy", "modified copy"), ("unknown origin",)
    bars = [("declares a base", [(sum(r["claim"] == "declared base confirmed" for r in rows), FOREST, "confirmed"),
                                 (sum(r["claim"] == "declared base not confirmed" for r in rows), VERM, "not confirmed")]),
            ("declares none", [(n("no base declared", copy), FOREST, "traced to a base"),
                               (n("no base declared", unknown), HAIR, "no base found")])]
    fig, ax = plt.subplots(figsize=(6.6, 1.5))
    for y, (name, parts) in zip((1, 0), bars):
        x = 0
        for k, (v, colour, what) in enumerate(parts):
            ax.barh(y, v, left=x, height=0.46, color=colour, linewidth=0)
            inside = v > 150
            ax.text(x + v / 2 if inside else x + v + 14, y, f"{v:,}", ha="center" if inside else "left", va="center", fontsize=8.5,
                    color="white" if inside and colour != HAIR else INK, **MONO)
            ax.text(x + v / 2 if inside else x + v + 14, y - 0.36, what, ha="center" if inside else "left", va="top", fontsize=7.4, color=MUTED)
            x += v
        ax.text(-22, y, name, ha="right", va="center", fontsize=9)
    ax.set_xlim(-10, 1260); ax.set_ylim(-0.75, 1.45); ax.axis("off")
    save(fig, out, "fig_census")
    return {name: [(what, v) for v, _, what in parts] for name, parts in bars}


def _box(ax, x, y, w, h, lw=1.2, fc="white", ec=INK, r=0.06, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", lw=lw, fc=fc, ec=ec, zorder=z))


def _arrow(ax, x0, y0, x1, y1, color=INK, lw=1.1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=5,
                arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.22", color=color, lw=lw, shrinkA=0, shrinkB=0))


def _table(ax, x, y, s, n=6, seed=0, color=FOREST, z=3):
    """A token-by-token table: the fingerprint of one matrix."""
    v = np.random.default_rng(seed).random((n, n)) ** 1.6
    c = s / n
    for i in range(n):
        for j in range(n):
            ax.add_patch(Rectangle((x + j * c, y + i * c), c * 0.92, c * 0.92, fc=color, ec="none", alpha=0.12 + 0.88 * v[i, j], zorder=z))


def _stack(ax, x, y, s, color, seed, n=5):
    """A list of reference fingerprints: one table in front, the rest as cards behind it."""
    for k in (2, 1):
        ax.add_patch(Rectangle((x + k * 0.075, y + k * 0.075), s, s, fc="white", ec=color, lw=0.7, alpha=0.55, zorder=3 - k * 0.1))
    ax.add_patch(Rectangle((x - 0.012, y - 0.012), s + 0.024, s + 0.024, fc="white", ec="none", zorder=3.2))
    _table(ax, x, y, s, n=n, seed=seed, color=color, z=3.4)


def _layers(ax, x, y, w, n=5, gap=0.13, h=0.075, color=INK):
    for i in range(n):
        ax.add_patch(FancyBboxPatch((x, y + i * gap), w, h, boxstyle="round,pad=0,rounding_size=0.03", fc=color, ec="none", zorder=3))


def architecture(res, out):
    fig, ax = plt.subplots(figsize=(10.6, 4.3))
    ax.set_xlim(0, 10.6); ax.set_ylim(0, 4.3); ax.axis("off")
    small = dict(fontsize=7.5, color=MUTED, ha="center", va="center")
    lane = dict(fontsize=6.6, color=MUTED, va="center", fontweight="semibold")
    # operator: the model being loaded
    ax.text(0.76, 3.86, "OPERATOR", ha="center", **lane)
    _layers(ax, 0.36, 1.72, 0.8)
    ax.text(0.76, 1.46, r"weights $M$", **small)
    _arrow(ax, 1.32, 2.02, 1.96, 2.02)
    # the enclave
    _box(ax, 2.05, 0.55, 4.8, 3.0, lw=1.5, r=0.12)
    ax.add_patch(Rectangle((2.23, 3.17), 0.2, 0.2, fc="white", ec=INK, lw=1.0, zorder=4))
    for k in range(3):
        ax.plot([2.265 + k * 0.065] * 2, [3.37, 3.43], color=INK, lw=0.9, zorder=4)
        ax.plot([2.265 + k * 0.065] * 2, [3.11, 3.17], color=INK, lw=0.9, zorder=4)
    ax.text(2.54, 3.27, "SEALED ENCLAVE", ha="left", **lane)
    # the challenge: a nonce from the verifier picks the tokens and the layers
    ax.plot([7.5, 3.72, 3.72], [3.86, 3.86, 2.92], color=MUTED, lw=0.9, zorder=3, solid_joinstyle="round")
    _arrow(ax, 3.72, 2.94, 3.72, 2.86, color=MUTED, lw=0.9)
    ax.text(5.86, 3.99, r"fresh nonce $n$", fontsize=7.5, color=MUTED, ha="center", va="center")
    ax.text(3.72, 2.72, r"tokens $S$, layers", fontsize=7, color=MUTED, ha="center", va="center")
    # the table of one matrix of the loaded model
    _table(ax, 2.38, 1.5, 0.8, seed=1, color=INK)
    ax.text(2.78, 1.24, r"$T = E_S\,\hat W^{\top}\hat W\,E_S^{\top}$", fontsize=7, color=INK, ha="center", va="center")
    ax.text(2.78, 0.98, "one per weight matrix", fontsize=6.4, color=MUTED, ha="center", va="center")
    _arrow(ax, 3.5, 2.6, 3.24, 2.4, color=MUTED, lw=0.8)
    # the two lists
    for yy, name, word, col, seed, rule in ((2.02, "W", "listed", VERM, 4, r"closest part  $d < 0.5$"),
                                            (0.9, "K", "known", FOREST, 9, r"every part  $d < 0.1$")):
        _stack(ax, 4.5, yy, 0.5, col, seed)
        ax.text(4.2, yy + 0.34, name, fontsize=11, color=col, ha="center", va="center", fontweight="semibold")
        ax.text(4.2, yy + 0.12, word, fontsize=6.6, color=MUTED, ha="center", va="center")
        ax.text(5.98, yy + 0.3, rule, fontsize=7.6, color=col, ha="center", va="center")
    _arrow(ax, 3.3, 2.0, 3.86, 2.24, color=MUTED, lw=0.8)
    _arrow(ax, 3.3, 1.72, 3.86, 1.34, color=MUTED, lw=0.8)
    # one signed, attested answer leaves
    _arrow(ax, 6.85, 2.02, 7.62, 2.02)
    ax.text(7.24, 2.2, "signed", fontsize=7, color=MUTED, ha="center", va="center")
    ax.text(7.24, 1.85, "attested", fontsize=7, color=MUTED, ha="center", va="center")
    ax.text(8.0, 3.86, "VERIFIER", ha="center", **lane)
    _box(ax, 7.75, 1.14, 0.5, 1.76, lw=1.2, r=0.25)
    for yy, col in ((2.58, VERM), (2.02, SUN), (1.46, FOREST)):
        ax.add_patch(Circle((8.0, yy), 0.17, fc=col, ec="none", zorder=4))
    for yy, col, text in ((2.58, VERM, "identity + content checks"), (2.02, "#b8741c", "identity check"), (1.46, FOREST, "runs")):
        ax.plot([8.25, 8.45], [yy, yy], color=col, lw=1.0, zorder=3)
        ax.text(8.53, yy, text, fontsize=8, color=col, ha="left", va="center", fontweight="medium")
    save(fig, out, "fig_architecture")


if __name__ == "__main__":
    res, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    todo = sys.argv[3:] or ["separation", "hashes", "architecture"]
    for name in todo:
        print(name, globals()[name](res, out) or "")
