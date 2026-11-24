"""The figures in the README and the paper, drawn from the files in results/.

  overview      the headline: copies traced by the check and by an exact hash, and what each attack ends with
  separation    distance against model size: copies, unrelated models, long retrains (universal_*, watchlist)
  hashes        what exact hashes recognise against what the fingerprint recognises (hash_baseline)
  robustness    retraining aimed at the tool: leaving Red, and trying to reach Green (gpu/, round7_forge_long_135m)
  census        declared bases against the weights, on the most-downloaded Hub models (hub_scan/compare.json)
  outputs       outputs of honest copies and of stress-test models, between a listed and a known model (output_check_135m)
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


LADDER = [("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", INK, "o", "Llama-3.1-8B-Instruct"),
          ("bartowski/Qwen2.5-7B-Instruct-GGUF", FOREST, "s", "Qwen2.5-7B-Instruct"),
          ("mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated-GGUF", VERM, "^", "Llama-3.1-8B, refusals removed")]


def gguf(res, out):
    """Every GGUF file of three public models: x = bits per weight, y = distance of all parts to the safetensors
    original the file was converted from. Distances under 0.0001 are drawn at 0.0001."""
    rows = [r for r in json.load(open(os.path.join(res, "gguf_ladder.json"))) if "to_original" in r]
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    for y in (0.1, 0.5):
        ax.axhline(y, color=HAIR, lw=0.8, zorder=0)
    found = {}
    for repo, color, marker, label in LADDER:
        pts = sorted((r["bits_per_weight"], max(r["to_original"][1], 1e-4)) for r in rows if r["repo"] == repo)
        if pts:
            ax.scatter(*zip(*pts), s=14, marker=marker, color=color, linewidths=0, label=label, zorder=3)
            found[label] = (len(pts), min(p[1] for p in pts), max(p[1] for p in pts))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([3, 4, 5, 6, 8, 16, 32])
    ax.set_xticklabels(["3", "4", "5", "6", "8", "16", "32"])
    ax.minorticks_off()
    ax.set_xlim(2.8, 38)
    ax.set_ylim(6e-5, 1.3)
    ax.set_yticks([0.0001, 0.001, 0.01, 0.1, 0.5])
    ax.set_yticklabels(["\u2264 0.0001", "0.001", "0.01", "0.1", "0.5"])
    ax.set_xlabel("bits per weight")
    ax.set_ylabel("distance")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), handletextpad=0.2)
    save(fig, out, "fig_gguf")
    return found


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


def _curves(ax, series, key, threshold, band, ylabel):
    ax.axhspan(-0.04, threshold, color=band, alpha=0.10, lw=0, zorder=0)        # inside the band the light is that colour
    last = max(r["step"] for _, rows, _, _ in series for r in rows)
    for label, rows, colour, lift in series:
        if not rows:
            continue
        x, y = [r["step"] for r in rows], [key(r) for r in rows]
        ax.plot(x, y, color=colour, lw=1.4, marker="o", ms=2.4, clip_on=False)
        ax.annotate(label, (x[-1], y[-1]), xytext=(5 if not lift else 0, lift), textcoords="offset points", color=colour,
                    fontsize=8, va="center" if not lift else "bottom", ha="left" if not lift else "center")
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
    erase = [("135M", _runs(os.path.join(g, "135m-v2-exp_adaptive_light.jsonl"), "erase"), INK, 0),
             ("1.7B", _runs(os.path.join(g, "1.7b-v2-erase-exp_adaptive_light.jsonl"), "erase"), FOREST, 0)]
    long = _runs(os.path.join(res, "round7_forge_long_135m.jsonl"), "forge")
    small = long or _runs(os.path.join(g, "135m-v2-exp_adaptive_light.jsonl"), "forge")
    big = _runs(os.path.join(g, "1.7b-v2-forge-exp_adaptive_light.jsonl"), "forge")
    forge = [("135M", small, INK, 0), ("1.7B", big, FOREST, 6 if small[-1]["step"] > big[-1]["step"] else 0)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"wspace": 0.42})
    _curves(a, erase, lambda r: r["to_X"][0], 0.5, VERM, "closest part, to the listed model")
    _curves(b, forge, lambda r: r["to_Y"][1], 0.1, FOREST, "all parts, to the known model")
    save(fig, out, "fig_stress")
    last = lambda rows: None if not rows else {k: rows[-1][k] for k in ("step", "light", "to_X", "to_Y", "kl", "lambada")}
    return {"erase": {n: last(r) for n, r, _, _ in erase}, "forge": {n: last(r) for n, r, _, _ in forge}}


def tradeoff(res, out):
    """What keeping the listed model's behaviour costs on the way to Green. Each line is one retraining run that
    pulls every table of a listed model X toward a known model Y, with a different weight on keeping X's outputs.
    x = all parts to Y on the audit nonce (Green below 0.1), y = accuracy on a held-out task (LAMBADA)."""
    rows = [json.loads(l) for l in open(os.path.join(res, "round7_tradeoff_135m.jsonl")) if l.strip()]
    series = [("outputs kept, weight 1", _runs(os.path.join(res, "round7_forge_long_135m.jsonl"), "forge"), INK)]
    for keep, colour, label in ((0.1, FOREST, "weight 0.1"), (0.0, VERM, "outputs not kept")):
        series.append((label, [r for r in rows if r.get("keep") == keep], colour))
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    ax.axvspan(0, 0.1, color=FOREST, alpha=0.10, lw=0, zorder=0)                  # inside the band the light is Green
    found = {}
    for label, run, colour in series:
        x, y = [r["to_Y"][1] for r in run], [r["lambada"] for r in run]
        ax.plot(x, y, color=colour, lw=1.2, marker="o", ms=2.2, label=label)
        at = min(run, key=lambda r: abs(r["step"] - 2000))
        found[label] = {"steps": run[-1]["step"], "all_parts_to_Y": run[-1]["to_Y"][1], "kl_to_X": round(run[-1]["kl"], 3),
                        "accuracy": run[-1]["lambada"], "light": run[-1]["light"], "all_parts_to_Y_at_2000_steps": at["to_Y"][1]}
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.1, 0.5, 1.0])
    ax.set_ylim(-0.02, 0.47)
    ax.set_xlabel("all parts, to the known model")
    ax.set_ylabel("accuracy")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), handletextpad=0.4)
    save(fig, out, "fig_tradeoff")
    return found


def overview(res, out):
    """The headline figure. Left: copies among the 3,000 most-downloaded open models, by how the copy was made: the
    share the check traces to the declared parent, and the share that shares a weight file with a base (what an
    exact hash can see). Right: the light a copy of a listed model ends with after each attack, read from the
    stress-test files."""
    scan = json.load(open(os.path.join(res, "hub_scan", "compare.json")))["rows"]
    pop = {p["repo"]: p for p in json.load(open(os.path.join(res, "hub_scan", "population.json")))["population"]}
    files = {r["repo"]: r for r in json.load(open(os.path.join(res, "hub_scan", "scan_hashes.json")))["rows"]}
    kind = lambda repo: "+".join(sorted({k for b in pop[repo]["declared"] for k in b["relations"]}))
    rows = []
    for key, label in (("finetune", "fine-tuned"), ("quantized", "compressed"), ("merge", "merged")):
        pairs = [r for r in scan if r.get("claim") in ("declared base confirmed", "declared base not confirmed") and kind(r["repo"]) == key]
        seen = [files[r["repo"]] for r in pairs if r["repo"] in files]
        rows.append({"label": label, "copies": len(pairs), "check": 100 * sum(r["claim"] == "declared base confirmed" for r in pairs) / len(pairs),
                     "hash": 100 * sum(r["shared_files"] > 0 for r in seen) / len(seen), "hash_checked": len(seen)})
    last = lambda f, mode: _runs(os.path.join(res, f), mode)[-1]["light"]
    relabelled = json.load(open(os.path.join(res, "relabelled.json")))["rows"]
    lights = [("reorder or rename", {r["with_geometry"]["light"] for r in relabelled if r["relabelled"] and "listed model" in r["what"]}),
              ("retrain to evade", {last("gpu/135m-v2-exp_adaptive_light.jsonl", "erase"), last("gpu/1.7b-v2-erase-exp_adaptive_light.jsonl", "erase")}),
              ("disguise as a known model", {last("gpu/1.7b-v2-forge-exp_adaptive_light.jsonl", "forge"), last("round7_forge_long_135m.jsonl", "forge")})]
    assert all(len(v) == 1 for _, v in lights), lights                # every run of an attack ends on the same light

    fig = plt.figure(figsize=(7.4, 2.3))
    ax, bx = fig.add_axes([0.125, 0.21, 0.34, 0.64]), fig.add_axes([0.775, 0.21, 0.2, 0.64])
    h = 0.26
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y + h / 2 + 0.02, r["check"], height=h, color=INK, lw=0, label="this check" if i == 0 else None)
        ax.barh(y - h / 2 - 0.02, r["hash"], height=h, color=SOFT, lw=0, label="exact hash" if i == 0 else None)
        ax.text(r["check"] + 1.5, y + h / 2 + 0.02, f"{r['check']:.0f}%", va="center", ha="left", fontsize=8.5)
        ax.text(r["hash"] + 1.5, y - h / 2 - 0.02, f"{r['hash']:.0f}%", va="center", ha="left", fontsize=8.5, color=MUTED)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["label"] for r in reversed(rows)], fontsize=9)
    ax.tick_params(axis="y", length=0, pad=6)
    for i, r in enumerate(rows):                                # how many copies each row stands for
        ax.annotate(f"{r['copies']:,}", (0, len(rows) - 1 - i), xytext=(-6, -11), textcoords="offset points", ha="right", va="center",
                    fontsize=7, color=SOFT)
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 50, 100])
    ax.set_ylim(-0.55, len(rows) - 0.25)
    ax.set_xlabel("copies traced to their parent (%)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, handlelength=1.0, handleheight=0.7, columnspacing=1.4, borderaxespad=0.2)

    colour = {"red": VERM, "amber": SUN, "green": FOREST}
    for i, (label, light) in enumerate(lights):
        y = len(lights) - 1 - i
        for j, name in enumerate(("red", "amber", "green")):
            on = name in light
            bx.scatter([j], [y], s=150, facecolor=colour[name] if on else "white", edgecolor="none" if on else "#d2cfcb", lw=0.9, zorder=3)
    bx.set_yticks(range(len(lights)))
    bx.set_yticklabels([label for label, _ in reversed(lights)], fontsize=9)
    bx.tick_params(axis="y", length=0, pad=8)
    bx.set_xticks(range(3))
    bx.set_xticklabels(["Red", "Amber", "Green"], fontsize=8.5)
    bx.xaxis.tick_top()
    bx.tick_params(axis="x", length=0, pad=4)
    bx.set_xlim(-0.6, 2.6)
    bx.set_ylim(-0.55, len(lights) - 0.25)
    for side in ("left", "bottom", "top", "right"):
        bx.spines[side].set_visible(False)
    bx.set_xlabel("light after the attack")
    save(fig, out, "fig_overview")
    return {"rows": [{k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()} for r in rows],
            "lights": {label: sorted(v)[0] for label, v in lights}}


def outputs(res, out):
    """Where each model's outputs sit between the listed model X and the known model Y, in total variation between
    next-token distributions on held-out text. No model can sit below the diagonal: TV(m, X) + TV(m, Y) >= TV(X, Y).
    Models further than 0.6 from both are left off the axes; the table in docs/RESULTS.md lists them."""
    d = json.load(open(os.path.join(res, "output_check_135m.json")))
    gap, rows = d["tv_X_Y"], d["rows"]
    kind = lambda n: "pair" if n.startswith(("X, the listed", "Y, the known")) else "Y" if n.startswith(("Y,", "Y's", "Y trained")) else "X"
    fig, ax = plt.subplots(figsize=(3.5, 3.3))
    top = 0.45
    ax.add_patch(Polygon([(0, 0), (gap, 0), (0, gap)], closed=True, fc=HAIR, ec="none", zorder=0, label="ruled out"))
    ax.plot([0, gap], [gap, 0], color=SOFT, lw=0.8, zorder=1)
    shown = [r for r in rows if max(r["tv_to_Y"], r["tv_to_X"]) <= 0.6]
    for key, colour, label in (("Y", FOREST, "copies of Y"), ("X", VERM, "made from X")):
        pts = [r for r in shown if kind(r["model"]) == key]
        ax.scatter([r["tv_to_Y"] for r in pts], [r["tv_to_X"] for r in pts], s=22, color=colour, lw=0, zorder=3, label=label, clip_on=False)
    ax.scatter([0, gap], [gap, 0], s=26, color=INK, lw=0, zorder=4, clip_on=False)
    ax.annotate("Y", (0, gap), xytext=(6, 5), textcoords="offset points", fontsize=9, fontweight="medium")
    ax.annotate("X", (gap, 0), xytext=(5, 6), textcoords="offset points", fontsize=9, fontweight="medium")
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_aspect("equal")
    ticks = [0, 0.1, 0.2, 0.3, 0.4]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel("outputs, to the known model Y")
    ax.set_ylabel("outputs, to the listed model X")
    ax.legend(loc="upper right", handletextpad=0.3, borderaxespad=0.2)
    save(fig, out, "fig_outputs")
    return {"tv_X_Y": gap, "shown": len(shown), "left_off": [r["model"] for r in rows if r not in shown],
            "copies_of_Y_furthest_from_Y": max(r["tv_to_Y"] for r in shown if kind(r["model"]) == "Y"),
            "made_from_X_closest_to_Y": min(r["tv_to_Y"] for r in shown if kind(r["model"]) == "X")}


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


def _bars(ax, x, y, w, h, seed, color, n=7):
    """A next-token distribution, as a few bars."""
    v = np.sort(np.random.default_rng(seed).random(n) ** 2.2)[::-1]
    v = v / v.max()
    step = w / n
    for k in range(n):
        ax.add_patch(Rectangle((x + k * step, y), step * 0.62, h * (0.1 + 0.9 * v[k]), fc=color, ec="none", zorder=3))
    ax.plot([x - 0.03, x + w], [y, y], color=color, lw=0.7, zorder=3)


def architecture(res, out):
    """The mechanism in one picture. A nonce from the verifier fixes the tokens, layers and text. Inside the
    enclave the model is read twice: its weight tables, and its outputs. Three tests in a column decide the light."""
    fig, ax = plt.subplots(figsize=(11.6, 4.75))
    ax.set_xlim(0, 11.6); ax.set_ylim(0, 4.75); ax.axis("off")
    small = dict(fontsize=7.5, color=MUTED, ha="center", va="center")
    lane = dict(fontsize=6.6, color=MUTED, va="center", fontweight="semibold")
    rule = dict(fontsize=8, ha="center", va="center")
    RED_Y, AMBER_Y, GREEN_Y = 3.15, 2.235, 1.32                  # one height per light
    K_Y, OUT_Y = 1.72, 0.92                                      # Green needs two tests: weights, then outputs
    AMBER = "#b8741c"
    # operator: the model being loaded
    ax.text(0.78, 4.45, "OPERATOR", ha="center", **lane)
    _layers(ax, 0.38, 2.0, 0.8)
    ax.text(0.78, 1.74, r"weights $M$", **small)
    _arrow(ax, 1.34, 2.3, 1.98, 2.3)
    # the enclave
    _box(ax, 2.08, 0.26, 5.75, 3.78, lw=1.5, r=0.12)
    ax.add_patch(Rectangle((6.2, 3.66), 0.2, 0.2, fc="white", ec=INK, lw=1.0, zorder=4))
    for k in range(3):
        ax.plot([6.235 + k * 0.065] * 2, [3.86, 3.92], color=INK, lw=0.9, zorder=4)
        ax.plot([6.235 + k * 0.065] * 2, [3.6, 3.66], color=INK, lw=0.9, zorder=4)
    ax.text(6.51, 3.76, "SEALED ENCLAVE", ha="left", **lane)
    # the challenge: a nonce from the verifier fixes the tokens, the layers and the text, after the weights are fixed
    ax.text(9.09, 4.45, "VERIFIER", ha="center", **lane)
    ax.plot([8.62, 2.85, 2.85], [4.45, 4.45, 3.56], color=MUTED, lw=0.9, zorder=3, solid_joinstyle="round")
    _arrow(ax, 2.85, 3.58, 2.85, 3.5, color=MUTED, lw=0.9)
    ax.text(5.75, 4.58, r"fresh nonce $n$", **small)
    ax.text(2.85, 3.36, r"$n \mapsto S,\ \ell,\ x$", fontsize=7.5, color=MUTED, ha="center", va="center")
    # the model, read twice: its weight tables and its outputs
    _table(ax, 2.48, 2.3, 0.74, seed=1, color=INK)
    ax.text(2.85, 2.1, r"$T = E_S\,\hat W_\ell^{\top}\hat W_\ell\,E_S^{\top}$", fontsize=7, color=INK, ha="center", va="center")
    ax.text(2.85, 1.87, r"$d = 1 - \mathrm{corr}(T, T')$", fontsize=7, color=MUTED, ha="center", va="center")
    _bars(ax, 2.55, OUT_Y - 0.2, 0.6, 0.44, seed=3, color=INK)
    ax.text(2.85, OUT_Y - 0.4, r"$p_M(\cdot \mid x)$", fontsize=7, color=INK, ha="center", va="center")
    # Red: one part matches a listed model
    _stack(ax, 4.2, RED_Y - 0.25, 0.5, VERM, 4)
    ax.text(3.92, RED_Y + 0.09, "W", fontsize=11, color=VERM, ha="center", va="center", fontweight="semibold")
    ax.text(3.92, RED_Y - 0.13, "listed", fontsize=6.6, color=MUTED, ha="center", va="center")
    ax.text(5.95, RED_Y, r"$\min_{\mathrm{part}}\; d(M, W) \,<\, 0.5$", color=VERM, **rule)
    # Green: every part matches a known model, nothing cut out, and the outputs agree
    _stack(ax, 4.2, K_Y - 0.25, 0.5, FOREST, 9)
    ax.text(3.92, K_Y + 0.09, "K", fontsize=11, color=FOREST, ha="center", va="center", fontweight="semibold")
    ax.text(3.92, K_Y - 0.13, "known", fontsize=6.6, color=MUTED, ha="center", va="center")
    ax.text(5.95, K_Y + 0.11, r"$d(M, K) \,<\, 0.1$ on every part", color=FOREST, **rule)
    ax.text(5.95, K_Y - 0.11, "no direction cut out", color=FOREST, **rule)
    _bars(ax, 4.2, OUT_Y - 0.2, 0.52, 0.42, seed=3, color=FOREST)
    ax.text(4.46, OUT_Y - 0.4, r"$p_K(\cdot \mid x)$", fontsize=7, color=FOREST, ha="center", va="center")
    ax.text(5.95, OUT_Y, r"$\mathrm{TV}(p_M, p_K) \,\leq\, 0.1$", color=FOREST, **rule)
    # what feeds what
    _arrow(ax, 3.32, 2.9, 3.72, RED_Y - 0.03, color=MUTED, lw=0.8)
    _arrow(ax, 3.32, 2.45, 3.72, K_Y + 0.16, color=MUTED, lw=0.8)
    _arrow(ax, 3.28, OUT_Y + 0.02, 4.06, OUT_Y + 0.02, color=MUTED, lw=0.8)
    # the three ways out: Red on one match, Green on both tests, Amber otherwise
    _arrow(ax, 7.0, RED_Y, 8.76, RED_Y, color=VERM)
    ax.plot([6.98, 7.12, 7.12, 6.98], [K_Y, K_Y, OUT_Y, OUT_Y], color=FOREST, lw=1.0, zorder=3, solid_joinstyle="round")   # both
    _arrow(ax, 7.12, GREEN_Y, 8.76, GREEN_Y, color=FOREST)
    _arrow(ax, 7.45, AMBER_Y, 8.76, AMBER_Y, color=SUN)
    ax.text(7.05, AMBER_Y, "otherwise", fontsize=7.5, color=AMBER, ha="center", va="center")
    ax.text(8.3, RED_Y + 0.35, "signed", fontsize=7, color=MUTED, ha="center", va="center")
    ax.text(8.3, RED_Y + 0.19, "attested", fontsize=7, color=MUTED, ha="center", va="center")
    _box(ax, 8.8, GREEN_Y - 0.43, 0.58, RED_Y - GREEN_Y + 0.86, lw=1.2, r=0.27)
    for yy, col in ((RED_Y, VERM), (AMBER_Y, SUN), (GREEN_Y, FOREST)):
        ax.add_patch(Circle((9.09, yy), 0.19, fc=col, ec="none", zorder=4))
    for yy, col, text in ((RED_Y, VERM, "identity + content checks"), (AMBER_Y, AMBER, "identity check"), (GREEN_Y, FOREST, "runs")):
        ax.plot([9.38, 9.56], [yy, yy], color=col, lw=1.0, zorder=3)
        ax.text(9.64, yy, text, fontsize=8, color=col, ha="left", va="center", fontweight="medium")
    save(fig, out, "fig_architecture")


if __name__ == "__main__":
    res, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    todo = sys.argv[3:] or ["separation", "hashes", "architecture"]
    for name in todo:
        print(name, globals()[name](res, out) or "")
