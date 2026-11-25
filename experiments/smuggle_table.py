"""The table of docs/RESULTS.md, "What the room in the test is worth", written from results/smuggle/summary.json.

Usage: python smuggle_table.py [results dir]
"""
import json, os, sys

RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
ORDER = ["pythia70_by_pythia410", "pythia70_by_pythia1b", "smollm135_by_smollm2_135", "smollm135_by_smollm17",
         "smollm360_by_smollm2_360", "smollm17_by_smollm2_17", "qwen05_by_qwen15"]
summary = json.load(open(os.path.join(RES, "smuggle", "summary.json")))
s, (cap8, budget8) = summary["pairs"], summary["settings"]["bars"]["8-bit copy"]
short = lambda repo: repo.split("/")[1].replace("pythia-", "Pythia-")
pc = lambda v: f"{v:.0%}" if abs(v) >= 0.1 or v == 0 else f"{v:.1%}"       # one decimal under 10%


def share(d, spacing, bar="deployed"):
    z = d[bar][f"spacing {spacing}"]["Z-operator"]
    if z["share_of_lead"] is None or d["z_right"] - d["y_right"] < 0.03:
        return "lead under 3 points"
    a, b = z["share_95"]
    return f"{pc(z['share_of_lead'])} ({pc(a)} to {pc(b)})"


rows = ["| Y, the known model | Z, the better model | Task | Y right | Z right | Share of Z's lead that passes, each answer alone (95% interval) "
        f"| Same, answers back to back | Same, alone, at the bars an 8-bit copy needs (cap {cap8}, budget {budget8}) "
        "| Right, reference operator that knows the answers: alone, back to back |",
        "|---|---|---|---|---|---|---|---|---|"]
for k in ORDER:
    if k not in s:
        continue
    r = s[k]
    for task, name in (("lambada", "last word"), ("trivia", "closed-book question")):
        if task not in r:
            continue
        d = r[task]
        know = d["deployed"]
        rows.append(f"| {short(r['Y'])} | {short(r['Z'])} | {name} | {pc(d['y_right'])} | {pc(d['z_right'])} | {share(d, 47)} | {share(d, 0)} "
                    f"| {share(d, 47, '8-bit copy')} | {pc(know['spacing 47']['reference']['right'])}, {pc(know['spacing 0']['reference']['right'])} |")
print("\n".join(rows))
