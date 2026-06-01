"""Analyse the DeepSeek-reasoner arm on the +100 high-d_c extension and pool the right tail.

(1) §4.8 extension table, apples-to-apples: reasoner added as a 4th solver next to DeepSeek /
    Kimi-K2 / Qwen-14B, same extension d_c labels (from highdc_multimodel_items).
(2) Pooled reasoner right tail: pilot258 reasoner (254) + extension reasoner (100) per d_c, with
    a univariate OR. The two halves use DIFFERENT d_c label sources (pilot = 4-family LLM
    consensus; extension = single-rater DeepSeek prelabel), so the pooled curve is reported as a
    coverage/robustness extension, not a single clean panel — this caveat is printed and figured.

Figure F18 (2 panels): (a) extension 4-solver per-d_c; (b) pooled reasoner per-d_c, right tail.
Report: evaluation/reasoner_highdc_extension_20260529.md
"""
from __future__ import annotations
import csv
import importlib.util
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT / "code" / "scripts"
EXT_REASONER = PROJECT / "data" / "results" / "solve_highdc100_deepseek_reasoner_judged_20260529.csv"
EXT_ITEMS = PROJECT / "evaluation" / "highdc_multimodel_items_20260526.csv"
PILOT_REASONER = PROJECT / "data" / "results" / "solve_pilot258_deepseek_reasoner_judged_20260529.csv"
PILOT_PANEL = PROJECT / "evaluation" / "w6_controlled_panel_with_logs.csv"
OUT_MD = PROJECT / "evaluation" / "reasoner_highdc_extension_20260529.md"
OUT_FIG = PROJECT / "figures" / "F18_reasoner_highdc_extension.png"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod


m07 = load_module(SCRIPTS / "07_w6_controlled_dc_analysis.py", "w6_07")


def read_csv(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def per_dc(pairs):
    d = defaultdict(list)
    for dc, c in pairs:
        d[dc].append(c)
    return {k: (len(v), float(np.mean(v))) for k, v in sorted(d.items())}


def univ_or(pairs):
    if len({p[0] for p in pairs}) < 2:
        return float("nan"), float("nan"), float("nan")
    dc = np.array([p[0] for p in pairs], float); y = np.array([p[1] for p in pairs], float)
    X = np.column_stack([np.ones(len(dc)), dc])
    fit = m07.fit_logit(X, y, ["intercept", "d_c"], 1.0)
    b = float(fit["coef"][1]); se = float(fit["se"][1])
    z = b / se if se > 0 else float("nan")
    return b, math.exp(b), (math.erfc(abs(z) / math.sqrt(2)) if not math.isnan(z) else float("nan"))


def main():
    # ---- extension d_c + 3 existing solvers ----
    ext_dc = {}
    solver_pairs = defaultdict(list)  # model_label -> [(dc, correct)]
    NICE = {"DeepSeek": "DeepSeek-chat", "Kimi-K2": "Kimi-K2", "Qwen-14B": "Qwen-14B"}
    for r in read_csv(EXT_ITEMS):
        if str(r.get("valid", "1")) not in {"1", "True", "true"}:
            continue
        try:
            dc = int(float(r["d_c"]))
        except Exception:
            continue
        ext_dc.setdefault(r["item_id"], dc)
        v = str(r.get("is_correct", "")).strip()
        if v in {"0", "1"}:
            solver_pairs[r["model_label"]].append((dc, int(v)))

    # ---- extension reasoner ----
    reas_ext = []
    for r in read_csv(EXT_REASONER):
        v = str(r.get("is_correct_judge", "")).strip()
        iid = r["item_id"]
        if v in {"0", "1"} and iid in ext_dc:
            reas_ext.append((ext_dc[iid], int(v)))
    solver_pairs["DeepSeek-reasoner"] = reas_ext

    # ---- pilot258 reasoner ----
    pilot_dc = {r["item_id"]: int(float(r["d_c"])) for r in read_csv(PILOT_PANEL)}
    reas_pilot = []
    for r in read_csv(PILOT_REASONER):
        v = str(r.get("is_correct_judge", "")).strip()
        if v in {"0", "1"} and r["item_id"] in pilot_dc:
            reas_pilot.append((pilot_dc[r["item_id"]], int(v)))

    pooled = reas_pilot + reas_ext

    # ---- stats ----
    def overall(pairs):
        return (len(pairs), float(np.mean([c for _, c in pairs])) if pairs else float("nan"))
    ext_tbl = {m: (overall(solver_pairs[m]), per_dc(solver_pairs[m]))
               for m in ["DeepSeek", "Kimi-K2", "Qwen-14B", "DeepSeek-reasoner"] if solver_pairs.get(m)}
    b_p, or_p, p_p = univ_or(reas_pilot)
    b_x, or_x, p_x = univ_or(reas_ext)
    b_a, or_a, p_a = univ_or(pooled)
    pd_pilot, pd_ext, pd_pool = per_dc(reas_pilot), per_dc(reas_ext), per_dc(pooled)

    # ---- figure ----
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.18})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.6, 5.2))

    # panel A: extension 4-solver per d_c
    col = {"DeepSeek": "#22223b", "Kimi-K2": "#4a7c59", "Qwen-14B": "#8896ab", "DeepSeek-reasoner": "#d1495b"}
    for m in ["DeepSeek", "Kimi-K2", "Qwen-14B"]:
        if not ext_tbl.get(m):
            continue
        pd = ext_tbl[m][1]; ds = sorted(pd)
        axA.plot(ds, [pd[d][1] for d in ds], "-o", ms=4, lw=1.1, alpha=0.55, color=col[m],
                 label=f"{NICE[m]} ({ext_tbl[m][0][1]:.2f})")
    if ext_tbl.get("DeepSeek-reasoner"):
        pd = ext_tbl["DeepSeek-reasoner"][1]; ds = sorted(pd)
        axA.plot(ds, [pd[d][1] for d in ds], "-s", ms=7, lw=2.8, color=col["DeepSeek-reasoner"],
                 markeredgecolor="white", markeredgewidth=0.8,
                 label=f"DeepSeek-reasoner ({ext_tbl['DeepSeek-reasoner'][0][1]:.2f})", zorder=5)
    axA.set_xlabel("extension $d_c$ (DeepSeek prelabel)"); axA.set_ylabel("judged accuracy")
    axA.set_title(f"(a) +100 high-$d_c$ extension: reasoner added as 4th solver\n"
                  f"reasoner univariate OR/$+1\\,d_c$ = {or_x:.2f}", fontsize=10)
    axA.legend(fontsize=8, loc="upper right")

    # panel B: reasoner d_c slope BY LABEL SOURCE (honest contrast, not a pool):
    # the negative slope is recovered under the clean 4-family consensus d_c (pilot258) but
    # NOT under the noisier single-rater extension prelabel d_c.
    for lab, pd, c, mk in [
        (f"pilot258, 4-family consensus $d_c$  (n={len(reas_pilot)}, OR {or_p:.2f})", pd_pilot, "#5b8e7d", "o"),
        (f"+100 ext, single-rater $d_c$  (n={len(reas_ext)}, OR {or_x:.2f})", pd_ext, "#e07a5f", "^")]:
        ds = sorted(pd)
        axB.plot(ds, [pd[d][1] for d in ds], "-", marker=mk, ms=6.5, lw=2.4, color=c, alpha=0.92, label=lab)
        for d in ds:
            axB.annotate(str(pd[d][0]), (d, pd[d][1]), textcoords="offset points",
                         xytext=(0, 7), ha="center", fontsize=6.5, color=c)
    axB.set_xlabel("$d_c$")
    axB.set_ylabel("reasoner judged accuracy")
    axB.set_title("(b) the negative slope needs clean labels:\n"
                  "recovered under consensus $d_c$, flat under single-rater extension $d_c$", fontsize=10)
    axB.legend(fontsize=7.5, loc="lower left", title="reasoner, by $d_c$ label source")
    fig.tight_layout(); OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG); fig.savefig(OUT_FIG.with_suffix(".pdf"))

    # ---- report ----
    md = ["# DeepSeek-reasoner on the +100 high-d_c extension (right-tail reinforcement)\n\n"]
    md.append("Reasoner arm added to the §4.8 extension using the identical solve/judge route "
              "(deepseek-reasoner@16384 -> deepseek-chat judge) and the same extension DeepSeek-prelabel "
              "d_c as the other three solvers.\n\n")
    md.append("## §4.8 extension, overall accuracy (4 solvers)\n\n| solver | valid n | accuracy |\n|---|---:|---:|\n")
    for m in ["DeepSeek", "Kimi-K2", "Qwen-14B", "DeepSeek-reasoner"]:
        if ext_tbl.get(m):
            md.append(f"| {NICE.get(m, m)} | {ext_tbl[m][0][0]} | {ext_tbl[m][0][1]:.3f} |\n")
    md.append("\n## Extension accuracy by d_c (4 solvers)\n\n| solver | " +
              " | ".join(f"d_c={d}" for d in range(0, 5)) + " |\n|---|" + "---:|" * 5 + "\n")
    for m in ["DeepSeek", "Kimi-K2", "Qwen-14B", "DeepSeek-reasoner"]:
        if not ext_tbl.get(m):
            continue
        pd = ext_tbl[m][1]
        cells = []
        for d in range(0, 5):
            cells.append(f"{pd[d][1]:.2f} (n={pd[d][0]})" if d in pd else "—")
        md.append(f"| {NICE.get(m, m)} | " + " | ".join(cells) + " |\n")
    md.append("\n## Reasoner d_c slope by label source (the honest result)\n\n")
    md.append("The extension was intended to reinforce the high-d_c right tail. It did not, and the "
              "reason is informative: the negative constraint-penalty slope is recovered on the strong "
              "solver under the **4-family consensus d_c** (pilot258) but **not** under the noisier "
              "**single-rater extension d_c**.\n\n")
    md.append(f"- pilot258 reasoner (consensus d_c): n={len(reas_pilot)}, univariate OR/$+1 d_c$ = **{or_p:.3f}** (p={p_p:.2g}) — clean negative slope\n")
    md.append(f"- extension reasoner (single-rater d_c): n={len(reas_ext)}, univariate OR/$+1 d_c$ = **{or_x:.3f}** (p={p_x:.2g}) — flat/slightly positive\n")
    md.append(f"- naive pool (different label sources, reported only for completeness, NOT a headline): "
              f"n={len(pooled)}, OR {or_a:.3f} (β={b_a:.3f}, p={p_a:.2g})\n\n")
    md.append("| d_c | pilot n | pilot acc | ext n | ext acc | pooled n | pooled acc |\n|---:|---:|---:|---:|---:|---:|---:|\n")

    def cell_n(t):
        return str(t[0]) if t else "0"

    def cell_a(t):
        return f"{t[1]:.2f}" if t else "—"
    for d in sorted(set(pd_pilot) | set(pd_ext)):
        pp = pd_pilot.get(d); xx = pd_ext.get(d); oo = pd_pool.get(d)
        md.append(f"| {d} | {cell_n(pp)} | {cell_a(pp)} | {cell_n(xx)} | {cell_a(xx)} | "
                  f"{cell_n(oo)} | {cell_a(oo)} |\n")
    md.append("\n## Caveat\n\n")
    md.append("The pilot and extension halves use different d_c label sources (pilot = 4-family LLM "
              "consensus; extension = single-rater DeepSeek prelabel under the same V4 protocol), so the "
              "pooled curve is a coverage/robustness extension that widens d_c support on a strong solver, "
              "not a single clean-labelled panel. The extension also remains light at d_c>=3. Figure F18.\n")
    OUT_MD.write_text("".join(md), encoding="utf-8")

    print(f"[ext overall] " + ", ".join(f"{m}={ext_tbl[m][0][1]:.3f}(n={ext_tbl[m][0][0]})"
                                        for m in ext_tbl))
    print(f"[ext reasoner per-dc] { {d:(pd_ext[d][0], round(pd_ext[d][1],3)) for d in sorted(pd_ext)} }")
    print(f"[pooled] n={len(pooled)} OR={or_a:.3f} p={p_a:.2g}  (pilot OR={or_p:.3f}, ext OR={or_x:.3f})")
    print(f"[wrote] {OUT_MD}\n[wrote] {OUT_FIG}")


if __name__ == "__main__":
    main()
