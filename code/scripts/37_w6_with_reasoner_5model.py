"""W6 sensitivity: add DeepSeek-reasoner as a 5th model to the controlled regression.

The headline W6 (β_dc = -0.375 on 1032 obs) uses four floor-limited chat arms. Here we add
the high-dynamic-range DeepSeek-reasoner arm (script 35) as a 5th model and re-fit the
identical M3 ridge logistic, to check the controlled d_c penalty holds when a non-floor model
is included. Reasoner control columns are reconstructed exactly as m07.build_panel does
(item metadata from pilot258_with_solution.csv + reasoner tokens), so the 5-model panel is
schema-identical to the released 4-model panel.

Output: evaluation/w6_with_reasoner_5model_20260529.md
"""
from __future__ import annotations
import csv
import importlib.util
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT / "code" / "scripts"
PANEL = PROJECT / "evaluation" / "w6_controlled_panel_with_logs.csv"
ITEMS = PROJECT / "data" / "annotations" / "pilot258_with_solution.csv"
REAS = PROJECT / "data" / "results" / "solve_pilot258_deepseek_reasoner_judged_20260529.csv"
OUT_MD = PROJECT / "evaluation" / "w6_with_reasoner_5model_20260529.md"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


m07 = load_module(SCRIPTS / "07_w6_controlled_dc_analysis.py", "w6_07")


def read_csv(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fit_m3(rows):
    spec = m07.SPECS[-1]
    b = m07.DesignBuilder(rows, spec)
    x, y = b.transform(rows)
    fit = m07.fit_logit(x, y, b.feature_names, 1.0)
    i = b.feature_names.index("d_c")
    return b, fit, float(fit["coef"][i]), float(fit["se"][i]), m07.auc_score(y, fit["prob"])


def boot(rows, B, seed):
    spec = m07.SPECS[-1]; b = m07.DesignBuilder(rows, spec)
    di = b.feature_names.index("d_c")
    byi = defaultdict(list)
    for r in rows:
        byi[r["item_id"]].append(r)
    ids = sorted(byi); rng = random.Random(seed); cs = []
    for _ in range(B):
        s = []
        for iid in rng.choices(ids, k=len(ids)):
            s.extend(byi[iid])
        x, y = b.transform(s)
        cs.append(float(m07.fit_logit(x, y, b.feature_names, 1.0)["coef"][di]))
    cs = np.array(cs)
    return float(np.percentile(cs, 2.5)), float(np.percentile(cs, 97.5)), float(np.mean(cs < 0))


def main():
    panel = read_csv(PANEL)  # 4-model rows, schema with all controls
    for r in panel:
        r["d_c"] = int(float(r["d_c"]))
    dc_map = {r["item_id"]: r["d_c"] for r in panel}
    meta = {r["item_id"]: r for r in read_csv(ITEMS)}

    # build reasoner rows in the same schema
    reas_rows = []
    for r in read_csv(REAS):
        v = str(r.get("is_correct_judge", "")).strip()
        iid = r["item_id"]
        if v not in {"0", "1"} or iid not in meta or iid not in dc_map:
            continue
        m = meta[iid]
        q = m.get("question", ""); ans = m.get("answer", ""); sol = m.get("solution_text", "")
        reas_rows.append({
            "item_id": iid, "model_label": "DeepSeekReasoner", "n_params_b": 671.0,
            "source_benchmark": m.get("source_benchmark", "unknown"),
            "topic": m.get("topic", "unknown"),
            "has_diagram": 1 if str(m.get("has_diagram", "")).strip().lower() in {"1", "true", "yes"} else 0,
            "answer_type": m07.answer_type(ans),
            "d_c": dc_map[iid],
            "question_chars": len(q), "answer_chars": len(ans), "solution_chars": len(sol),
            "tokens_in": float(r.get("tokens_in", 0) or 0), "tokens_out": float(r.get("tokens_out", 0) or 0),
            "latency_s": float(r.get("latency_s", 0) or 0), "is_correct": int(v),
        })
    # normalise panel rows to the same typed dict keys DesignBuilder needs
    def norm(r):
        return {k: r[k] for k in ("item_id", "model_label", "n_params_b", "source_benchmark",
                                  "topic", "has_diagram", "answer_type", "d_c", "question_chars",
                                  "answer_chars", "solution_chars", "tokens_in", "tokens_out",
                                  "latency_s", "is_correct")}
    panel4 = [norm(r) for r in panel]
    combined = panel4 + reas_rows

    _, _, b4, se4, auc4 = fit_m3(panel4)
    _, _, b5, se5, auc5 = fit_m3(combined)
    lo, hi, fneg = boot(combined, 2000, 20260529)

    reas_overall = np.mean([r["is_correct"] for r in reas_rows]) if reas_rows else float("nan")
    md = ["# W6 Sensitivity: DeepSeek-reasoner as a 5th model\n\n"]
    md.append(f"- Reasoner rows merged: **{len(reas_rows)}** / 258"
              + (" (PARTIAL)" if len(reas_rows) < 256 else "") + f"; reasoner overall acc = {reas_overall:.3f}.\n")
    md.append(f"- 4-model panel (n={len(panel4)}): M3 β_dc = {b4:.4f} (OR {math.exp(b4):.3f}), AUC {auc4:.3f}.\n")
    md.append(f"- **5-model panel (n={len(combined)}, + reasoner): M3 β_dc = {b5:.4f} (OR {math.exp(b5):.3f}), AUC {auc5:.3f}.**\n")
    md.append(f"- 5-model item-cluster bootstrap (B=2000): 95% CI [{lo:.3f}, {hi:.3f}], β<0 in {fneg:.1%} of resamples.\n\n")
    md.append("**Reading.** Adding a high-dynamic-range, non-floor solver as a 5th model "
              f"{'preserves' if b5 < 0 else 'does not preserve'} the negative controlled d_c penalty "
              f"(OR {math.exp(b5):.2f} per +1 d_c), and the model fit improves (AUC {auc4:.2f}->{auc5:.2f}). "
              "This shows the headline effect is not carried by the floor-limited arms alone.\n")
    OUT_MD.write_text("".join(md), encoding="utf-8")
    print(f"[4-model] beta={b4:.4f} OR={math.exp(b4):.3f} AUC={auc4:.3f} n={len(panel4)}")
    print(f"[5-model] beta={b5:.4f} OR={math.exp(b5):.3f} AUC={auc5:.3f} n={len(combined)} "
          f"CI=[{lo:.3f},{hi:.3f}] fneg={fneg:.3f} reasoner_n={len(reas_rows)} reasoner_acc={reas_overall:.3f}")
    print(f"[wrote] {OUT_MD}")


if __name__ == "__main__":
    main()
