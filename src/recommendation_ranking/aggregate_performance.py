"""
Aggregated LODO performance across evaluable drugs
===============================================================
Reproduces the aggregated leave-one-drug-out performance from
lodo_ranking_results.csv, at BOTH aggregation levels:

  * drug-level  — average within each drug, then across drugs (each drug one
                  vote; data-rich drugs do not dominate). This is the paper's
                  primary framing. CI = cluster bootstrap over drugs.
  * cell-level  — flat average over all clinical trial arms (arm-weighted).
                  CI = bootstrap over cells.

Metrics per level: normalized ranking score (higher = better, 0.5 = random),
random selection baseline (status-quo) gap, recommended gap, 
gap reduction in percentage points, and gap reduction in percent.
The percent reduction is computed as ratio-of-means:
[(mean baseline - mean recommended)/mean baseline]

Groups: all 9 evaluable drugs; and the 7 study-design-distinct drugs (excluding
tirzepatide and medi0382, whose candidate pools are dose variants of a single
design, not distinct study designs).

All bootstraps: 10,000 iterations, seed 42.
Output: outputs/aggregate_performance.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).parent.parent.parent / "outputs" / "ranking"
LODO_CSV    = RESULTS_DIR / "lodo_ranking_results.csv"
OUT_CSV     = RESULTS_DIR / "aggregate_performance.csv"

B         = 10_000
SEED      = 42
EXCLUDE_7 = {"tirzepatide", "medi0382"}      # dose-arm-only (single study design)

METRICS = ["norm_rank", "baseline_gap", "recommended_gap", "reduction_pp", "reduction_pct"]


def _ci(samples):
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def cell_level(df, rng):
    """Arm-weighted: flat mean over cells; CI resamples cells (paired)."""
    sq = df["status_quo"].values
    cg = df["chosen_gap"].values
    nr = df["chosen_pct_rank"].values / 100.0
    n = len(sq)
    idx = rng.integers(0, n, size=(B, n))                    # paired cell resample
    sqb, cgb, nrb = sq[idx].mean(1), cg[idx].mean(1), nr[idx].mean(1)
    return {
        "norm_rank":       (nr.mean(),                       *_ci(nrb)),
        "baseline_gap":    (sq.mean(),                       *_ci(sqb)),
        "recommended_gap": (cg.mean(),                       *_ci(cgb)),
        "reduction_pp":    ((sq - cg).mean(),                *_ci(sqb - cgb)),
        "reduction_pct":   ((sq.mean() - cg.mean()) / sq.mean() * 100,
                            *_ci((sqb - cgb) / sqb * 100)),
    }


def drug_level(df, rng):
    """Each drug one vote: mean within drug, then across drugs; CI = cluster
    bootstrap resampling drugs."""
    dm = df.groupby("drug").agg(
        sq=("status_quo", "mean"),
        cg=("chosen_gap", "mean"),
        nr=("chosen_pct_rank", "mean"),
    )
    dm["nr"] = dm["nr"] / 100.0
    k = len(dm)
    pos = rng.integers(0, k, size=(B, k))                    # cluster resample of drugs
    SQ, CG, NR = dm["sq"].values[pos], dm["cg"].values[pos], dm["nr"].values[pos]
    sqb, cgb, nrb = SQ.mean(1), CG.mean(1), NR.mean(1)
    return {
        "norm_rank":       (dm["nr"].mean(),                 *_ci(nrb)),
        "baseline_gap":    (dm["sq"].mean(),                 *_ci(sqb)),
        "recommended_gap": (dm["cg"].mean(),                 *_ci(cgb)),
        "reduction_pp":    ((dm["sq"] - dm["cg"]).mean(),    *_ci(sqb - cgb)),
        "reduction_pct":   ((dm["sq"].mean() - dm["cg"].mean()) / dm["sq"].mean() * 100,
                            *_ci((sqb - cgb) / sqb * 100)),
    }


def summarize(df, group_label, rows):
    dm = df.groupby("drug").agg(sq=("status_quo", "mean"), cg=("chosen_gap", "mean"))
    improving = dm.index[dm["sq"] > dm["cg"]].tolist()
    worse     = dm.index[dm["sq"] <= dm["cg"]].tolist()
    n_drugs, n_cells = df["drug"].nunique(), df["eval_cell"].nunique()
    print(f"\n=== {group_label}: {n_drugs} drugs, {n_cells} clinical trial arms ===")
    print(f"    drugs improving on baseline: {len(improving)}/{n_drugs}"
          + (f"   (worse: {', '.join(worse)})" if worse else ""))
    for level, fn in [("drug", drug_level), ("cell", cell_level)]:
        rng = np.random.default_rng(SEED)                    # fresh seed per level -> reproducible
        res = fn(df, rng)
        for metric in METRICS:
            m, lo, hi = res[metric]
            rows.append({"group": group_label, "level": level, "metric": metric,
                         "mean": round(m, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})
    return improving, worse


def _fmt(rows, group, level, metric, d=2):
    r = next(x for x in rows if x["group"] == group and x["level"] == level and x["metric"] == metric)
    return f"{r['mean']:.{d}f} ({r['ci_lo']:.{d}f}-{r['ci_hi']:.{d}f})"


def main():
    df = pd.read_csv(LODO_CSV)
    groups = [
        ("all 9 drugs", df),
        ("7 drugs (excl. tirzepatide, medi0382)", df[~df["drug"].isin(EXCLUDE_7)].copy()),
    ]
    rows = []
    for label, sub in groups:
        summarize(sub, label, rows)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    # formatted comparison table
    label_map = {
        "norm_rank": "Normalized ranking score", "baseline_gap": "Baseline gap  %BW",
        "recommended_gap": "Recommended gap  %BW", "reduction_pp": "Gap reduction  pp",
        "reduction_pct": "Gap reduction  %",
    }
    for label, _ in groups:
        print(f"\n{'-'*88}\n{label}\n{'metric':30s}| {'DRUG-level (1 vote/drug)':26s}| cell-level (arm-weighted)")
        print("-" * 88)
        for metric in METRICS:
            d = 1 if metric == "reduction_pct" else 2
            print(f"{label_map[metric]:30s}| {_fmt(rows,label,'drug',metric,d):26s}| {_fmt(rows,label,'cell',metric,d)}")
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
