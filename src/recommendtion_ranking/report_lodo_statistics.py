"""
Generate a summary statistics table from LODO results.

Outputs:
  outputs/ranking_evaluation/lodo_statistics_table.csv
  (also prints a formatted table to stdout)

Statistics reported
-------------------
Table 1 – Drug-level mean normalized ranking score
Table 2 – Overall mean normalized ranking score vs random (all 9 drugs & 7-drug subset)
Table 3 – Mean translation gap (baseline vs recommendation) with CI
Table 4 – Mean translation gap reduction with CI
Table 5 – Tirzepatide normalized ranking score
"""

from pathlib import Path
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).parent.parent.parent / "outputs" / "ranking_evaluation"
LODO_CSV    = RESULTS_DIR / "lodo_ranking_results.csv"
OUT_CSV     = RESULTS_DIR / "lodo_statistics_table.csv"

B   = 10_000
rng = np.random.default_rng(42)
EXCLUDE = {"medi0382", "tirzepatide"}

df = pd.read_csv(LODO_CSV)
df["norm_rank"] = df["chosen_pct_rank"] / 100


def bootstrap_mean_ci(vals, B=B, alpha=0.05):
    vals = np.asarray(vals)
    boot = np.array([vals[rng.integers(0, len(vals), len(vals))].mean()
                     for _ in range(B)])
    return vals.mean(), np.percentile(boot, 100 * alpha / 2), np.percentile(boot, 100 * (1 - alpha / 2))


def fmt(mean, lo, hi, decimals=3):
    d = decimals
    return f"{mean:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


# ── Table 1: Drug-level mean normalized ranking score ────────────────────────
print("\n=== Table 1: Drug-level mean normalized ranking score ===")
rows_t1 = []
for drug, grp in df.groupby("drug"):
    m, lo, hi = bootstrap_mean_ci(grp["norm_rank"])
    rows_t1.append({"drug": drug, "n_cells": len(grp),
                    "mean_norm_rank": round(m, 3),
                    "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
                    "mean_norm_rank_fmt": fmt(m, lo, hi)})
t1 = pd.DataFrame(rows_t1).sort_values("mean_norm_rank", ascending=False)
print(t1[["drug", "n_cells", "mean_norm_rank_fmt"]].to_string(index=False))


# ── Table 2: Overall mean normalized ranking score & improvement vs random ───
# "Improvement" = 0.5 (random) − mean_norm_rank  (positive = better than random)
print("\n=== Table 2: Mean normalized ranking score vs random (0.5) ===")
rows_t2 = []
for label, mask in [
    ("All 9 drugs",  pd.Series([True] * len(df))),
    ("7 drugs (excl. medi0382 & tirzepatide)", ~df["drug"].isin(EXCLUDE)),
]:
    subset = df[mask]["norm_rank"]
    m, lo, hi = bootstrap_mean_ci(subset)
    impr_m  = m - 0.5
    impr_lo = lo - 0.5
    impr_hi = hi - 0.5
    rows_t2.append({
        "group":           label,
        "n_cells":         int(mask.sum()),
        "mean_norm_rank":  fmt(m, lo, hi),
        "improvement_vs_random": fmt(impr_m, impr_lo, impr_hi),
    })
t2 = pd.DataFrame(rows_t2)
print(t2.to_string(index=False))


# ── Table 3: Mean translation gap (baseline vs recommendation) with CI ───────
print("\n=== Table 3: Mean translation gap — baseline vs recommendation ===")
rows_t3 = []
for label, mask in [
    ("All 9 drugs",  pd.Series([True] * len(df))),
    ("7 drugs (excl. medi0382 & tirzepatide)", ~df["drug"].isin(EXCLUDE)),
]:
    subset = df[mask]
    sq_m, sq_lo, sq_hi = bootstrap_mean_ci(subset["status_quo"])
    cg_m, cg_lo, cg_hi = bootstrap_mean_ci(subset["chosen_gap"])
    rows_t3.append({
        "group":               label,
        "n_cells":             int(mask.sum()),
        "baseline_gap_%BW":    fmt(sq_m, sq_lo, sq_hi, decimals=2),
        "recommended_gap_%BW": fmt(cg_m, cg_lo, cg_hi, decimals=2),
    })
t3 = pd.DataFrame(rows_t3)
print(t3.to_string(index=False))


# ── Table 4: Mean translation gap reduction with CI ──────────────────────────
# Bootstrapped jointly so CI reflects the paired difference
print("\n=== Table 4: Mean translation gap reduction (baseline − recommendation) ===")
rows_t4 = []
for label, mask in [
    ("All 9 drugs",  pd.Series([True] * len(df))),
    ("7 drugs (excl. medi0382 & tirzepatide)", ~df["drug"].isin(EXCLUDE)),
]:
    subset = df[mask]
    sq = subset["status_quo"].values
    cg = subset["chosen_gap"].values
    diff = sq - cg
    d_m, d_lo, d_hi = bootstrap_mean_ci(diff)
    pct_diff = diff / sq * 100
    p_m, p_lo, p_hi = bootstrap_mean_ci(pct_diff)
    rows_t4.append({
        "group":              label,
        "n_cells":            int(mask.sum()),
        "gap_reduction_%BW":  fmt(d_m, d_lo, d_hi, decimals=2),
        "gap_reduction_%":    fmt(p_m, p_lo, p_hi, decimals=1),
    })
t4 = pd.DataFrame(rows_t4)
print(t4.to_string(index=False))


# ── Table 5: Tirzepatide normalized ranking score ────────────────────────────
print("\n=== Table 5: Tirzepatide normalized ranking score ===")
tirz = df[df["drug"] == "tirzepatide"]["norm_rank"]
m, lo, hi = bootstrap_mean_ci(tirz)
print(f"  tirzepatide  n={len(tirz)}  mean norm. rank = {fmt(m, lo, hi)}")
rows_t5 = [{"drug": "tirzepatide", "n_cells": len(tirz),
            "mean_norm_rank_fmt": fmt(m, lo, hi)}]
t5 = pd.DataFrame(rows_t5)


# ── Save combined CSV ─────────────────────────────────────────────────────────
def tag(table_df, table_name):
    table_df = table_df.copy()
    table_df.insert(0, "table", table_name)
    return table_df

combined = pd.concat([
    tag(t1[["drug", "n_cells", "mean_norm_rank_fmt"]].rename(
            columns={"drug": "group", "mean_norm_rank_fmt": "value"}),
        "T1_drug_norm_rank"),
    tag(t2.rename(columns={"mean_norm_rank": "value"})[
            ["group", "n_cells", "value", "improvement_vs_random"]],
        "T2_overall_norm_rank"),
    tag(t3[["group", "n_cells", "baseline_gap_%BW", "recommended_gap_%BW"]],
        "T3_translation_gap"),
    tag(t4[["group", "n_cells", "gap_reduction_%BW", "gap_reduction_%"]],
        "T4_gap_reduction"),
    tag(t5[["drug", "n_cells", "mean_norm_rank_fmt"]].rename(
            columns={"drug": "group", "mean_norm_rank_fmt": "value"}),
        "T5_tirzepatide"),
], ignore_index=True)

combined.to_csv(OUT_CSV, index=False)
print(f"\nSaved: {OUT_CSV}")
