"""
Generate per_drug_gap_ci.csv from the FOLD-SAFE lodo_ranking_results.csv.

Bootstraps evaluation cells within each drug (N=10,000, seed=42) to produce
95% CI on mean status_quo gap and mean chosen_gap per drug. Drugs with only
one evaluation cell have no CI (empty bounds).

Output: outputs/per_drug_gap_ci.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent.parent / "outputs" / "ranking"
LODO_CSV    = RESULTS_DIR / "lodo_ranking_results.csv"
OUT_CSV     = RESULTS_DIR / "per_drug_gap_ci.csv"
N_BOOT      = 10_000
SEED        = 42


def main():
    lodo_df = pd.read_csv(LODO_CSV)
    rng = np.random.default_rng(SEED)

    rows = []
    for drug, grp in lodo_df.groupby("drug"):
        cells = grp.groupby("eval_cell").agg(
            status_quo=("status_quo", "mean"),
            chosen_gap=("chosen_gap", "mean"),
        ).reset_index()
        n_cells = len(cells)
        sq_mean = cells["status_quo"].mean()
        cg_mean = cells["chosen_gap"].mean()

        if n_cells > 1:
            sq_vals = cells["status_quo"].values
            cg_vals = cells["chosen_gap"].values
            idx = rng.integers(0, n_cells, size=(N_BOOT, n_cells))
            sq_boot = sq_vals[idx].mean(axis=1)
            cg_boot = cg_vals[idx].mean(axis=1)
            sq_lo, sq_hi = np.percentile(sq_boot, [2.5, 97.5])
            cg_lo, cg_hi = np.percentile(cg_boot, [2.5, 97.5])
        else:
            sq_lo = sq_hi = cg_lo = cg_hi = float("nan")

        rows.append({
            "drug":            drug,
            "n_cells":         n_cells,
            "status_quo_mean": round(sq_mean, 2),
            "sq_lo":           round(sq_lo, 2) if not np.isnan(sq_lo) else "",
            "sq_hi":           round(sq_hi, 2) if not np.isnan(sq_hi) else "",
            "chosen_gap_mean": round(cg_mean, 2),
            "cg_lo":           round(cg_lo, 2) if not np.isnan(cg_lo) else "",
            "cg_hi":           round(cg_hi, 2) if not np.isnan(cg_hi) else "",
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(out.to_string(index=False))
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
