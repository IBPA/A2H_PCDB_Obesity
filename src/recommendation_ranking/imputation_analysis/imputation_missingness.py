"""
How much of the recommender's input was imputed?
================================================
Descriptive companion to imputation_sensitivity.py. Answers the "how much of the data
were imputed" question for the PRODUCTION recommender input — the exact matrix
train_production_model.py hands to SimpleImputer: every arm of all 11 drugs
(build_eval_cells(..., rankable_only=False)), which is also the matrix behind the
SHAP figure.

The imputer targets no pre-specified column list: it is fit on the whole feature
matrix and fills whatever is NaN. So the imputed set is discovered from the data,
and this script reports what is actually filled rather than what was intended.

Output: outputs/imputation_analysis/imputation_missingness.csv
        Per-column rows, then TOTAL rows carrying the dataset-level figures. Every
        row means the same thing: n_missing of n_values were imputed (pct_missing).
        The unit is the CANDIDATE PAIRING (preclinical arm x clinical arm), which is
        what the model ingests — an arm linked to k clinical arms is counted k times.
"""

import sys
from pathlib import Path

import pandas as pd

# ranking_core lives two directories up (src/recommendation/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ranking_core import (
    build_base, build_eval_cells, feat_cols_of, ohe_dummy_cols,
    ABS_TGT, GROUP,
)

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT  = SCRIPT_DIR.parent.parent.parent
OUT_DIR    = REPO_ROOT / "outputs" / "ranking" / "imputation_analysis"
OUT_CSV    = OUT_DIR / "imputation_missingness.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    cells, feat_cols, numeric_cols, onehot_cols = load_production_features()
    per_column = count_missing_per_column(cells, feat_cols, onehot_cols)
    totals     = dataset_totals(cells, feat_cols, numeric_cols, onehot_cols)

    table = pd.concat([per_column, totals], ignore_index=True)
    table.to_csv(OUT_CSV, index=False)

    report(table, cells, feat_cols)
    return 0


def load_production_features():
    """The production input: all 11 drugs, no >=2-candidate ranking filter — matching
    train_production_model.py, so the reported rates describe the model in the figure."""
    cells = build_eval_cells(build_base(), rankable_only=False)
    feat_cols = feat_cols_of(cells)
    onehot_cols = ohe_dummy_cols(feat_cols)
    numeric_cols = [c for c in feat_cols if c not in set(onehot_cols)]
    return cells, feat_cols, numeric_cols, onehot_cols


def count_missing_per_column(cells, feat_cols: list[str],
                             onehot_cols: list[str]) -> pd.DataFrame:
    onehot = set(onehot_cols)
    counts = cells[feat_cols].isna().sum()
    return (pd.DataFrame({
        "scope":       "column",
        "name":        counts.index,
        "block":       ["one_hot" if c in onehot else "numeric" for c in counts.index],
        "n_missing":   counts.values,
        "n_values":    len(cells),
        "pct_missing": (counts.values / len(cells) * 100).round(2),
    }).sort_values("n_missing", ascending=False).reset_index(drop=True))


def dataset_totals(cells, feat_cols: list[str], numeric_cols: list[str],
                   onehot_cols: list[str]) -> pd.DataFrame:
    """Dataset-level figures, in the same numerator/denominator shape as the per-column
    rows. Categoricals are all-zero encoded by build_base and the target is complete,
    so those two rows are reported as verification rather than as expected findings."""
    n_rows    = len(cells)      # candidate pairings, not distinct preclinical arms
    n_missing = int(cells[feat_cols].isna().sum().sum())
    n_rows_affected = int(cells[feat_cols].isna().any(axis=1).sum())
    totals = [
        ("all numeric features",  "numeric", n_missing, n_rows * len(numeric_cols)),
        ("all one-hot features",  "one_hot", int(cells[onehot_cols].isna().sum().sum()),
                                             n_rows * len(onehot_cols)),
        ("all features",          "all",     n_missing, n_rows * len(feat_cols)),
        ("candidate pairings with >=1 imputed", "pairing", n_rows_affected, n_rows),
        ("target (abs_gap)",      "target",  int(cells[ABS_TGT].isna().sum()), n_rows),
    ]
    return pd.DataFrame([
        {"scope": "TOTAL", "name": name, "block": block, "n_missing": n_missing_,
         "n_values": n_values, "pct_missing": round(n_missing_ / n_values * 100, 2)}
        for name, block, n_missing_, n_values in totals
    ])


def report(table: pd.DataFrame, cells, feat_cols: list[str]) -> None:
    print("=" * 78)
    print(f"  Missingness in the PRODUCTION recommender input "
          f"({len(cells)} rows, {cells[GROUP].nunique()} drugs, {len(feat_cols)} features)")
    print("=" * 78)
    columns = table[table["scope"] == "column"]
    for r in columns[columns["n_missing"] > 0].itertuples():
        print(f"  {r.pct_missing:6.2f}%  {r.n_missing:5d}/{r.n_values}  {r.name}")
    print(f"  (all other {int((columns['n_missing'] == 0).sum())} feature columns complete)")
    print()
    for r in table[table["scope"] == "TOTAL"].itertuples():
        print(f"  {r.pct_missing:6.2f}%  {r.n_missing:5d}/{r.n_values}  {r.name}")
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    raise SystemExit(main())
