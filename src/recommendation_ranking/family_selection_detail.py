"""
Per-fold model-family selection detail
======================================
lodo_ranking.py records only which family won each outer fold and the seed vote.
This reports what that vote was decided on: every family's mean normalized ranking
score and the mean |gap| of the arm it picks, per fold.

The comparison is the one select_family_inner runs — inner leave-one-drug-out over
the 8 TRAINING drugs, default hyperparameters, 10 seeds — so the values describe
performance on the training drugs, NOT on the held-out drug. Held-out performance
is lodo_per_drug_summary.csv.

Family selection happens before hyperparameter tuning and does not depend on it,
so this reproduces the pipeline's choice without re-running Optuna. Every computed
winner is checked against lodo_per_fold_params.json.

A final block repeats the comparison over all 9 drugs — model_selection.py's
full-dataset vote, which picks the family for the deployed recommender.

Output: outputs/ranking/lodo_family_selection.csv
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from model_selection import family_comparison
from ranking_core import (
    GROUP, _DEFAULT_PARAMS, build_base, build_eval_cells, feat_cols_of,
)

SEEDS = list(range(10))
FAMILIES = list(_DEFAULT_PARAMS)
FULL_DATASET = "all 9 drugs"

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR.parent.parent / "outputs" / "ranking"
FOLD_JSON = RESULTS_DIR / "lodo_per_fold_params.json"
OUT_CSV = RESULTS_DIR / "lodo_family_selection.csv"


def main() -> int:
    print("=" * 74)
    print(f"  Per-fold family selection detail — {len(SEEDS)} seeds, {len(FAMILIES)} families")
    print("=" * 74)
    cells = build_eval_cells(build_base())
    feat_cols = feat_cols_of(cells)

    table = build_table(cells, feat_cols)
    table.to_csv(OUT_CSV, index=False)

    print()
    print(table.to_string(index=False))
    verify_against_pipeline(table)
    print(f"\n  Saved: {OUT_CSV}")
    return 0


def build_table(cells, feat_cols) -> pd.DataFrame:
    """One block of three rows per outer fold, then one for the full dataset."""
    rows = []
    for drug in sorted(cells[GROUP].unique()):
        rows += compare_families(cells[cells[GROUP] != drug], feat_cols, held_out=drug)
        print(f"  fold done: {drug}", flush=True)

    rows += compare_families(cells, feat_cols, held_out=FULL_DATASET)
    print(f"  fold done: {FULL_DATASET}", flush=True)
    return pd.DataFrame(rows)


def compare_families(frame, feat_cols, held_out):
    """Each family's mean score and realised gap over the seeds, plus its win count."""
    scores = {family: [] for family in FAMILIES}
    gaps = {family: [] for family in FAMILIES}
    wins = {family: 0 for family in FAMILIES}

    for seed in SEEDS:
        scored = family_comparison(frame, feat_cols, seed)
        for family in FAMILIES:
            scores[family].append(scored[family]["pct_rank"])
            gaps[family].append(scored[family]["chosen_gap"])
        wins[max(FAMILIES, key=lambda f: scored[f]["pct_rank"])] += 1

    selected = max(FAMILIES, key=lambda f: wins[f])
    return [{
        "held_out_drug": held_out,
        "family": family,
        "mean_norm_rank": round(float(np.mean(scores[family])) / 100, 3),
        "mean_top_rank_gap": round(float(np.mean(gaps[family])), 2),
        "seed_wins": wins[family],
        "win_frac": round(wins[family] / len(SEEDS), 2),
        "selected": family == selected,
    } for family in FAMILIES]


def verify_against_pipeline(table) -> None:
    """Each fold's winner here must be the family lodo_ranking.py tuned and scored."""
    folds = json.loads(FOLD_JSON.read_text())["folds"]
    print("\n  Winner check against lodo_per_fold_params.json:")
    mismatches = []
    for drug, group in table.groupby("held_out_drug", sort=False):
        if drug not in folds:
            continue
        computed = group.loc[group["selected"], "family"].iloc[0]
        recorded = folds[drug]["family"]
        if computed != recorded:
            mismatches.append(f"{drug}: computed {computed}, pipeline {recorded}")
        print(f"    {drug:14s} {computed:13s} {'ok' if computed == recorded else 'MISMATCH'}")
    if mismatches:
        raise ValueError("family selection does not reproduce the pipeline: "
                         + "; ".join(mismatches))


if __name__ == "__main__":
    sys.exit(main())
