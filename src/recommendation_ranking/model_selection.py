"""
Model selection — fold-safe LODO family comparison by MAJORITY VOTE across seeds.
================================================================================
Compares LightGBM, XGBoost, and RandomForest with DEFAULT hyperparameters via
leave-one-drug-out cross-validation, scored by mean chosen-arm percentile rank
(drug-averaged) — the criterion in ranking_evaluation.lodo_mae_benchmark.

Fold-safe: within each LODO fold, missing continuous values are imputed with the
training-fold median (plain per-column median) applied to the held-out drug — no
test information leaks into the fill.

The three families are statistically close, so any single seed can flip the
winner. To be robust, this repeats the comparison over N seeds and selects the
family that wins the most seeds (majority vote), rather than trusting one seed.

Alongside the selection criterion, each family also reports the translation gap
actually realised by the arm it picks (mean_chosen_gap), on the same drug-first
footing: mean across cells within a drug, then across drugs, then across seeds.
This is a magnitude in the target's own units, whereas pct-rank is positional.

Output: outputs/model_selection.json  (selected family; train_production_model.py reads it)
        outputs/model_selection.csv   (per-family summary table)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from ranking_core import (
    build_base, feat_cols_of, build_eval_cells,
    ohe_dummy_cols, _ohe,
    _DEFAULT_PARAMS, _make_model, normalized_ranking_score,
    ABS_TGT, GROUP, EVAL_GROUP,
)

SEEDS = list(range(10))    # 0..9 — majority vote over these

SCRIPT_DIR  = Path(__file__).parent
REPO_ROOT   = SCRIPT_DIR.parent.parent
RESULTS_DIR = REPO_ROOT / "outputs" / "ranking"
OUT_JSON    = RESULTS_DIR / "model_selection.json"
OUT_CSV     = RESULTS_DIR / "model_selection.csv"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def family_comparison(cells, feat_cols, seed):
    """Fold-safe LODO family comparison with default params at one seed.
    Missing values imputed in-fold with scikit-learn's median SimpleImputer
    (fit on the training drugs, applied to the held-out drug). The imputer is fit
    once per fold and shared across families.

    Returns {family: {"pct_rank": float, "chosen_gap": float}}. Both are
    drug-first averages: mean across the (NCT × clinical arm) cells within a
    drug, then mean across drugs — so each drug carries one vote regardless of
    how many cells it contributes.
    """
    y = cells[ABS_TGT].values
    groups = cells[GROUP].values
    cell_ids = cells[EVAL_GROUP].values
    drugs = sorted(np.unique(groups))
    ohe_cols = ohe_dummy_cols(feat_cols)
    per_drug = {f: {"pct_rank": [], "chosen_gap": []} for f in _DEFAULT_PARAMS}
    for drug in drugs:
        tr = groups != drug
        te = groups == drug
        # fold-safe one-hot: categories absent from the training drugs -> all-zero
        Xtr_df, Xte_df = _ohe(cells.loc[tr, feat_cols], cells.loc[te, feat_cols], ohe_cols)
        imp = SimpleImputer(strategy="median")           # fold-safe median imputer
        Xtr = imp.fit_transform(Xtr_df.values)
        Xte = imp.transform(Xte_df.values)
        for family, base_params in _DEFAULT_PARAMS.items():
            params = dict(base_params)
            params["random_state"] = seed
            sc = StandardScaler()
            model = _make_model(family, params)
            model.fit(sc.fit_transform(Xtr), y[tr]) # z-score normalization within training fold
            pred = model.predict(sc.transform(Xte)) # apply the same normalization to the test set
            cells_te = cell_ids[te]
            cell_prs, cell_gaps = [], []
            for cell_id in np.unique(cells_te):
                mask = cells_te == cell_id
                actual = y[te][mask]
                chosen = actual[np.argmin(pred[mask])]     # arm with lowest predicted gap
                cell_prs.append(normalized_ranking_score(actual, chosen))
                cell_gaps.append(float(chosen))            # |gap| actually realised by that arm
            per_drug[family]["pct_rank"].append(float(np.mean(cell_prs)))
            per_drug[family]["chosen_gap"].append(float(np.mean(cell_gaps)))
    return {f: {metric: float(np.mean(vals)) for metric, vals in per_drug[f].items()}
            for f in _DEFAULT_PARAMS}


def main():
    print("=" * 74)
    print(f"  Model selection — fold-safe LODO family comparison, majority vote / {len(SEEDS)} seeds")
    print("=" * 74)
    cells = build_eval_cells(build_base())     # raw features, NaN left for in-fold imputation
    feat_cols = feat_cols_of(cells)
    families = list(_DEFAULT_PARAMS.keys())

    per_seed, wins = [], {f: 0 for f in families}
    pct_by_family = {f: [] for f in families}
    gap_by_family = {f: [] for f in families}
    for seed in SEEDS:
        scored = family_comparison(cells, feat_cols, seed)
        # Selection criterion is unchanged: highest drug-averaged pct-rank.
        winner = max(families, key=lambda f: scored[f]["pct_rank"])
        wins[winner] += 1
        for f in families:
            pct_by_family[f].append(scored[f]["pct_rank"])
            gap_by_family[f].append(scored[f]["chosen_gap"])
        per_seed.append({
            "seed": seed,
            **{f"{f}_score": round(scored[f]["pct_rank"], 1) for f in families},
            **{f"{f}_gap":   round(scored[f]["chosen_gap"], 2) for f in families},
            "winner": winner,
        })

    selected = max(wins, key=wins.get)

    print("\n  Per-seed winner (pct-rank per family):")
    print(pd.DataFrame(per_seed).to_string(index=False))

    summary = pd.DataFrame([{
        "family":         f,
        "seed_wins":      wins[f],
        "win_frac":       round(wins[f] / len(SEEDS), 2),
        "mean_pct_rank":  round(float(np.mean(pct_by_family[f])), 1),
        "mean_chosen_gap": round(float(np.mean(gap_by_family[f])), 2),
        "selected":       f == selected,
    } for f in families]).sort_values(["seed_wins", "mean_pct_rank"], ascending=False).reset_index(drop=True)

    # Machine-readable selection result — consumed by train_production_model.py to
    # know which family to train the final recommender on.
    result = {
        "selected_family": selected,
        "n_seeds":         len(SEEDS),
        "seed_wins":       {f: int(wins[f]) for f in families},
        "win_frac":        {f: round(wins[f] / len(SEEDS), 2) for f in families},
        "mean_pct_rank":   {f: round(float(np.mean(pct_by_family[f])), 1) for f in families},
        "mean_chosen_gap": {f: round(float(np.mean(gap_by_family[f])), 2) for f in families},
        "per_seed":        per_seed,
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(result, fh, indent=2)
    summary.to_csv(OUT_CSV, index=False)

    print("\n  Summary (majority vote):")
    print(summary.to_string(index=False))
    print(f"\n  Selected family: {selected}  ({wins[selected]}/{len(SEEDS)} seeds)")
    print(f"  Saved: {OUT_JSON}")
    print(f"  Saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
