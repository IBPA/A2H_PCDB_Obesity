"""
Fold-safe LODO ranking — the methodologically correct version.
==============================================================
Imputation is performed STRICTLY WITHIN EACH TRAINING FOLD (the training-drug
median of each continuous feature is applied to the held-out drug). 
Hyperparameters are re-tuned per fold on that fold-safe-imputed
training data (nested CV).

Writes to  outputs/.

Outputs: lodo_ranking_results.csv, lodo_per_drug_summary.csv, lodo_per_fold_params.json
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from ranking_core import (
    build_base, feat_cols_of, build_eval_cells,
    ohe_dummy_cols, _ohe,
    lono_tune_mae, _make_model, aggregate, normalized_ranking_score, per_drug_summary,
    ABS_TGT, GROUP, NCT, EVAL_GROUP,
)

warnings.filterwarnings("ignore")

FAMILY   = "LightGBM"
N_TRIALS = 60

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT  = SCRIPT_DIR.parent.parent
OUT_DIR    = REPO_ROOT / "outputs" / "ranking"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def foldsafe_lodo(cells, feat_cols, n_trials=N_TRIALS):
    """For each held-out drug: impute in-fold (training median -> held-out),
    retune per fold on the fold-safe data, train, predict, rank per cell."""
    y = cells[ABS_TGT].values
    groups = cells[GROUP].values #Intervention
    ohe_cols = ohe_dummy_cols(feat_cols)
    assert cells.groupby(EVAL_GROUP)[GROUP].nunique().max() == 1, \
        "eval cell spans >1 drug — leave-one-drug-out would split a cell across train/test"
    records, per_fold = [], {}
    for drug in sorted(np.unique(groups)):
        tr = groups != drug
        Xtr = cells.loc[tr, feat_cols].copy()
        Xte = cells.loc[~tr, feat_cols].copy()
        # fold-safe one-hot: categories absent from the training drugs -> all-zero
        Xtr, Xte = _ohe(Xtr, Xte, ohe_cols)

        # ── fold-safe imputation: scikit-learn median imputer, fit on the training
        #    drugs and applied to the held-out drug ──
        imp = SimpleImputer(strategy="median")
        Xtr = pd.DataFrame(imp.fit_transform(Xtr.values), columns=feat_cols, index=Xtr.index)
        Xte = pd.DataFrame(imp.transform(Xte.values), columns=feat_cols, index=Xte.index)

        # ── retune hyperparameters on the fold-safe-imputed training drugs only ──
        train_df = cells.loc[tr].copy()
        train_df[feat_cols] = Xtr
        # ── Nested-CV boundary guard (first-class leakage check): the Optuna tuning
        #    frame is the 8 TRAINING drugs only, so no held-out-drug row can enter any
        #    inner LONO fold / trial. (Enforced again inside lono_tune_mae.) ──
        assert (train_df[GROUP] != drug).all(), (
            f"NESTED-CV LEAKAGE: {int((train_df[GROUP] == drug).sum())} held-out "
            f"'{drug}' rows in the Optuna tuning frame")
        fold_params, fold_mae = lono_tune_mae(train_df, feat_cols, FAMILY,
                                              n_trials=n_trials, holdout_group=drug)
        per_fold[drug] = {"params": fold_params, "lono_mae": fold_mae}

        sc = StandardScaler()
        model = _make_model(FAMILY, fold_params)
        model.fit(sc.fit_transform(Xtr.values), y[tr])
        te = cells.loc[~tr].copy()
        te["pred_abs_gap"] = model.predict(sc.transform(Xte.values))

        for cell_id in te[EVAL_GROUP].unique():  # EVAL_GROUP (NCT × clinical arm) cell identifier
            rows = te[te[EVAL_GROUP] == cell_id]
            n_pre = len(rows)
            best = rows["pred_abs_gap"].idxmin()
            chosen_gap = float(rows.loc[best, ABS_TGT])
            status_quo = float(rows[ABS_TGT].mean())
            if n_pre > 2:
                rho, pval = stats.spearmanr(rows["pred_abs_gap"].values, rows[ABS_TGT].values)
            else:
                rho, pval = np.nan, np.nan
            records.append({
                "eval_cell": cell_id, "NCT": rows[NCT].iloc[0], "drug": drug,
                "n_preclinical": n_pre, "status_quo": status_quo, "chosen_gap": chosen_gap,
                "improvement": status_quo - chosen_gap,
                "chosen_pct_rank": normalized_ranking_score(rows[ABS_TGT].values, chosen_gap),
                "spearman_rho": rho, "spearman_p": pval, "picker": "LODO_model",
            })
        print(f"  [{drug}] fold done  (lono_mae={fold_mae:.3f})", flush=True)
    return pd.DataFrame(records), per_fold


def main():
    print("=" * 74)
    print("  Fold-safe LODO ranking (imputation within each training fold)")
    print("=" * 74)
    cells = build_eval_cells(build_base())     # NO imputation; all continuous NaN
    feat_cols = feat_cols_of(cells)
    print(f"  {cells.shape[0]} rows | {cells[GROUP].nunique()} drugs | "
          f"{cells[EVAL_GROUP].nunique()} eval cells | {len(feat_cols)} features")

    lodo_df, per_fold = foldsafe_lodo(cells, feat_cols)

    lodo_df.to_csv(OUT_DIR / "lodo_ranking_results.csv", index=False)
    json.dump({"model_family": FAMILY, "folds": per_fold},
              open(OUT_DIR / "lodo_per_fold_params.json", "w"), indent=2)
    per_drug = per_drug_summary(lodo_df)
    per_drug.to_csv(OUT_DIR / "lodo_per_drug_summary.csv", index=False)

    ag = aggregate(lodo_df)
    print(f"\n  cell_gap={ag['cell_mean_chosen_gap']:.2f}  drug_gap={ag['drug_mean_chosen_gap']:.2f}  "
          f"cell_pct_rank={ag['cell_mean_pct_rank']:.1f}  drug_pct_impr={ag['drug_pct_improvement']:.1f}%")
    print("\n  Per-drug performance:")
    print(per_drug[["drug", "n_ncts", "n_cells", "status_quo", "chosen_gap",
                    "pct_improvement", "mean_pct_rank", "mean_rho", "n_sig"]].to_string(index=False))
    print(f"\n  Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
