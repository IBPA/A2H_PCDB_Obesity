"""
Train the PRODUCTION recommender on ALL available data.
=======================================================
The deployed recommender is the model used to rank candidate preclinical designs for
a NEW clinical target, so it is trained on the COMPLETE dataset — all 11 drugs,
including single-candidate clinical arms (there is no held-out fold and no
>=2-candidate ranking filter).
Model selection has already chosen the family by majority vote across seeds
(model_selection.py → model_selection.json); this script:

  1. reads the selected family from model_selection.json,
  2. tunes the final hyperparameters by Leave-One-NCT-Out (LONO) inner CV over the
     FULL dataset (nested CV, same tuner as lodo_ranking / ranking_core.lono_tune_mae),
  3. fits the production model on all data and saves it.

Outputs (outputs/):
  * recommender_model_params.json — {model_family, params, lono_mae, n_rows, n_features}
  * recommender_model.pkl         — fitted bundle {imputer, scaler, model, feat_cols}

Usage:
    python src/recommendation/model_selection.py          # writes model_selection.json
    python src/recommendation/train_production_model.py    # reads it, trains production model
"""

import json
import warnings
from pathlib import Path

import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from ranking_core import (
    build_base, feat_cols_of, build_eval_cells,
    lono_tune_mae, _make_model,
    ABS_TGT, GROUP,
)

warnings.filterwarnings("ignore")

N_TRIALS = 60                      # matches lodo_ranking / ranking_evaluation tuning

SCRIPT_DIR   = Path(__file__).parent
REPO_ROOT    = SCRIPT_DIR.parent.parent
OUT_DIR      = REPO_ROOT / "outputs" / "ranking"
SELECTION    = OUT_DIR / "model_selection.json"
PARAMS_PATH  = OUT_DIR / "recommender_model_params.json"
MODEL_PATH   = OUT_DIR / "recommender_model.pkl"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_selected_family():
    """Read the family chosen by model_selection.py (majority vote across seeds)."""
    if not SELECTION.exists():
        raise FileNotFoundError(
            f"{SELECTION} not found — run model_selection.py first to choose the family.")
    sel = json.load(open(SELECTION))
    fam = sel["selected_family"]
    print(f"  Selected family (from {SELECTION.name}): {fam}  "
          f"({sel['seed_wins'][fam]}/{sel['n_seeds']} seeds)")
    return fam


def main():
    print("=" * 74)
    print("  Train PRODUCTION recommender — all drugs, LONO-tuned final params")
    print("=" * 74)
    family = load_selected_family()

    # Complete dataset: all 11 drugs, including single-candidate clinical arms (valid
    # training examples). The >=2-candidate filter is only for ranking EVALUATION, so
    # it is disabled here — the production model should learn from every available arm.
    cells = build_eval_cells(build_base(), rankable_only=False)
    feat_cols = feat_cols_of(cells)
    y = cells[ABS_TGT].values
    print(f"  {cells.shape[0]} rows | {cells[GROUP].nunique()} drugs | "
          f"{len(feat_cols)} features")

    # Impute on ALL data — the production model legitimately uses every drug (no
    # held-out fold). One-hot columns come pre-encoded from build_base; only the
    # continuous features carry NaN.
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(cells[feat_cols].values)
    tune_df = cells.copy()
    tune_df[feat_cols] = X_imp

    # Final hyperparameters by Leave-One-NCT-Out inner CV over the full dataset.
    print(f"\n  LONO-tuning {family} ({N_TRIALS} trials) on all drugs…")
    params, lono_mae = lono_tune_mae(tune_df, feat_cols, family, n_trials=N_TRIALS)

    with open(PARAMS_PATH, "w") as fh:
        json.dump({
            "model_family": family,
            "params":       params,
            "lono_mae":     lono_mae,
            "n_rows":       int(cells.shape[0]),
            "n_features":   len(feat_cols),
            "trained_on":   "complete dataset, all 11 drugs incl. single-candidate arms "
                            "(production model — NOT the LODO generalization estimate; that "
                            "uses per-fold params in lodo_per_fold_params.json)",
        }, fh, indent=2)
    print(f"  Saved params: {PARAMS_PATH}")

    # Fit the production model on all data with the tuned params, and save it.
    sc = StandardScaler()
    model = _make_model(family, params)
    model.fit(sc.fit_transform(X_imp), y)
    joblib.dump({"imputer": imp, "scaler": sc, "model": model, "feat_cols": feat_cols},
                MODEL_PATH)
    print(f"  Saved model:  {MODEL_PATH}  (LONO MAE={lono_mae:.3f})")


if __name__ == "__main__":
    main()
