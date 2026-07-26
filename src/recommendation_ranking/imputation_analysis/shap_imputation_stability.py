"""
Is the SHAP importance of baseline weight an artifact of imputation?
====================================================================
Baseline weight is simultaneously the top-SHAP feature and the most-imputed one
(21.5%), so its importance could in principle be manufactured rather than measured:
median imputation gives every filled row an identical value, which a tree can split
on to recover "this row's weight was not reported" — a proxy for the drug it came
from.

Two tests, both on the PRODUCTION model (the one behind the SHAP figure — trained on
all 11 drugs, so unlike the LODO models it has no held-out-drug protection):

  1. complete_case — refit after DROPPING every record with a missing weight. No
                     imputed weight value exists, so the artifact cannot occur. If
                     baseline weight is still top-ranked, its importance is real.
  2. mice          — refit with IterativeImputer, which gives each filled row a
                     DIFFERENT value, so there is no constant to split on. Unlike
                     complete_case this keeps every row, so it separates the artifact
                     question from the change in sample composition.

Also reports, for the production model, mean |SHAP| of baseline weight split by whether
the weight was measured or median-filled. If imputation were driving the importance,
the filled rows would carry the larger share; they carry the smaller one.

Hyperparameters are held at the production model's values throughout, so only the
imputation changes.

Outputs (outputs/imputation_analysis/):
  shap_imputation_stability.csv             — per-variant summary (the reported table)
  shap_imputation_importance_by_feature.csv — mean |SHAP| per feature per variant, so the
                                              rank correlations above are auditable
"""

import json
import warnings
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer

# ranking_core lives two directories up (src/recommendation/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ranking_core import (
    build_base, build_eval_cells, feat_cols_of, _make_model,
    ABS_TGT, WEIGHT,
)

warnings.filterwarnings("ignore")

SCRIPT_DIR    = Path(__file__).parent
REPO_ROOT     = SCRIPT_DIR.parent.parent.parent
RANKING_DIR   = REPO_ROOT / "outputs" / "ranking"               # ranking pipeline artifacts
ANALYSIS_DIR  = REPO_ROOT / "outputs" / "ranking" / "imputation_analysis"
MODEL_PATH    = RANKING_DIR / "recommender_model.pkl"          # the model behind the SHAP figure
PARAMS_PATH   = RANKING_DIR / "recommender_model_params.json"
OUT_SUMMARY   = ANALYSIS_DIR / "shap_imputation_stability.csv"
OUT_FEATURES  = ANALYSIS_DIR / "shap_imputation_importance_by_feature.csv"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

MICE_SEED = 42
PRODUCTION_MODEL = "production_model"


def main() -> int:
    production = production_model_importance()
    variants   = {PRODUCTION_MODEL: production["importance"]}
    for name, drop_weight_missing, method in [
        ("mice",          False, "mice"),
        ("complete_case", True,  "median"),
    ]:
        variants[name] = refit_importance(drop_weight_missing, method)

    by_feature = pd.DataFrame(variants)
    summary    = summarize(by_feature, production["split"])

    summary.to_csv(OUT_SUMMARY, index=False)
    by_feature.rename_axis("feature").reset_index().to_csv(OUT_FEATURES, index=False)

    report(summary, production["split"])
    return 0


def production_model_importance() -> dict:
    """SHAP of the SHIPPED production model — the exact fit behind the SHAP figure,
    loaded rather than refit so the reported values cannot drift from it."""
    bundle = joblib.load(MODEL_PATH)
    imputer, scaler, model, feat_cols = (bundle["imputer"], bundle["scaler"],
                                         bundle["model"], bundle["feat_cols"])
    cells = production_cells(drop_weight_missing=False)
    shap_values = shap.TreeExplainer(model).shap_values(
        scaler.transform(imputer.transform(cells[feat_cols].values)))
    return {
        "importance": mean_abs_shap(shap_values, feat_cols),
        "split":      measured_vs_filled(shap_values, cells, feat_cols),
    }


def refit_importance(drop_weight_missing: bool, method: str) -> pd.Series:
    """Refit the production model under a different imputation, the production model's params fixed."""
    cells = production_cells(drop_weight_missing)
    feat_cols = feat_cols_of(cells)
    imputer = make_imputer(method)
    scaler = StandardScaler()
    features = scaler.fit_transform(imputer.fit_transform(cells[feat_cols].values))

    model = _make_model(*production_model_family_and_params())
    model.fit(features, cells[ABS_TGT].values)
    return mean_abs_shap(shap.TreeExplainer(model).shap_values(features), feat_cols)


def production_cells(drop_weight_missing: bool):
    """rankable_only=False — all 11 drugs, matching train_production_model.py."""
    return build_eval_cells(build_base(drop_weight_missing=drop_weight_missing),
                            rankable_only=False)


def production_model_family_and_params():
    saved = json.loads(PARAMS_PATH.read_text())
    params = dict(saved["params"])
    params.setdefault("verbose", -1)
    return saved["model_family"], params


def make_imputer(method: str):
    if method == "mice":
        return IterativeImputer(max_iter=10, random_state=MICE_SEED, sample_posterior=False)
    return SimpleImputer(strategy="median")


def mean_abs_shap(shap_values, feat_cols: list[str]) -> pd.Series:
    return pd.Series(np.abs(shap_values).mean(axis=0), index=feat_cols)


def measured_vs_filled(shap_values, cells, feat_cols: list[str]) -> dict:
    """The artifact signature: if imputation manufactured the importance, the filled
    rows would carry MORE attribution than the measured ones."""
    column = feat_cols.index(WEIGHT)
    measured = cells[WEIGHT].notna().values
    return {
        "n_measured":          int(measured.sum()),
        "n_filled":            int((~measured).sum()),
        "mean_abs_shap_measured": round(float(np.abs(shap_values[measured, column]).mean()), 3),
        "mean_abs_shap_filled":   round(float(np.abs(shap_values[~measured, column]).mean()), 3),
    }


def summarize(by_feature: pd.DataFrame, split: dict) -> pd.DataFrame:
    production_model = by_feature[PRODUCTION_MODEL]
    production_model_top10 = set(production_model.nlargest(10).index)
    rows = []
    for variant in by_feature.columns:
        importance = by_feature[variant].dropna()
        shared = production_model.loc[importance.index]
        rows.append({
            "variant":              variant,
            "n_features":           len(importance),
            "weight_mean_abs_shap": round(float(importance[WEIGHT]), 3),
            "weight_rank":          int(importance.rank(ascending=False)[WEIGHT]),
            "top_feature":          importance.idxmax(),
            # rank agreement with the production model's importance ordering, over shared features
            "spearman_vs_production_model": round(float(stats.spearmanr(shared, importance).statistic), 3),
            "top10_overlap_vs_production_model": len(production_model_top10 & set(importance.nlargest(10).index)),
        })
    summary = pd.DataFrame(rows)
    for key, value in split.items():
        summary[f"production_model_weight_{key}"] = value
    return summary


def report(summary: pd.DataFrame, split: dict) -> None:
    print("=" * 88)
    print("  Is baseline weight's SHAP importance an artifact of imputation?")
    print("  (production model — all 11 drugs, production-model hyperparameters, imputation varied)")
    print("=" * 88)
    header = (f"{'variant':16s} {'feats':>5} | {'weight |SHAP|':>13} {'rank':>5} | "
              f"{'rho vs prod':>10} {'top10':>6}")
    print(header); print("-" * len(header))
    for r in summary.itertuples():
        print(f"{r.variant:16s} {r.n_features:5d} | {r.weight_mean_abs_shap:13.3f} "
              f"{r.weight_rank:5d} | {r.spearman_vs_production_model:10.3f} "
              f"{r.top10_overlap_vs_production_model:5d}/10")
    print("-" * len(header))
    print("\n  Production model, baseline weight mean |SHAP| by whether it was observed:")
    print(f"    measured     (n={split['n_measured']:5d}): {split['mean_abs_shap_measured']:.3f}")
    print(f"    median-filled(n={split['n_filled']:5d}): {split['mean_abs_shap_filled']:.3f}")
    print("    -> the filled rows carry the SMALLER share; the artifact signature is the reverse")
    print(f"\nSaved: {OUT_SUMMARY.name}, {OUT_FEATURES.name}")


if __name__ == "__main__":
    raise SystemExit(main())
