"""
Are the top SHAP features stable across the LODO folds?
=======================================================
The Discussion explains per-drug LODO performance in terms of the features the model
relies on — baseline body weight, treatment duration and animal age. Every LODO fold
trains its own model on 8 drugs, and it is those fold models that produced the LODO
scores, so this checks that they rest on the same features: the argument is then made
about the models that actually did the ranking.

Each fold is rebuilt exactly as lodo_ranking.py builds it — the fold's own tuned
hyperparameters, its imputer and scaler fit on its 8 training drugs, and the fold-safe
one-hot mask — then explained.

SHAP is computed for EVERY fold on the SAME matrix: the LODO evaluation universe, the
9 evaluable drugs over 1,426 candidate pairings. Holding the rows fixed across folds is
what makes them comparable — it isolates differences between the MODELS, which is the
question, rather than confounding them with differences in the rows explained. Each
fold's own imputer and scaler are applied to that matrix, so no fold sees a transform
fit on data it was not trained on.

Agreement is measured BETWEEN folds. The production model is not a reference here: its
importances are already reported in the SHAP figure, and the question is whether the
LODO models agree with each other.

Output: outputs/ranking/shap_fold_stability_summary.csv — one row per fold:
    rank_weight / rank_duration / rank_age  rank of each named feature
    top3_is_the_named_three                 strict set equality, not overlap
    named_three_share_pct                   share of the fold's total mean |SHAP|
    mean_rho / min_rho_vs_other_folds       agreement of the fold's FULL ranking
                                            with the other eight folds'
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from ranking_core import (
    build_base, build_eval_cells, feat_cols_of, ohe_dummy_cols, _ohe,
    _make_model, ABS_TGT, GROUP, WEIGHT,
)

warnings.filterwarnings("ignore")

SCRIPT_DIR  = Path(__file__).parent
REPO_ROOT   = SCRIPT_DIR.parent.parent
RANKING_DIR = REPO_ROOT / "outputs" / "ranking"
LODO_PARAMS = RANKING_DIR / "lodo_per_fold_params.json"
OUT_SUMMARY = RANKING_DIR / "shap_fold_stability_summary.csv"

NAMED = {
    WEIGHT:                                          "weight",
    "preclinical_dosage_duration(days)":             "duration",
    "preclinical_animal_age_before_treatment(days)": "age",
}


def main() -> int:
    explain_rows, feat_cols = load_explanation_matrix()
    by_feature = pd.DataFrame(fold_importances(explain_rows, feat_cols))
    summary = summarize_folds(by_feature)

    summary.to_csv(OUT_SUMMARY, index=False)
    report(by_feature, summary)
    return 0


def load_explanation_matrix():
    """The LODO evaluation universe: the 9 drugs with >=2-candidate clinical arms."""
    cells = build_eval_cells(build_base())
    return cells, feat_cols_of(cells)


def fold_importances(explain_rows, feat_cols: list[str]) -> dict[str, pd.Series]:
    """One model per held-out drug, rebuilt as lodo_ranking.py builds it."""
    saved = json.loads(LODO_PARAMS.read_text())
    family = saved["model_family"]
    fold_params = {drug: saved["folds"][drug]["params"] for drug in saved["folds"]}
    ohe_cols = ohe_dummy_cols(feat_cols)

    importances = {}
    for drug in sorted(fold_params):
        train_rows = explain_rows[explain_rows[GROUP] != drug]
        # fold-safe one-hot: categories absent from the training drugs are zeroed in
        # the explained matrix too, matching what this model would see at predict time
        Xtr_df, Xexplain_df = _ohe(train_rows[feat_cols],
                                   explain_rows[feat_cols], ohe_cols)
        imputer, scaler = SimpleImputer(strategy="median"), StandardScaler()
        Xtr = scaler.fit_transform(imputer.fit_transform(Xtr_df.values))

        params = dict(fold_params[drug])
        params.setdefault("verbose", -1)                   # no random_state, as in LODO
        model = _make_model(family, params)
        model.fit(Xtr, train_rows[ABS_TGT].values)

        shap_values = shap.TreeExplainer(model).shap_values(
            scaler.transform(imputer.transform(Xexplain_df.values)))
        importances[f"minus_{drug}"] = pd.Series(
            np.abs(shap_values).mean(axis=0), index=feat_cols)
    return importances


def summarize_folds(by_feature: pd.DataFrame) -> pd.DataFrame:
    ranks = by_feature.rank(ascending=False, method="min").astype(int)
    agreement = fold_agreement(by_feature)
    return pd.DataFrame([{
        "fold": fold,
        **{f"rank_{label}": int(ranks.loc[feature, fold])
           for feature, label in NAMED.items()},
        "top3_is_the_named_three": set(by_feature[fold].nlargest(3).index) == set(NAMED),
        "named_three_share_pct": round(
            100 * by_feature[fold][list(NAMED)].sum() / by_feature[fold].sum(), 1),
        "mean_rho_vs_other_folds": round(float(agreement[fold].mean()), 3),
        "min_rho_vs_other_folds":  round(float(agreement[fold].min()), 3),
    } for fold in by_feature.columns])


def fold_agreement(by_feature: pd.DataFrame) -> pd.DataFrame:
    """Spearman between each pair of folds' full feature rankings, self-pairs dropped."""
    folds = by_feature.columns
    rho = pd.DataFrame(
        [[stats.spearmanr(by_feature[a], by_feature[b]).statistic for a in folds]
         for b in folds], index=folds, columns=folds)
    np.fill_diagonal(rho.values, np.nan)
    return rho


def report(by_feature: pd.DataFrame, summary: pd.DataFrame) -> None:
    print("=" * 78)
    print(f"  SHAP feature stability across the {len(by_feature.columns)} LODO folds")
    print("  (each trained on 8 drugs; all explained on the same 1,426-row LODO set)")
    print("=" * 78)

    hdr = (f"  {'fold':20s} {'wt':>3} {'dur':>4} {'age':>4} {'top3=named':>11} "
           f"{'share%':>7} {'mean rho':>9} {'min rho':>8}")
    print("\n" + hdr); print("  " + "-" * (len(hdr) - 2))
    for r in summary.itertuples():
        print(f"  {r.fold:20s} {r.rank_weight:3d} {r.rank_duration:4d} {r.rank_age:4d} "
              f"{str(r.top3_is_the_named_three):>11} {r.named_three_share_pct:7.1f} "
              f"{r.mean_rho_vs_other_folds:9.3f} {r.min_rho_vs_other_folds:8.3f}")

    for label in NAMED.values():
        r = summary[f"rank_{label}"]
        print(f"\n  {label:9s} rank: min={r.min()} median={int(r.median())} max={r.max()}")
    print(f"\n  Folds whose top 3 are exactly the named three: "
          f"{int(summary['top3_is_the_named_three'].sum())}/{len(summary)}")
    print(f"  Between-fold agreement: mean rho {summary['mean_rho_vs_other_folds'].mean():.3f}, "
          f"lowest pair {summary['min_rho_vs_other_folds'].min():.3f}")
    print(f"\nSaved: {OUT_SUMMARY.name}")


if __name__ == "__main__":
    raise SystemExit(main())
