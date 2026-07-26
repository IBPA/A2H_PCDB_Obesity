"""
Which preclinical design did the LODO model select for liraglutide and
semaglutide?
==========================================================================
Reproduces the LODO model's per-cell recommendation (using the fold-safe
per-fold params in outputs/lodo_per_fold_params.json) and, for each drug, reports one
row per selected design with:
  * how often it was the per-cell top pick (selection frequency),
  * its mean PREDICTED translation gap across the drug's clinical arms — the row with
    the lowest value is the arm quoted in the Results ("...from PMC____..."),
  * its source study (pmcid) and its mean ACTUAL gap when chosen.

Design label = make_design_label (species/strain/model/sex/route/duration/n/age/
weight), built from OBSERVED features ('?' where missing). pmcid is read from the raw
file (build_base drops it) and aligned to the evaluation rows by index.

Output: outputs/selected_design.csv
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from ranking_core import (
    build_base, feat_cols_of, build_eval_cells,
    ohe_dummy_cols, _ohe,
    make_design_label, _make_model,
    RAW_PATH, ABS_TGT, GROUP, EVAL_GROUP,
)

DRUGS = ["liraglutide", "semaglutide"]

SCRIPT_DIR       = Path(__file__).parent
REPO_ROOT        = SCRIPT_DIR.parent.parent
FOLDSAFE_DIR     = REPO_ROOT / "outputs" / "ranking"
FOLD_PARAMS_PATH = FOLDSAFE_DIR / "lodo_per_fold_params.json"
OUT_CSV          = FOLDSAFE_DIR / "selected_design.csv"


def predict_holdout(cells, feat_cols, ohe_cols, y, groups, drug, family, fold_params):
    """Fold-safe LODO: fit on the other drugs, predict this drug's candidate arms.
    Returns the held-out rows with a 'pred' column (predicted absolute gap)."""
    tr, te = groups != drug, groups == drug
    Xtr_df, Xte_df = _ohe(cells.loc[tr, feat_cols], cells.loc[te, feat_cols], ohe_cols)
    imp = SimpleImputer(strategy="median")      # median of the training drugs only
    Xtr, Xte = imp.fit_transform(Xtr_df.values), imp.transform(Xte_df.values)
    params = dict(fold_params[drug]); params.setdefault("verbose", -1)   # no random_state
    sc = StandardScaler()
    model = _make_model(family, params); model.fit(sc.fit_transform(Xtr), y[tr])
    te_df = cells.loc[te].copy()
    te_df["pred"] = model.predict(sc.transform(Xte))
    return te_df


def summarize_designs(te_df, drug):
    """One row per selected design: selection frequency, mean PREDICTED gap over all
    cells (lowest row = the arm quoted in the Results), source study, and mean actual
    gap when chosen. Sorted by mean predicted gap so the top-ranked design is first."""
    n_cells = te_df[EVAL_GROUP].nunique()
    picks = te_df.loc[te_df.groupby(EVAL_GROUP)["pred"].idxmin()]
    chosen = picks.groupby("design_id").agg(
        n_cells_selected=("design_id", "size"),
        mean_chosen_gap=(ABS_TGT, "mean"))
    predicted = te_df.groupby("design_id").agg(
        mean_pred_gap=("pred", "mean"),
        pmcid=("pmcid", lambda s: ", ".join(sorted(s.astype(str).unique()))))
    out = chosen.join(predicted).reset_index()
    out.insert(0, "drug", drug)
    out["n_cells_total"]   = n_cells
    out["selection_frac"]  = (out["n_cells_selected"] / n_cells).round(2)
    out["mean_pred_gap"]   = out["mean_pred_gap"].round(3)
    out["mean_chosen_gap"] = out["mean_chosen_gap"].round(2)
    return out.sort_values("mean_pred_gap")


def main():
    fp = json.load(open(FOLD_PARAMS_PATH))
    family = fp["model_family"]
    fold_params = {d: fp["folds"][d]["params"] for d in fp["folds"]}

    cells = build_eval_cells(build_base()).copy()
    feat_cols = feat_cols_of(cells)
    cells["design_id"] = cells.apply(make_design_label, axis=1)     # from observed features
    raw = pd.read_csv(RAW_PATH)                                     # pmcid dropped by build_base
    cells["pmcid"] = raw.loc[cells.index, "pmcid"].values

    y = cells[ABS_TGT].values
    groups = cells[GROUP].values
    ohe_cols = ohe_dummy_cols(feat_cols)

    tables = []
    for drug in DRUGS:
        te_df = predict_holdout(cells, feat_cols, ohe_cols, y, groups, drug, family, fold_params)
        tables.append(summarize_designs(te_df, drug))

    cols = ["drug", "design_id", "pmcid", "n_cells_selected", "n_cells_total",
            "selection_frac", "mean_pred_gap", "mean_chosen_gap"]
    out = pd.concat(tables, ignore_index=True)[cols]
    out.to_csv(OUT_CSV, index=False)

    for drug in DRUGS:
        sub = out[out["drug"] == drug]
        top = sub.iloc[0]                                           # lowest mean_pred_gap
        print(f"\n=== {drug}  ({sub['n_cells_total'].iloc[0]} clinical arms) ===")
        print(sub[["design_id", "pmcid", "n_cells_selected", "selection_frac",
                   "mean_pred_gap", "mean_chosen_gap"]].to_string(index=False))
        print(f"  -> lowest mean predicted gap: {top['design_id']}  (from {top['pmcid']})")
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
