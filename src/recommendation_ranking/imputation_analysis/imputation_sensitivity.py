"""
Imputation sensitivity for the fold-safe LODO translation-gap ranking.
======================================================================

Reruns the leave-one-drug-out (LODO) ranking under three imputations of
the three continuous fields (baseline weight 21.5%, age at start 13.0%, group size
1.3% missing). Every imputer is fit STRICTLY WITHIN EACH TRAINING FOLD (fit on the
training drugs, applied to the held-out drug) — no test information leaks into the
fill — and the published fold-safe per-fold hyperparameters are held fixed so that
only the imputer changes:

  * fold_median    — per-column median of the TRAINING DRUGS of each fold, computed
                     within training fold (scikit-learn SimpleImputer).
                     Reference row: reproduces the fold-safe LODO.
  * mice           — IterativeImputer (MICE, BayesianRidge), fit in each fold.
  * complete_case  — drop the imputed-weight records entirely (no weight imputer);
                     age/size still training-fold-median imputed.

Reported per variant: drug-level recommended gap, % improvement over the random
baseline, and drug-level mean chosen-arm percentile rank (average within drug,
then across drugs — one vote per drug; 100 = best arm always chosen, 50 = chance)
with its between-fold (per-drug) SD/SE. All three columns share the drug-level
aggregation so the point estimate sits at the same level as its SE.

Output: outputs/imputation_analysis/imputation_sensitivity.csv
        One row per variant: the unmatched table, then the matched/paired
        comparison columns (see paired_comparison). Counts are named for what
        they count — n_candidate_pairings (preclinical arm x clinical arm rows,
        the unit the model sees) and n_clinical_arms (the ranking decisions).
"""

import json
import warnings
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer

# ranking_core lives two directories up (src/recommendation/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ranking_core import (
    build_base, feat_cols_of, build_eval_cells,
    ohe_dummy_cols, _ohe,
    aggregate, normalized_ranking_score, _make_model,
    ABS_TGT, GROUP, NCT, EVAL_GROUP,
)

warnings.filterwarnings("ignore")

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR       = Path(__file__).parent
REPO_ROOT        = SCRIPT_DIR.parent.parent.parent
RANKING_DIR      = REPO_ROOT / "outputs" / "ranking"            # ranking pipeline artifacts
LODO_PARAMS_PATH = RANKING_DIR / "lodo_per_fold_params.json"    # fold-safe params
ANALYSIS_DIR     = REPO_ROOT / "outputs" / "ranking" / "imputation_analysis"
OUT_CSV          = ANALYSIS_DIR / "imputation_sensitivity.csv"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

MICE_SEED = 42
REFERENCE_VARIANT = "fold_median"   # the published pipeline


# ══════════════════════════════════════════════════════════════════════════════
# Fold-safe imputers (fit on the training drugs only, applied to the held-out drug)
# ══════════════════════════════════════════════════════════════════════════════

def _make_imputer(method):
    """Returns an UNFITTED imputer. Fold scope is set by the caller, which fits it on
    the training drugs and only transforms the held-out drug (see lodo_foldsafe)."""
    if method == "mice":
        return IterativeImputer(max_iter=10, random_state=MICE_SEED, sample_posterior=False)
    return SimpleImputer(strategy="median")     # per-column, unstratified


# ══════════════════════════════════════════════════════════════════════════════
# Fold-safe LODO ranking with the published fold-safe per-fold params
# ══════════════════════════════════════════════════════════════════════════════

def load_lodo_params():
    d = json.load(open(LODO_PARAMS_PATH))
    return d["model_family"], {drug: d["folds"][drug]["params"] for drug in d["folds"]}


def lodo_foldsafe(cells, feat_cols, family, fold_params, method):
    """cells carries NaN in the imputed fields; the imputer is fit on the training
    drugs of each fold and applied to the held-out drug (no leakage)."""
    y = cells[ABS_TGT].values
    groups = cells[GROUP].values
    ohe_cols = ohe_dummy_cols(feat_cols)
    records = []
    for drug in sorted(np.unique(groups)):
        tr = groups != drug
        te = groups == drug
        # fold-safe one-hot: categories absent from the training drugs -> all-zero
        Xtr_df, Xte_df = _ohe(cells.loc[tr, feat_cols], cells.loc[te, feat_cols], ohe_cols)
        imp = _make_imputer(method)
        Xtr = imp.fit_transform(Xtr_df.values)   # fit on training drugs
        Xte = imp.transform(Xte_df.values)       # apply to held-out drug
        params = dict(fold_params[drug])             # tuned with this drug held out
        params.setdefault("verbose", -1)             # NO random_state — LightGBM default seed
        sc = StandardScaler()
        model = _make_model(family, params)
        model.fit(sc.fit_transform(Xtr), y[tr])
        te_df = cells[te].copy()
        te_df["pred"] = model.predict(sc.transform(Xte))
        for cell_id in te_df[EVAL_GROUP].unique():
            rows = te_df[te_df[EVAL_GROUP] == cell_id]
            chosen_gap = float(rows.loc[rows["pred"].idxmin(), ABS_TGT])
            status_quo = float(rows[ABS_TGT].mean())
            records.append({
                "drug": drug, "eval_cell": cell_id,
                # eval_cell is a per-build ngroup() index, so it is NOT comparable across
                # variants (complete_case drops rows and renumbers). cell_key is the
                # stable (NCT x clinical arm) identity used to match variants.
                "cell_key": f"{rows[NCT].iloc[0]}|{rows['clinical_arm_id'].iloc[0]}",
                "status_quo": status_quo, "chosen_gap": chosen_gap,
                "improvement": status_quo - chosen_gap,
                "chosen_pct_rank": normalized_ranking_score(rows[ABS_TGT].values, chosen_gap),
            })
    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

VARIANTS = [
    ("fold_median", False, "Median of the training drugs, unstratified — pipeline method"),
    ("mice",        False, "MICE / IterativeImputer, fit on the training drugs"),
    ("fold_median", True,  "Complete-case (measured weight only)"),
]


def paired_comparison(records: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compare the variants on the drugs and clinical arms evaluable under all of them.

    Clinical arms are matched on cell_key, the stable (NCT x clinical arm) identity;
    eval_cell is a per-build ngroup() index and renumbers when rows are dropped.

    Returns one row per variant: the matched drug-level mean normalized ranking score
    (ranking_core.normalized_ranking_score, carried as chosen_pct_rank), its mean
    per-drug difference from REFERENCE_VARIANT, and the paired SE of that difference.
    """
    shared_drugs = set.intersection(*(set(r["drug"]) for r in records.values()))
    shared_cells = set.intersection(*(set(r["cell_key"]) for r in records.values()))
    matched = {
        name: (rec[rec["drug"].isin(shared_drugs) & rec["cell_key"].isin(shared_cells)]
               .groupby("drug")["chosen_pct_rank"].mean())
        for name, rec in records.items()
    }

    reference = REFERENCE_VARIANT
    rows = []
    for name, pct_rank in matched.items():
        delta = matched[reference] - pct_rank
        rows.append({
            "variant":            name,
            "n_drugs_matched":    len(pct_rank),
            "n_clinical_arms_matched": len(shared_cells),
            "matched_drug_pct_rank": round(float(pct_rank.mean()), 1),
            "delta_vs_reference": round(float(delta.mean()), 1),
            "paired_se":          round(float(delta.std(ddof=1) / np.sqrt(len(delta))), 1)
                                  if name != reference else 0.0,
        })
    return pd.DataFrame(rows)


def main():
    print("=" * 80)
    print("  Imputation sensitivity — FOLD-SAFE imputation (fit within training fold),")
    print("  LODO with published fold-safe per-fold params")
    print("=" * 80)
    family, fold_params = load_lodo_params()
    print(f"  Loaded fold-safe per-fold params ({family}) for {len(fold_params)} folds")

    rows = []
    records = {}                       # variant -> per-cell records, for the paired test
    for method, drop_w, label in VARIANTS:
        cells = build_eval_cells(build_base(drop_weight_missing=drop_w))  # NaN left in-fold
        feat_cols = feat_cols_of(cells)
        rec = lodo_foldsafe(cells, feat_cols, family, fold_params, method)
        records["complete_case" if drop_w else method] = rec
        agg = aggregate(rec)
        # Per-drug mean pct_rank: the drug-level point estimate (drug_mean_pct_rank
        # = fold_pr.mean()) AND the between-fold SD/SE are both built from this, so
        # the reported mean sits at the same aggregation level as its SE.
        fold_pr = rec.groupby("drug")["chosen_pct_rank"].mean()
        rows.append({
            "variant":          "complete_case" if drop_w else method,
            "description":      label,
            # one row per (preclinical arm x clinical arm) candidate pairing — NOT per
            # preclinical arm: an arm linked to k clinical arms appears k times.
            "n_candidate_pairings": int(cells.shape[0]),
            "n_clinical_arms":      int(cells[EVAL_GROUP].nunique()),
            # complete_case drops survodutide entirely (100% weight-missing), so it
            # runs 8 folds, not 9 — fold_pct_rank_se is over THIS many drugs.
            "n_drugs":          int(len(fold_pr)),
            "arm_chosen_gap":   round(agg["cell_mean_chosen_gap"], 2),
            "drug_chosen_gap":  round(agg["drug_mean_chosen_gap"], 2),
            "drug_pct_impr":    round(agg["drug_pct_improvement"], 1),
            "drug_pct_rank":    round(agg["drug_mean_pct_rank"], 1),
            "fold_pct_rank_sd": round(float(fold_pr.std(ddof=1)), 1),
            "fold_pct_rank_se": round(float(fold_pr.std(ddof=1) / np.sqrt(len(fold_pr))), 1),
        })

    out = pd.DataFrame(rows).merge(paired_comparison(records), on="variant", how="left")
    out.to_csv(OUT_CSV, index=False)

    hdr = (f"{'variant':16s} {'pairs':>6} {'clin_arms':>9} {'drugs':>5} | {'arm_gap':>7} "
           f"{'drug_gap':>8} {'impr':>5} | {'pct_rank':>8} {'fold_SE':>7}")
    print("\n" + "-" * len(hdr)); print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['variant']:16s} {r['n_candidate_pairings']:6d} {r['n_clinical_arms']:9d} "
              f"{r['n_drugs']:5d} | {r['arm_chosen_gap']:7.2f} {r['drug_chosen_gap']:8.2f} "
              f"{r['drug_pct_impr']:4.0f}% | "
              f"{r['drug_pct_rank']:8.1f} {r['fold_pct_rank_se']:7.1f}")
    print("-" * len(hdr))
    print("(every imputer is fit on the TRAINING drugs of each fold and only applied to")
    print(" the held-out drug; fold_median row reproduces the fold-safe LODO: 5.86 / 6.61.")
    print(" pct_rank is drug-level: 100 = best arm always chosen, 50 = chance; fold_SE is")
    print(" over the n_drugs folds — complete_case has 8, not 9: survodutide is 100% weight-missing.)")
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
