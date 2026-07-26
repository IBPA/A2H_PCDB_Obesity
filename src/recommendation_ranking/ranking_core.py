"""
ranking_core.py — shared core utilities for the pipeline.
==============================================================================

Provides
  constants : ABS_TGT, GROUP, NCT, EVAL_GROUP, RANDOM_STATE
  models    : _make_model, _DEFAULT_PARAMS, _suggest_params
  ranking   : normalized_ranking_score, aggregate, per_drug_summary
  tuning    : lono_tune_mae
  labels    : make_design_label
"""

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# ── Column constants ──────────────────────────────────────────────────────────
ABS_TGT      = "abs_gap"
GROUP        = "intervention"
NCT          = "NCT Number"
EVAL_GROUP   = "eval_group"          # (NCT × clinical arm) cell identifier
RANDOM_STATE = 42

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT  = SCRIPT_DIR.parent.parent
RAW_PATH   = REPO_ROOT / "data" / "obesity" / "obesity_a2h_dataset.csv"

# Imputation targets no pre-specified column list: the median imputer is fit on the
# whole training-fold feature matrix and fills whatever is NaN. WEIGHT is named only
# because build_base can drop its missing rows for the complete-case sensitivity run.
WEIGHT = "preclinical_animal_weight_before_treatment(grams)"   # top SHAP feature

DROP_COLS = [
    "preclinical_arm_id", "pmcid", "clinical_outcome", "preclinical_outcome",
    "preclinical_animal_weight_before_experiment(grams)", "clinical_total_doses",
    "preclinical_total_doses", "clinical_dose_escalation",
    "preclinical_animal_age_before_experiment(days)",
]
ORAL_VARIANTS = {"gastric gavage", "gavage", "intragastric", "oral", "oral gavage"}
KM_FACTORS = {"mice": 0.081, "hamsters": 0.135, "rats": 0.162, "cynomolgus monkeys": 0.324}
HUMAN_WEIGHT_KG = 100

# The 8 categorical variables one-hot encoded by build_base. Named so the fold-safe
# encoder (_ohe) can identify the resulting dummy columns.
CAT_COLS = [
    "preclinical_administration_route", "preclinical_animal_strain",
    "preclinical_animal_species", "preclinical_animal_sex", "preclinical_disease_model",
    "clinical_age_groups", "clinical_phases", "clinical_administration_route",
]


def _norm_route(v):
    if pd.isna(v):
        return v
    return "oral" if str(v).strip().lower() in ORAL_VARIANTS else str(v).strip().lower()

def _consolidate_disease(v):
    if pd.isna(v):
        return v
    s = str(v).strip().lower()
    if s.startswith("genetic + diet") or s == "genetic+diet":
        return "genetic+diet"
    if s.startswith("genetic"):
        return "genetic"
    return "diet-induced"

def _consolidate_strain(v):
    if pd.isna(v):
        return v
    s = str(v).strip()
    for pat, out in [(r"(?i)C57BL/6", "C57BL/6"), (r"(?i)Wistar", "Wistar"),
                     (r"(?i)Sprague", "Sprague-Dawley")]:
        if re.match(pat, s):
            return out
    return s

def _parse_interval(fs):
    """Dosing-frequency string "N + every M <unit>" -> the dosing INTERVAL M, in days.
    """
    if pd.isna(fs):
        return np.nan
    m = re.search(r"every\s+(\d+(?:\.\d+)?)\s+day", str(fs), re.IGNORECASE)
    return float(m.group(1)) if m else np.nan


def build_base(drop_weight_missing=False):
    df = pd.read_csv(RAW_PATH)
    if drop_weight_missing:
        df = df[df[WEIGHT].notna()].copy()
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    for col in ["preclinical_administration_route", "clinical_administration_route"]:
        df[col] = df[col].map(_norm_route)
    df["preclinical_disease_model"] = df["preclinical_disease_model"].map(_consolidate_disease)
    df["preclinical_animal_strain"] = df["preclinical_animal_strain"].map(_consolidate_strain)

    km = df["preclinical_animal_species"].str.lower().map(KM_FACTORS)
    df["dose_translation_ratio"] = (
        df["clinical_dosage_amount_value(mg)"]
        / (df["preclinical_dosage_amount_value(mg/kg)"] * km * HUMAN_WEIGHT_KG))
    df["cumulative_dose_translation_ratio"] = (
        df["clinical_total_dosage_amount(mg)"]
        / (df["preclinical_total_dosage_amount(mg/kg)"] * km * HUMAN_WEIGHT_KG))
    df.drop(columns=[c for c in [
        "preclinical_dosage_amount_value(mg/kg)", "clinical_dosage_amount_value(mg)",
        "preclinical_total_dosage_amount(mg/kg)", "clinical_total_dosage_amount(mg)"]
        if c in df.columns], inplace=True)

    df["duration_ratio"] = df["clinical_dosage_duration(days)"] / df["preclinical_dosage_duration(days)"]
    df.drop(columns=["clinical_dosage_duration(days)"], inplace=True)
    # frequency_ratio: clinical dosing frequency relative to preclinical, from the
    # dosing intervals (1 = matched; <1 = dosed less often in the clinic).
    pre_i = df["preclinical_dosage_frequency"].map(_parse_interval)
    clin_i = df["clinical_dosage_frequency"].map(_parse_interval)
    df["frequency_ratio"] = (7.0 / clin_i) / (7.0 / pre_i)
    df.drop(columns=["preclinical_dosage_frequency", "clinical_dosage_frequency"], inplace=True)
    df["is_route_match"] = (
        df["preclinical_administration_route"] == df["clinical_administration_route"]).astype(int)

    ohe = [c for c in CAT_COLS if c in df.columns]
    keep = {k: df[k].copy() for k in ["intervention", "translation_outcome"]}
    df.drop(columns=["intervention", "translation_outcome"], inplace=True)
    df = pd.get_dummies(df, columns=ohe, drop_first=False, dtype=int)
    for k, v in keep.items():
        df[k] = v.values

    return df


def feat_cols_of(df):
    bridge = ["dose_translation_ratio", "cumulative_dose_translation_ratio",
              "duration_ratio", "frequency_ratio", "is_route_match"]
    clinical = [c for c in df.columns if c.startswith("clinical_") and c != "clinical_arm_id"]
    preclinical = [c for c in df.columns if c.startswith("preclinical_")]
    return preclinical + bridge + clinical


def ohe_dummy_cols(feat_cols):
    """The one-hot dummy columns among feat_cols (produced by build_base's get_dummies
    over CAT_COLS), named '<categorical>_<value>'."""
    return [c for c in feat_cols if any(c.startswith(p + "_") for p in CAT_COLS)]


def _ohe(Xtr, Xte, ohe_cols):
    """Fold-safe one-hot encoding on the fixed codebook: zero out, in the held-out
    matrix Xte, every one-hot column whose category is ABSENT from the training fold
    (all-zero in Xtr). The model therefore only ever sees categories present in its
    own training data — the same guarantee as OneHotEncoder(handle_unknown='ignore')
    fit on the training fold — while the column codebook stays fixed across folds, so
    feature-subsampling models (LightGBM/XGBoost colsample_bytree) are not perturbed by
    a changing column count. Missing categoricals are already all-zero from get_dummies.
    Xtr/Xte are DataFrames over the same feature columns; returns (Xtr, Xte) with Xte's
    held-out-only categories zeroed (Xtr unchanged — its "unseen" columns are already
    all-zero by definition)."""
    unseen = [c for c in ohe_cols if (Xtr[c] == 0).all()]
    if unseen:
        Xte = Xte.copy()
        Xte.loc[:, unseen] = 0
    return Xtr, Xte


def build_eval_cells(df, rankable_only=True):
    """Attach the target (ABS_TGT) and the (NCT x clinical arm) cell id (EVAL_GROUP).
    rankable_only=True (default) keeps only cells with >=2 candidate preclinical arms —
    the ranking-evaluation set (drops single-candidate cells, i.e. the 9-drug set).
    rankable_only=False keeps EVERY arm — the complete dataset (all 11 drugs, including
    single-candidate clinical arms), as used to train the production recommender."""
    df = df.copy()
    df[ABS_TGT] = df["translation_outcome"].abs()
    df[EVAL_GROUP] = df.groupby([NCT, "clinical_arm_id"]).ngroup().astype(str)
    if not rankable_only:
        return df                                      # full data: no >=2-candidate filter
    sizes = df.groupby(EVAL_GROUP).size()
    rankable = sizes[sizes >= 2].index                 # >=2 preclinical options -> a real choice
    return df[df[EVAL_GROUP].isin(rankable)].copy()


# ══════════════════════════════════════════════════════════════════════════════
# Ranking metric
# ══════════════════════════════════════════════════════════════════════════════

def normalized_ranking_score(gap_values, chosen_gap):
    """
    Normalized rank score of chosen_gap among gap_values (higher = better).
    score = 100 × (n − k) / (n − 1), where k is the mid-rank of chosen_gap
    (k=1 = smallest gap = best arm).
    Ties are resolved by mid-rank so that random selection yields exactly 50%
    for any pool size n ≥ 2.
    100% = best arm chosen, 50% = random expectation, 0% = worst arm chosen.
    """
    n = len(gap_values)
    if n == 1:
        return 100.0
    n_better = sum(1 for g in gap_values if g < chosen_gap)
    n_equal  = sum(1 for g in gap_values if g == chosen_gap)
    k = n_better + (n_equal + 1) / 2
    return 100.0 * (n - k) / (n - 1)


def aggregate(records_df):
    """
    Two aggregation levels:
    - cell_level:  each (NCT × clinical arm) cell gets equal weight.
                   Dominated by data-rich drugs (semaglutide 17 cells, liraglutide 19).
    - drug_level:  average per drug first, then average across drugs.
                   Each drug gets one vote; robust to unequal cell counts.
                   % improvement uses ratio-of-means (mean drug improvement /
                   mean drug baseline) so it chains with the reported baseline
                   and gap — NOT a mean-of-per-drug-ratios (which does not chain).
    """
    pct_rank = records_df["chosen_pct_rank"].mean() \
        if "chosen_pct_rank" in records_df.columns else float("nan")

    # Drug-level: mean per drug, then mean across drugs
    drug_means = records_df.groupby("drug").agg(
        n_cells=("eval_cell", "nunique"),
        status_quo=("status_quo", "mean"),
        chosen_gap=("chosen_gap", "mean"),
        improvement=("improvement", "mean"),
        pct_rank=("chosen_pct_rank", "mean"),
    )
    # Ratio-of-means: mean drug improvement / mean drug baseline. Chains exactly
    # with drug_mean_status_quo (baseline) and drug_mean_chosen_gap (gap):
    #   drug_pct_improvement == 100 * (drug_mean_status_quo - drug_mean_chosen_gap)
    #                           / drug_mean_status_quo
    drug_pct_imp = (100 * drug_means["improvement"].mean()
                    / drug_means["status_quo"].mean())
    weighted_pct_rank = float(np.average(drug_means["pct_rank"],
                                         weights=drug_means["n_cells"]))

    return {
        # Cell-level (current behaviour)
        "cell_mean_status_quo":      records_df["status_quo"].mean(),
        "cell_mean_chosen_gap":      records_df["chosen_gap"].mean(),
        "cell_pct_improvement":      100 * records_df["improvement"].mean()
                                     / records_df["status_quo"].mean(),
        "cell_mean_pct_rank":        pct_rank,
        # Drug-level (equal weight per drug)
        "drug_mean_status_quo":      drug_means["status_quo"].mean(),
        "drug_mean_chosen_gap":      drug_means["chosen_gap"].mean(),
        "drug_pct_improvement":      drug_pct_imp,
        "drug_mean_pct_rank":        drug_means["pct_rank"].mean(),
        # Drug-level (weighted by n_cells — downweights sparse drugs)
        "drug_weighted_pct_rank":    weighted_pct_rank,
        "n_drugs":                   len(drug_means),
    }


def per_drug_summary(lodo_df):
    """Aggregate LODO results by drug (over eval cells)."""
    rows = []
    for drug, grp in lodo_df.groupby("drug"):
        evaluable = grp[grp["n_preclinical"] >= 3]
        sig = (evaluable["spearman_p"] < 0.05).sum() if len(evaluable) > 0 else 0
        # Unique preclinical studies per drug: all NCTs share the same preclinical pool,
        # so the pool size is constant across cells. Use the mode (most common value).
        n_preclinical_total = int(grp["n_preclinical"].mode().iloc[0])
        rows.append({
            "drug":              drug,
            "n_cells":           grp["eval_cell"].nunique(),
            "n_ncts":            grp["NCT"].nunique(),
            "n_preclinical_tot": int(n_preclinical_total),
            "n_cells_eval":      len(evaluable),
            "n_sig":          int(sig),
            "status_quo":     round(grp["status_quo"].mean(), 2),
            "chosen_gap":     round(grp["chosen_gap"].mean(), 2),
            "improvement":    round(grp["improvement"].mean(), 2),
            "pct_improvement":round(100 * grp["improvement"].mean()
                                    / grp["status_quo"].mean(), 1),
            "mean_pct_rank":  round(grp["chosen_pct_rank"].mean(), 1),
            "mean_rho":       round(evaluable["spearman_rho"].mean(), 3)
                              if len(evaluable) > 0 else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("pct_improvement", ascending=False)


# ══════════════════════════════════════════════════════════════════════════════
# Models
# ══════════════════════════════════════════════════════════════════════════════

def _make_model(family, params):
    if family == "LightGBM":
        return lgb.LGBMRegressor(**params)
    elif family == "XGBoost":
        return xgb.XGBRegressor(**params)
    elif family == "RandomForest":
        return RandomForestRegressor(**{k: v for k, v in params.items()
                                        if k != "n_jobs"}, n_jobs=-1)
    else:
        return lgb.LGBMRegressor(**params)


_DEFAULT_PARAMS = {
    "LightGBM": {
        "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "verbose": -1, "random_state": RANDOM_STATE,
    },
    "XGBoost": {
        "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "verbosity": 0, "random_state": RANDOM_STATE,
    },
    "RandomForest": {
        "n_estimators": 300, "max_depth": 8,
        "random_state": RANDOM_STATE,
    },
}


def _suggest_params(trial, family):
    """Optuna parameter suggestions per family."""
    if family == "LightGBM":
        return {
            "n_estimators":    trial.suggest_int("n_estimators", 100, 800),
            "max_depth":       trial.suggest_int("max_depth", 3, 10),
            "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample":       trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":       trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":      trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "verbose": -1, "random_state": RANDOM_STATE,
        }
    elif family == "XGBoost":
        return {
            "n_estimators":    trial.suggest_int("n_estimators", 100, 800),
            "max_depth":       trial.suggest_int("max_depth", 3, 10),
            "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample":       trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":       trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":      trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "verbosity": 0, "random_state": RANDOM_STATE,
        }
    else:  # RandomForest
        return {
            "n_estimators":trial.suggest_int("n_estimators", 100, 600),
            "max_depth":   trial.suggest_int("max_depth", 3, 15),
            "max_features":trial.suggest_float("max_features", 0.3, 1.0),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "random_state": RANDOM_STATE,
        }


def lono_tune_mae(df, feat_cols, family, n_trials=60, holdout_group=None):
    """
    Tune the winning model family using Leave-One-NCT-Out inner CV.
    Objective: minimize mean MAE on abs_gap across all NCT folds.
    No same-drug leakage: LONO holds out one trial at a time.

    holdout_group: the LODO outer held-out drug. If given, this asserts the tuning
    frame `df` contains ZERO rows from that drug. Because every Optuna trial's inner
    LONO fold is a subset of `df`, that single guard proves no held-out-drug row can
    enter ANY trial — the nested-CV boundary is enforced here, at the innermost point.
    Returns (best_params, best_value).
    """
    if holdout_group is not None:
        leaked = int((df[GROUP] == holdout_group).sum())
        assert leaked == 0, (
            f"NESTED-CV LEAKAGE: {leaked} row(s) from the LODO held-out drug "
            f"'{holdout_group}' are present in the Optuna tuning frame — the inner LONO "
            f"tuning must run strictly inside the outer 8-drug training fold.")

    ncts = sorted(df[NCT].unique())
    X = df[feat_cols].values
    y = df[ABS_TGT].values
    nct_arr = df[NCT].values

    def objective(trial):
        params = _suggest_params(trial, family)
        maes = []
        for nct_id in ncts:
            tr = nct_arr != nct_id
            te = nct_arr == nct_id
            # inner LONO is a proper partition of the training frame only
            assert not (tr & te).any() and (tr | te).all()
            sc = StandardScaler()
            m  = _make_model(family, params)
            m.fit(sc.fit_transform(X[tr]), y[tr])
            pred = m.predict(sc.transform(X[te]))
            maes.append(mean_absolute_error(y[te], pred))
        return float(np.mean(maes))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    print(f"  LONO-tuned {family}: MAE={study.best_value:.3f}  params={best}")
    return best, study.best_value


# ══════════════════════════════════════════════════════════════════════════════
# Design label  (migrated verbatim from bootstrap_rank_stability.py)
# ══════════════════════════════════════════════════════════════════════════════

def _decode_onehot(row: pd.Series, prefix: str, abbrevs: dict = None) -> str:
    """Return the category name for a one-hot encoded group, or '?' if none set."""
    cols = {c: c[len(prefix):] for c in row.index if c.startswith(prefix)}
    for col, label in cols.items():
        if row[col] == 1:
            return abbrevs.get(label, label) if abbrevs else label
    return "?"


ROUTE_ABBREV = {
    "subcutaneous": "SC",
    "intraperitoneal": "IP",
    "oral": "oral",
    "unknown": "?route",
}
MODEL_ABBREV = {
    "diet-induced": "DIO",
    "genetic": "genetic",
    "genetic+diet": "genetic+DIO",
}
STRAIN_ABBREV = {
    "C57BL/6": "C57BL6",
    "Sprague-Dawley": "SD",
    "DIO-prone Sprague–Dawley": "DIO-SD",
    "Unknown": "?strain",
}


def make_design_label(row: pd.Series) -> str:
    """
    Build a human-readable label from one row of preclinical features.
    All seven biological dimensions are encoded so the label is always unique.
    Format: {species}/{strain}/{model}/{sex}/{route}/{duration}d/n={N}/age={A}d/w={W}g
    Example: mice/C57BL6/DIO/male/SC/28d/n=10/age=161d/w=45g
    """
    species = _decode_onehot(row, "preclinical_animal_species_")
    strain  = _decode_onehot(row, "preclinical_animal_strain_", STRAIN_ABBREV)
    model   = _decode_onehot(row, "preclinical_disease_model_", MODEL_ABBREV)
    sex     = _decode_onehot(row, "preclinical_animal_sex_")
    route   = _decode_onehot(row, "preclinical_administration_route_", ROUTE_ABBREV)
    dur     = int(row["preclinical_dosage_duration(days)"])
    n       = int(row["preclinical_animal_subject_size"]) if not pd.isna(row["preclinical_animal_subject_size"]) else "?"
    age_val = row["preclinical_animal_age_before_treatment(days)"]
    wt_val  = row["preclinical_animal_weight_before_treatment(grams)"]
    age     = int(age_val) if not pd.isna(age_val) else "?"
    wt      = int(wt_val)  if not pd.isna(wt_val)  else "?"
    return f"{species}/{strain}/{model}/{sex}/{route}/{dur}d/n={n}/age={age}d/w={wt}g"
