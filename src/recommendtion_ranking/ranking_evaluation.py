"""
Ranking Evaluation for Preclinical Design Recommender
======================================================
Implements two fixes to the original evaluation:

Fix 1 — Weak baseline:
  - Bootstrap random-pick baseline with 95% CI
  - Multiple rule-based baselines grounded in translational biology

Fix 2 — Same-drug leakage:
  - LONO ranking (model has seen other NCTs of same drug)
  - LODO ranking (model has never seen any data from held-out drug)
  Both evaluate on the SAME metric: actual |gap| of top-1 ranked arm,
  chosen from arms ACTUALLY RUN in the held-out NCT.

Output: app/results/ranking_evaluation/
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
REPO_ROOT   = SCRIPT_DIR.parent.parent
DATA_PATH   = REPO_ROOT / "data" / "ml_ready_obesity_dataset.csv"
RESULTS_DIR = REPO_ROOT / "outputs" / "ranking_evaluation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PARAMS_PATH       = REPO_ROOT / "outputs" / "recommender_model_params.json"
LODO_PARAMS_PATH  = REPO_ROOT / "outputs" / "lodo_per_fold_params.json"

TARGET   = "translation_outcome"
ABS_TGT  = "abs_gap"
GROUP    = "intervention"
NCT      = "NCT Number"
RANDOM_STATE = 42
N_BOOTSTRAP  = 1000


# ══════════════════════════════════════════════════════════════════════════════
# 1. Data Loading
# ══════════════════════════════════════════════════════════════════════════════

EVAL_GROUP = "eval_group"   # (NCT × clinical arm) cell identifier


def load_data():
    df = pd.read_csv(DATA_PATH)
    df[ABS_TGT] = df[TARGET].abs()

    bridge_cols = [
        "dose_translation_ratio", "cumulative_dose_translation_ratio",
        "duration_ratio", "frequency_ratio", "is_route_match",
    ]
    clinical_cols  = [c for c in df.columns if c.startswith("clinical_")
                      and c != "clinical_arm_id"]
    preclinical_cols = [c for c in df.columns if c.startswith("preclinical_")]
    feat_cols = preclinical_cols + bridge_cols + clinical_cols

    # Each unique clinical arm (identified by clinical_arm_id) within an NCT is one
    # evaluation cell. Using clinical_arm_id directly avoids merging arms that differ
    # only in dose/duration (dropped during preprocessing into bridge features).
    df[EVAL_GROUP] = (df.groupby([NCT, "clinical_arm_id"])
                        .ngroup()
                        .astype(str))

    # Drop eval cells with only 1 preclinical option — no ranking decision possible
    cell_sizes = df.groupby(EVAL_GROUP).size()
    rankable_cells = cell_sizes[cell_sizes >= 2].index
    n_dropped_cells = cell_sizes[cell_sizes < 2].shape[0]
    dropped_drugs = df[~df[EVAL_GROUP].isin(rankable_cells)][GROUP].unique()
    df = df[df[EVAL_GROUP].isin(rankable_cells)].copy()
    if n_dropped_cells:
        print(f"Dropped {n_dropped_cells} unrankable cell(s) (only 1 preclinical option): "
              f"{sorted(dropped_drugs)}")

    n_cells = df[EVAL_GROUP].nunique()
    pre_per_cell = df.groupby(EVAL_GROUP).size()
    print(f"Loaded: {df.shape[0]} rows | {df[GROUP].nunique()} drugs | "
          f"{df[NCT].nunique()} NCTs | {n_cells} eval cells (NCT × clinical arm)")
    print(f"Preclinical studies per cell: min={pre_per_cell.min()} "
          f"median={pre_per_cell.median():.0f} max={pre_per_cell.max()}")
    return df, feat_cols


# ══════════════════════════════════════════════════════════════════════════════
# 2. Core Evaluation Helper
# ══════════════════════════════════════════════════════════════════════════════

def _arm_count_stratum(n):
    if n <= 5:   return "small (1-5)"
    if n <= 20:  return "medium (6-20)"
    return "large (21+)"


def _pct_rank(gap_values, chosen_gap):
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


def _eval_picker(df, pick_fn, name):
    """
    For each (NCT × clinical arm) eval cell, apply pick_fn(cell_rows) -> index of chosen row.
    Ranking is over preclinical designs only; clinical context is fixed within each cell.
    chosen_pct_rank: 100=best, 50=random expectation, 0=worst.
    """
    records = []
    for cell_id in sorted(df[EVAL_GROUP].unique()):
        rows = df[df[EVAL_GROUP] == cell_id].copy()
        if len(rows) == 0:
            continue
        chosen_idx  = pick_fn(rows)
        chosen_gap  = float(rows.loc[chosen_idx, ABS_TGT])
        status_quo  = float(rows[ABS_TGT].mean())
        pct_rank    = _pct_rank(rows[ABS_TGT].values, chosen_gap)
        records.append({
            "eval_cell":       cell_id,
            "NCT":             rows[NCT].iloc[0],
            "drug":            rows[GROUP].iloc[0],
            "n_preclinical":   len(rows),
            "arm_stratum":     _arm_count_stratum(len(rows)),
            "status_quo":      status_quo,
            "chosen_gap":      chosen_gap,
            "improvement":     status_quo - chosen_gap,
            "chosen_pct_rank": pct_rank,
            "picker":          name,
        })
    return pd.DataFrame(records)


def aggregate(records_df):
    """
    Two aggregation levels:
    - cell_level:  each (NCT × clinical arm) cell gets equal weight.
                   Dominated by data-rich drugs (semaglutide 17 cells, liraglutide 19).
    - drug_level:  average per drug first, then average across drugs.
                   Each drug gets one vote; robust to unequal cell counts.
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
    drug_pct_imp = (100 * drug_means["improvement"] / drug_means["status_quo"]).mean()
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
        "drug_mean_chosen_gap":      drug_means["chosen_gap"].mean(),
        "drug_pct_improvement":      drug_pct_imp,
        "drug_mean_pct_rank":        drug_means["pct_rank"].mean(),
        # Drug-level (weighted by n_cells — downweights sparse drugs)
        "drug_weighted_pct_rank":    weighted_pct_rank,
        "n_drugs":                   len(drug_means),
    }


def stratum_summary(records_df, label=""):
    """Aggregate by preclinical-study-count stratum to check n_preclinical confound."""
    order = ["small (1-5)", "medium (6-20)", "large (21+)"]
    rows = []
    for s in order:
        sub = records_df[records_df["arm_stratum"] == s]
        if len(sub) == 0:
            continue
        n_col = "n_preclinical" if "n_preclinical" in sub.columns else "n_arms"
        rows.append({
            "stratum":          s,
            "n_cells":          len(sub),
            "mean_n_preclin":   round(sub[n_col].mean(), 1),
            "status_quo":       round(sub["status_quo"].mean(), 2),
            "chosen_gap":       round(sub["chosen_gap"].mean(), 2),
            "pct_improvement":  round(100 * sub["improvement"].mean()
                                      / sub["status_quo"].mean(), 1),
            "mean_pct_rank":    round(sub["chosen_pct_rank"].mean(), 1),
        })
    df_out = pd.DataFrame(rows)
    if label:
        print(f"\n  Stratum breakdown [{label}]  "
              "(random expectation: pct_rank=50; higher pct_rank = better)")
        print(df_out.to_string(index=False))
    return df_out


# ══════════════════════════════════════════════════════════════════════════════
# 3. Baselines
# ══════════════════════════════════════════════════════════════════════════════

# --- 3a. Random bootstrap ------------------------------------------------

def bootstrap_random_baseline(df, n_bootstrap=N_BOOTSTRAP, seed=RANDOM_STATE):
    """
    For each bootstrap replicate, pick one preclinical study uniformly at random
    per (NCT × clinical arm) eval cell.
    Returns mean chosen gap ± 95% CI across replicates.
    """
    rng = np.random.default_rng(seed)
    cells = df[EVAL_GROUP].unique()
    replicate_means = []
    for _ in range(n_bootstrap):
        gaps = []
        for cell_id in cells:
            rows = df[df[EVAL_GROUP] == cell_id]
            gaps.append(float(rows[ABS_TGT].sample(1, random_state=None).iloc[0]))
        replicate_means.append(np.mean(gaps))
    arr = np.array(replicate_means)
    return {
        "mean":   float(arr.mean()),
        "ci_lo":  float(np.percentile(arr, 2.5)),
        "ci_hi":  float(np.percentile(arr, 97.5)),
    }


# --- 3b. Rule-based baselines -------------------------------------------
#
# Each rule encodes a domain-motivated hypothesis about which arm
# is most translationally predictive.
#
# Rule                   | Hypothesis
# -----------------------|------------------------------------------------
# duration_ratio→1       | Closest preclinical/clinical duration match
# dose_ratio→1           | Closest dose (mg/kg) match
# cumulative_dose→1      | Closest total exposure match
# route_match            | Same administration route as human trial
# longest_preclinical    | Longer animal study → better chronic validity
# DIO_model              | Diet-induced obesity = most human-like model
# largest_n              | Largest animal group → most stable estimate

def _closest_to_one(rows, col):
    return rows[col].sub(1).abs().idxmin()

def _route_match(rows):
    matched = rows[rows["is_route_match"] == 1]
    pool = matched if len(matched) > 0 else rows
    return pool.index[0]

def _dio_model(rows):
    col = "preclinical_disease_model_diet-induced"
    if col in rows.columns:
        dio = rows[rows[col] == 1]
        pool = dio if len(dio) > 0 else rows
    else:
        pool = rows
    return pool.index[0]

RULES = {
    "rule_duration_ratio→1":    lambda rows: _closest_to_one(rows, "duration_ratio"),
    "rule_dose_ratio→1":        lambda rows: _closest_to_one(rows, "dose_translation_ratio"),
    "rule_cumdose_ratio→1":     lambda rows: _closest_to_one(rows, "cumulative_dose_translation_ratio"),
    "rule_route_match":         _route_match,
    "rule_longest_preclinical": lambda rows: rows["preclinical_dosage_duration(days)"].idxmax(),
    "rule_DIO_model":           _dio_model,
    "rule_largest_n":           lambda rows: rows["preclinical_animal_subject_size"].idxmax(),
}


def run_all_rule_baselines(df):
    results = {}
    for name, fn in RULES.items():
        records = _eval_picker(df, fn, name)
        results[name] = {**aggregate(records), "records": records}
        print(f"  {name:35s}  chosen_gap={results[name]['cell_mean_chosen_gap']:.3f}  "
              f"SQ={results[name]['cell_mean_status_quo']:.3f}  "
              f"Δ(cell)={results[name]['cell_pct_improvement']:+.1f}%  "
              f"Δ(drug)={results[name]['drug_pct_improvement']:+.1f}%")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 4. Model Loading / Training
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


def load_or_default_params():
    """Load saved recommender params; fall back to sensible LightGBM defaults."""
    if PARAMS_PATH.exists():
        with open(PARAMS_PATH) as f:
            saved = json.load(f)
        family = saved.get("model_family", "LightGBM")
        params = saved.get("params", saved)
        print(f"  Loaded params from {PARAMS_PATH}  (family={family})")
    else:
        print("  No saved params found — using default LightGBM")
        family = "LightGBM"
        params = {
            "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "verbose": -1, "random_state": RANDOM_STATE,
        }
    return family, params


# ══════════════════════════════════════════════════════════════════════════════
# 4b. Model Selection — LODO family comparison + LONO-Optuna tuning
# ══════════════════════════════════════════════════════════════════════════════

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


def lodo_mae_benchmark(df, feat_cols):
    """
    Compare LightGBM / XGBoost / RandomForest via LODO cross-validation.
    Primary metric: mean chosen-arm percentile rank (lower = better), computed
    per eval cell and averaged drug-level (each drug one vote) — directly aligned
    with the final evaluation.  MAE is also reported for reference.

    Only rows that belong to rankable eval cells (≥2 preclinical arms per cell)
    are used — drugs with no rankable cells (e.g. naltrexone, phentermine) are
    excluded from both training and test folds.

    Returns a dict: family → {mean_pct_rank, mean_mae, std_mae, per_drug_pct_rank}.
    """
    # Restrict to rankable eval cells only (≥2 preclinical options per cell)
    cell_sizes = df.groupby(EVAL_GROUP).size()
    rankable   = cell_sizes[cell_sizes >= 2].index
    df = df[df[EVAL_GROUP].isin(rankable)].copy()

    X     = df[feat_cols].values
    y     = df[ABS_TGT].values
    drugs = sorted(df[GROUP].unique())
    cells = df[EVAL_GROUP].values
    results = {}

    for family, params in _DEFAULT_PARAMS.items():
        per_drug_pct_rank = []
        per_drug_mae      = []
        for drug in drugs:
            tr = df[GROUP].values != drug
            te = df[GROUP].values == drug
            sc = StandardScaler()
            m  = _make_model(family, params)
            m.fit(sc.fit_transform(X[tr]), y[tr])
            pred = m.predict(sc.transform(X[te]))

            per_drug_mae.append(mean_absolute_error(y[te], pred))

            # Per-cell ranking: for each eval cell in this held-out drug,
            # rank the predicted scores and compute chosen-arm pct_rank
            cells_te    = cells[te]
            cell_pct_ranks = []
            for cell_id in np.unique(cells_te):
                cell_mask   = cells_te == cell_id
                actual_gaps = y[te][cell_mask]
                preds_cell  = pred[cell_mask]
                chosen_idx  = np.argmin(preds_cell)      # model picks lowest predicted gap
                chosen_gap  = actual_gaps[chosen_idx]
                cell_pct_ranks.append(_pct_rank(actual_gaps, chosen_gap))
            per_drug_pct_rank.append(float(np.mean(cell_pct_ranks)))

        results[family] = {
            "mean_pct_rank":    float(np.mean(per_drug_pct_rank)),
            "mean_mae":         float(np.mean(per_drug_mae)),
            "std_mae":          float(np.std(per_drug_mae)),
            "per_drug_pct_rank": per_drug_pct_rank,
        }
        print(f"  {family:15s}  LODO pct_rank = {results[family]['mean_pct_rank']:.1f}%"
              f"  (MAE={results[family]['mean_mae']:.3f} ± {results[family]['std_mae']:.3f})")

    best = max(results, key=lambda f: results[f]["mean_pct_rank"])
    print(f"\n  → Best family: {best}  (pct_rank={results[best]['mean_pct_rank']:.1f}%)")
    return results, best


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


def lono_tune_mae(df, feat_cols, family, n_trials=60):
    """
    Tune the winning model family using Leave-One-NCT-Out inner CV.
    Objective: minimize mean MAE on abs_gap across all 36 NCT folds.
    No same-drug leakage: LONO holds out one trial at a time.
    Returns best_params dict.
    """
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


def run_model_selection(df, feat_cols, n_trials=60):
    """
    Phase 1: LODO family comparison (MAE, default params).
    Phase 2: LONO-Optuna tuning of the winning family.
    Saves results to PARAMS_PATH and returns (family, params).
    """
    print("\n── Model Selection Phase 1: LODO family comparison (MAE) ──")
    benchmark, best_family = lodo_mae_benchmark(df, feat_cols)

    print(f"\n── Model Selection Phase 2: LONO-Optuna tuning ({best_family}, "
          f"{n_trials} trials) ──")
    best_params, best_mae = lono_tune_mae(df, feat_cols, best_family, n_trials)

    # Save for reuse
    out = {"model_family": best_family, "params": best_params,
           "lono_mae": best_mae, "lodo_benchmark": benchmark}
    with open(PARAMS_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved to {PARAMS_PATH}")

    return best_family, best_params


# ══════════════════════════════════════════════════════════════════════════════
# 5. Model-based ranking: LONO (Leave-One-NCT-Out)
# ══════════════════════════════════════════════════════════════════════════════

def lono_ranking(df, feat_cols, family, params):
    """
    For each held-out NCT:
      - Train on ALL other NCTs (all drugs, including same drug)
      - For each (NCT × clinical arm) eval cell, rank preclinical studies by
        predicted |gap| — clinical context is fixed within a cell
      - Pick top-1; record its actual |gap|
    Same-drug NCTs are in training data → optimistic upper bound.
    """
    ncts = sorted(df[NCT].unique())
    records = []
    X = df[feat_cols].values
    y = df[ABS_TGT].values

    for nct_id in ncts:
        te_mask = df[NCT] == nct_id
        tr_mask = ~te_mask
        sc = StandardScaler()
        m  = _make_model(family, params)
        m.fit(sc.fit_transform(X[tr_mask]), y[tr_mask])

        df_te = df[te_mask].copy()
        df_te["pred_abs_gap"] = m.predict(sc.transform(df_te[feat_cols].values))

        # Iterate over clinical arm cells within this NCT
        for cell_id in df_te[EVAL_GROUP].unique():
            rows = df_te[df_te[EVAL_GROUP] == cell_id]
            best_idx   = rows["pred_abs_gap"].idxmin()
            chosen_gap = float(rows.loc[best_idx, ABS_TGT])
            status_quo = float(rows[ABS_TGT].mean())
            n_pre      = len(rows)

            if n_pre > 2:
                rho, pval = stats.spearmanr(
                    rows["pred_abs_gap"].values, rows[ABS_TGT].values)
            else:
                rho, pval = np.nan, np.nan

            records.append({
                "eval_cell":       cell_id,
                "NCT":             nct_id,
                "drug":            rows[GROUP].iloc[0],
                "n_preclinical":   n_pre,
                "arm_stratum":     _arm_count_stratum(n_pre),
                "status_quo":      status_quo,
                "chosen_gap":      chosen_gap,
                "improvement":     status_quo - chosen_gap,
                "chosen_pct_rank": _pct_rank(rows[ABS_TGT].values, chosen_gap),
                "spearman_rho":    rho,
                "spearman_p":      pval,
                "picker":          "LONO_model",
            })
        n_cells = df_te[EVAL_GROUP].nunique()
        print(f"    [{nct_id}] {n_cells} cells  "
              f"mean_SQ={df_te[ABS_TGT].mean():.2f}")

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Model-based ranking: LODO (Leave-One-Drug-Out)
# ══════════════════════════════════════════════════════════════════════════════

def lodo_ranking(df, feat_cols, family, n_trials=60):
    """
    Nested CV: for each held-out drug:
      - Tune hyperparameters via LONO on the remaining drugs' NCTs only
        (no held-out drug data in tuning — eliminates hyperparameter leakage)
      - Train on ALL other drugs with the fold-specific tuned params
      - For each (NCT × clinical arm) eval cell, rank preclinical studies by
        predicted |gap| — clinical context fixed within a cell
      - Pick top-1; record its actual |gap|
    No same-drug leakage — true cross-drug generalization test.
    """
    drugs = sorted(df[GROUP].unique())
    records = []
    per_fold_params = {}
    X = df[feat_cols].values
    y = df[ABS_TGT].values
    groups = df[GROUP].values

    for drug in drugs:
        tr_mask = groups != drug
        te_mask = groups == drug

        # Tune on training drugs only — held-out drug never seen during tuning
        print(f"    [{drug}] tuning hyperparameters on {tr_mask.sum()} training rows ...")
        fold_params, fold_mae = lono_tune_mae(df[tr_mask], feat_cols, family, n_trials=n_trials)
        per_fold_params[drug] = {"params": fold_params, "lono_mae": fold_mae}

        sc = StandardScaler()
        m  = _make_model(family, fold_params)
        m.fit(sc.fit_transform(X[tr_mask]), y[tr_mask])

        df_te = df[te_mask].copy()
        df_te["pred_abs_gap"] = m.predict(sc.transform(df_te[feat_cols].values))

        for cell_id in df_te[EVAL_GROUP].unique():
            rows = df_te[df_te[EVAL_GROUP] == cell_id]
            nct_id     = rows[NCT].iloc[0]
            best_idx   = rows["pred_abs_gap"].idxmin()
            chosen_gap = float(rows.loc[best_idx, ABS_TGT])
            status_quo = float(rows[ABS_TGT].mean())
            n_pre      = len(rows)

            if n_pre > 2:
                rho, pval = stats.spearmanr(
                    rows["pred_abs_gap"].values, rows[ABS_TGT].values)
            else:
                rho, pval = np.nan, np.nan

            records.append({
                "eval_cell":       cell_id,
                "NCT":             nct_id,
                "drug":            drug,
                "n_preclinical":   n_pre,
                "arm_stratum":     _arm_count_stratum(n_pre),
                "status_quo":      status_quo,
                "chosen_gap":      chosen_gap,
                "improvement":     status_quo - chosen_gap,
                "chosen_pct_rank": _pct_rank(rows[ABS_TGT].values, chosen_gap),
                "spearman_rho":    rho,
                "spearman_p":      pval,
                "picker":          "LODO_model",
            })
            print(f"    [{drug}/{nct_id}/cell={cell_id}] n_pre={n_pre:3d}  "
                  f"SQ={status_quo:.2f}  rec={chosen_gap:.2f}  ρ={rho:.3f}")

    with open(LODO_PARAMS_PATH, "w") as f:
        json.dump({"model_family": family, "folds": per_fold_params}, f, indent=2)
    print(f"\n  Per-fold params saved to {LODO_PARAMS_PATH}")

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# 7. Plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_comparison(summary_df, bootstrap, save_path):
    """
    Horizontal bar chart: mean chosen gap for each picker.
    Random bootstrap shown with 95% CI error bar.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = []
    for _, row in summary_df.iterrows():
        if "LONO" in row["picker"]:
            colors.append("#2ecc71")
        elif "LODO" in row["picker"]:
            colors.append("#27ae60")
        elif "rule" in row["picker"]:
            colors.append("#3498db")
        else:
            colors.append("#bdc3c7")

    bars = ax.barh(summary_df["picker"], summary_df["drug_chosen_gap"],
                   color=colors, edgecolor="white")

    # Bootstrap CI as error bar on random baseline
    rand_row = summary_df[summary_df["picker"] == "random_pick"]
    if not rand_row.empty:
        idx = rand_row.index[0]
        y_pos = list(summary_df["picker"]).index("random_pick")
        ax.errorbar(
            x=bootstrap["mean"], y=y_pos,
            xerr=[[bootstrap["mean"] - bootstrap["ci_lo"]],
                  [bootstrap["ci_hi"] - bootstrap["mean"]]],
            fmt="none", color="black", capsize=5, linewidth=2,
            label="Bootstrap 95% CI"
        )

    # Status quo line (same for all)
    sq_mean = summary_df["cell_chosen_gap"].iloc[-1]  # random_pick row = status quo
    ax.axvline(sq_mean, color="red", linestyle="--", linewidth=1.5,
               label=f"Status quo mean = {sq_mean:.2f}")

    for bar, val in zip(bars, summary_df["drug_chosen_gap"]):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=9)

    ax.set_xlabel("Mean actual |gap| of top-1 selected arm (%BW)")
    ax.set_title("Ranking Evaluation: All Baselines vs Model\n"
                 "(lower = better recommendation)", fontweight="bold")
    ax.legend(fontsize=9)

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", label="Model (LONO — same-drug seen)"),
        Patch(facecolor="#27ae60", label="Model (LODO — no same-drug data)"),
        Patch(facecolor="#3498db", label="Rule-based"),
        Patch(facecolor="#bdc3c7", label="Random"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


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


def plot_stratum_analysis(lono_df, lodo_df, save_path):
    """
    Two panels:
    Left:  mean chosen_pct_rank by arm-count stratum for LONO and LODO
           (red dashed line = 50, random expectation)
    Right: scatter n_arms vs improvement coloured by model type
    """
    order = ["small (1-5)", "medium (6-20)", "large (21+)"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: pct_rank by stratum ────────────────────────────────────────────
    ax = axes[0]
    x = np.arange(len(order))
    width = 0.35
    for offset, (df_r, label, color) in enumerate([
        (lono_df, "LONO", "#2ecc71"),
        (lodo_df, "LODO", "#27ae60"),
    ]):
        vals = []
        for s in order:
            sub = df_r[df_r["arm_stratum"] == s]
            vals.append(sub["chosen_pct_rank"].mean() if len(sub) > 0 else float("nan"))
        bars = ax.bar(x + (offset - 0.5) * width, vals, width,
                      label=label, color=color, edgecolor="white")
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.5,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(50, color="red", linestyle="--", lw=1.5, label="Random (50th pct)")
    ax.set_xticks(x); ax.set_xticklabels(order)
    ax.set_ylabel("Mean chosen arm percentile rank\n(higher = better; 50 = random)")
    ax.set_title("Ranking Quality by Arm-Count Stratum\n"
                 "(controls for n_arms confound)", fontweight="bold")
    ax.set_ylim(0, 112)
    ax.legend(fontsize=9)

    # ── Right: n_arms vs improvement scatter ─────────────────────────────────
    ax2 = axes[1]
    for df_r, label, color, marker in [
        (lono_df, "LONO", "#2ecc71", "o"),
        (lodo_df, "LODO", "#27ae60", "s"),
    ]:
        n_col = "n_preclinical" if "n_preclinical" in df_r.columns else "n_arms"
        ax2.scatter(df_r[n_col], df_r["improvement"],
                    label=label, color=color, marker=marker,
                    alpha=0.6, s=40, edgecolors="white")

    ax2.axhline(0, color="black", lw=1, linestyle="--")
    ax2.set_xlabel("Preclinical studies per (NCT × clinical arm) cell")
    ax2.set_ylabel("Improvement (status quo − chosen gap, %BW)")
    ax2.set_title("Does improvement scale with n_arms?\n"
                  "(confound present if strong positive slope)", fontweight="bold")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_lodo_by_drug(lodo_df, save_path):
    """
    Three-panel LODO-by-drug plot.

    Panel 1 — Status Quo vs Recommended |gap|:
      Sorted by % improvement (best left), green bars where model improves,
      red bars where model makes things worse. Cell count annotated on x-axis.

    Panel 2 — pct_rank per drug (debiased headline metric):
      Bubble size = n_cells. Reference line at 50 (random). Text annotations.

    Panel 3 — Mean Spearman ρ:
      Bubble size = n_evaluable cells (cells with ≥3 preclinical arms).
      Drugs with no evaluable cells shown as open diamonds at y=0 with "n/a" label.
    """
    drug_df = per_drug_summary(lodo_df)
    # merge bootstrap CI
    ci_path = Path(save_path).parent / "per_drug_gap_ci.csv"
    if ci_path.exists():
        ci_df = pd.read_csv(ci_path)
        drug_df = drug_df.merge(ci_df[["drug", "sq_lo", "sq_hi", "cg_lo", "cg_hi"]],
                                on="drug", how="left")
    else:
        for col in ["sq_lo", "sq_hi", "cg_lo", "cg_hi"]:
            drug_df[col] = np.nan

    # sort best → worst improvement for all panels
    drug_df = drug_df.sort_values("pct_improvement", ascending=False).reset_index(drop=True)
    drugs = drug_df["drug"].tolist()
    x = np.arange(len(drugs))
    width = 0.32

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("LODO: Per-Drug Recommendation Performance  "
                 "(sorted best → worst improvement)",
                 fontweight="bold", fontsize=12)

    # Flag drugs with binary-only preclinical pool (n=2) — results within noise
    binary_drugs = set(drug_df.loc[drug_df["n_preclinical_tot"] <= 2, "drug"])

    # x-axis labels: add † for binary-choice drugs
    xlabels = [
        f"{d}{'†' if d in binary_drugs else ''}\n"
        f"(pre={r.n_preclinical_tot}, clin.arms={r.n_cells}, trials={r.n_ncts})"
        for d, r in zip(drugs, drug_df.itertuples())
    ]

    # ── Panel 1: Status quo vs recommended gap ────────────────────────────────
    ax = axes[0]
    ax.bar(x - width/2, drug_df["status_quo"], width,
           label="Baseline (avg. all arms)", color="#bdc3c7", edgecolor="white")
    sq_yerr = np.array([
        (drug_df["status_quo"] - drug_df["sq_lo"]).clip(lower=0)
         .where(drug_df["sq_lo"].notna(), other=0).values,
        (drug_df["sq_hi"] - drug_df["status_quo"]).clip(lower=0)
         .where(drug_df["sq_hi"].notna(), other=0).values,
    ])
    ax.errorbar(x - width/2, drug_df["status_quo"], yerr=sq_yerr,
                fmt="none", color="#7f8c8d", capsize=4, linewidth=1.4, capthick=1.4)

    def _bar_color(imp, drug):
        if drug in binary_drugs:
            return "#f39c12"   # amber: binary choice (n=2), result within noise
        return "#27ae60" if imp >= 0 else "#e74c3c"

    rec_colors = [_bar_color(imp, d)
                  for imp, d in zip(drug_df["improvement"], drug_df["drug"])]
    b2 = ax.bar(x + width/2, drug_df["chosen_gap"], width,
                label="Model recommendation (LODO)", color=rec_colors, edgecolor="white")
    cg_yerr = np.array([
        (drug_df["chosen_gap"] - drug_df["cg_lo"]).clip(lower=0)
         .where(drug_df["cg_lo"].notna(), other=0).values,
        (drug_df["cg_hi"] - drug_df["chosen_gap"]).clip(lower=0)
         .where(drug_df["cg_hi"].notna(), other=0).values,
    ])
    ax.errorbar(x + width/2, drug_df["chosen_gap"], yerr=cg_yerr,
                fmt="none", color="#2c3e50", capsize=4, linewidth=1.4, capthick=1.4)

    for bar, row in zip(b2, drug_df.itertuples()):
        h = bar.get_height()
        ci_top = row.cg_hi if not np.isnan(row.cg_hi) else h
        if row.drug in binary_drugs:
            sign, color = "~", "#d68910"
        elif row.improvement >= 0:
            sign, color = "▼", "#1a7a45"
        else:
            sign, color = "▲", "#c0392b"
        ax.text(bar.get_x() + bar.get_width()/2, ci_top + 0.5,
                f"{h:.1f}\n{sign}{abs(row.pct_improvement):.0f}%",
                ha="center", va="bottom", fontsize=6.5, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Mean actual |translation gap| (%BW)")
    ax.set_title("Recommended vs Status-Quo Translation Gap\n"
                 "(lower = closer to zero gap = better preclinical design)",
                 fontweight="bold", fontsize=10)
    ax.text(0.01, -0.22, "† n=2 preclinical arms (binary choice only; results within statistical noise)",
            transform=ax.transAxes, fontsize=7, color="#7f8c8d", va="top")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#bdc3c7", label="Status quo (mean of all arms)"),
        Patch(facecolor="#27ae60", label="Model recommendation — improves"),
        Patch(facecolor="#f39c12", label="Model recommendation — binary choice (†)"),
    ]
    ax.legend(handles=legend_elements, fontsize=7.5, loc="upper left")

    # ── Panel 2: Chosen-arm percentile rank per drug ──────────────────────────
    # pct_rank: 100 = best arm chosen, 50 = equivalent to random, 0 = worst arm chosen
    # Dot color encodes number of evaluation contexts (darker = more data, more reliable).
    ax2 = axes[1]

    n_cells_arr = drug_df["n_cells"].values.astype(float)
    # Blues colormap: darker = more eval contexts = more reliable estimate
    blues = plt.cm.Blues
    # Shift vmin below 0 so n_cells=1 maps to a visible blue rather than near-white
    norm  = plt.Normalize(vmin=-n_cells_arr.max() * 0.4, vmax=n_cells_arr.max())
    dot_colors = blues(norm(n_cells_arr))

    ax2.scatter(x, drug_df["mean_pct_rank"],
                s=180, c=dot_colors,
                edgecolors="#333333", linewidths=0.8, zorder=3)

    ax2.axhline(50, color="#e74c3c", linestyle="--", lw=1.5,
                label="50 — random pick baseline")

    for xi, row in zip(x, drug_df.itertuples()):
        ax2.annotate(f"{row.mean_pct_rank:.0f}",
                     (xi, row.mean_pct_rank),
                     textcoords="offset points", xytext=(0, 10),
                     ha="center", fontsize=8, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(xlabels, rotation=40, ha="right", fontsize=8)
    ax2.set_ylabel("Chosen arm percentile rank (%)\namong all arms in trial context")
    ax2.set_ylim(0, 112)
    ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.set_title("Chosen-Arm Rank Among Alternatives\n"
                  "(100% = best possible; 50% = random; 0% = worst possible)",
                  fontweight="bold", fontsize=10)

    # Colorbar: darker = more eval contexts = more reliable
    sm = plt.cm.ScalarMappable(cmap=blues, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax2, shrink=0.6, pad=0.02)
    cbar.set_label("# evaluation contexts\n(trial × clinical arm)", fontsize=8)
    ax2.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    svg_path = str(save_path).replace(".png", ".svg")
    plt.savefig(svg_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")
    print(f"  Saved: {svg_path}")


def plot_lono_vs_lodo(lono_df, lodo_df, save_path):
    """
    Two-panel LONO vs LODO comparison.

    Panel A — Dumbbell chart by drug:
      Each drug: LONO pct_rank (circle) connected to LODO pct_rank (square).
      Gap = same-drug leakage benefit. Sorted by LODO pct_rank (best → worst).

    Panel B — Scatter per eval cell:
      x = LONO pct_rank, y = LODO pct_rank, coloured by drug.
      Above diagonal: LONO better (leakage helps).
      Below diagonal: LODO surprisingly competitive.
    """
    # ── Per-drug aggregation ──────────────────────────────────────────────────
    lono_drug = (lono_df.groupby("drug")["chosen_pct_rank"]
                 .mean().rename("lono_pct_rank").reset_index())
    lodo_drug = (lodo_df.groupby("drug")["chosen_pct_rank"]
                 .mean().rename("lodo_pct_rank").reset_index())
    drug_df = lono_drug.merge(lodo_drug, on="drug")
    drug_df["leakage_gap"] = drug_df["lono_pct_rank"] - drug_df["lodo_pct_rank"]
    # sort by LODO pct_rank descending (best cross-drug performers first)
    drug_df = drug_df.sort_values("lodo_pct_rank", ascending=False).reset_index(drop=True)

    # ── Per-cell merge ────────────────────────────────────────────────────────
    merged = lono_df[["eval_cell", "drug", "chosen_pct_rank"]].merge(
        lodo_df[["eval_cell", "chosen_pct_rank"]].rename(
            columns={"chosen_pct_rank": "lodo_pct_rank"}),
        on="eval_cell", how="inner"
    ).rename(columns={"chosen_pct_rank": "lono_pct_rank"})

    drugs_sorted = drug_df["drug"].tolist()
    cmap = plt.cm.get_cmap("tab10", len(drugs_sorted))
    drug_color = {d: cmap(i) for i, d in enumerate(sorted(merged["drug"].unique()))}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("LONO vs LODO: Quantifying Same-Drug Leakage",
                 fontweight="bold", fontsize=12)

    # ── Panel A: Dumbbell chart ───────────────────────────────────────────────
    ax = axes[0]
    y = np.arange(len(drug_df))

    for yi, row in zip(y, drug_df.itertuples()):
        col = drug_color[row.drug]
        # connecting line
        ax.plot([row.lono_pct_rank, row.lodo_pct_rank], [yi, yi],
                color="#cccccc", lw=1.5, zorder=1)
        # LONO dot
        ax.scatter(row.lono_pct_rank, yi, s=100, color="#2ecc71",
                   marker="o", zorder=3, label="LONO" if yi == 0 else "")
        # LODO dot
        ax.scatter(row.lodo_pct_rank, yi, s=100, color="#1a6b3c",
                   marker="s", zorder=3, label="LODO" if yi == 0 else "")
        # value labels
        ax.text(row.lono_pct_rank - 1.5, yi, f"{row.lono_pct_rank:.0f}",
                ha="right", va="center", fontsize=7.5, color="#2ecc71")
        ax.text(row.lodo_pct_rank + 1.5, yi, f"{row.lodo_pct_rank:.0f}",
                ha="left", va="center", fontsize=7.5, color="#1a6b3c")

    ax.axvline(50, color="#e74c3c", linestyle="--", lw=1.5, label="50 — random baseline")
    ax.set_yticks(y)
    ax.set_yticklabels(drugs_sorted, fontsize=9)
    ax.set_xlabel("Mean chosen-arm percentile rank (%)\n(100% = best; 50% = random; 0% = worst)")
    ax.set_xlim(-5, 115)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_title("Per-Drug pct_rank: LONO vs LODO\n"
                 "(gap = same-drug leakage benefit; sorted by LODO rank)",
                 fontweight="bold", fontsize=10)
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71",
               markersize=9, label="LONO (same-drug seen)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#1a6b3c",
               markersize=9, label="LODO (no same-drug data)"),
        Line2D([0], [0], color="#e74c3c", linestyle="--", lw=1.5,
               label="50 — random baseline"),
    ]
    ax.legend(handles=legend_els, fontsize=8, loc="lower right")

    # ── Panel B: Per-cell scatter ─────────────────────────────────────────────
    ax2 = axes[1]
    for drug in sorted(merged["drug"].unique()):
        sub = merged[merged["drug"] == drug]
        ax2.scatter(sub["lono_pct_rank"], sub["lodo_pct_rank"],
                    label=drug, color=drug_color[drug],
                    s=60, alpha=0.8, edgecolors="white", linewidths=0.4)

    ax2.plot([0, 100], [0, 100], "k--", lw=1, label="LONO = LODO")
    ax2.axhline(50, color="#e74c3c", linestyle=":", lw=1, alpha=0.6)
    ax2.axvline(50, color="#e74c3c", linestyle=":", lw=1, alpha=0.6)

    # Quadrant annotations
    ax2.text(5,  95, "LONO better\n(leakage helps)", fontsize=7, color="#888888",
             va="top")
    ax2.text(70, 10, "LODO better\n(unexpected)", fontsize=7, color="#888888",
             va="bottom")

    ax2.set_xlabel("LONO pct_rank — same-drug data seen (%; higher = better)")
    ax2.set_ylabel("LODO pct_rank — no same-drug data (%; higher = better)")
    ax2.set_xlim(-5, 112); ax2.set_ylim(-5, 112)
    ax2.set_xticks([0, 25, 50, 75, 100]); ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.set_title("Per Eval-Cell pct_rank: LONO vs LODO\n"
                  "(above diagonal = same-drug leakage helps; colored by drug)",
                  fontweight="bold", fontsize=10)
    ax2.legend(fontsize=7, ncol=2, loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def _plot_lono_vs_lodo_gap(lono_df, lodo_df, save_path):
    """(Legacy) Scatter: LONO vs LODO chosen gap per eval cell."""
    merged = lono_df[["eval_cell", "NCT", "drug", "chosen_gap"]].merge(
        lodo_df[["eval_cell", "chosen_gap"]].rename(columns={"chosen_gap": "chosen_gap_lodo"}),
        on="eval_cell", how="inner"
    )
    drugs = sorted(merged["drug"].unique())
    cmap = plt.cm.get_cmap("tab10", len(drugs))
    drug_color = {d: cmap(i) for i, d in enumerate(drugs)}

    fig, ax = plt.subplots(figsize=(7, 6))
    for drug in drugs:
        sub = merged[merged["drug"] == drug]
        ax.scatter(sub["chosen_gap"], sub["chosen_gap_lodo"],
                   label=drug, color=drug_color[drug], s=60, alpha=0.8)

    lo = min(merged["chosen_gap"].min(), merged["chosen_gap_lodo"].min()) - 1
    hi = max(merged["chosen_gap"].max(), merged["chosen_gap_lodo"].max()) + 1
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="LONO = LODO")
    ax.set_xlabel("LONO chosen |gap| (same-drug seen)")
    ax.set_ylabel("LODO chosen |gap| (no same-drug data)")
    ax.set_title("LONO vs LODO: Per-NCT Recommendation Quality",
                 fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_spearman_distribution(lono_df, lodo_df, save_path):
    """KDE of per-NCT Spearman ρ for LONO vs LODO."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for df_r, label, color in [
        (lono_df, "LONO (same-drug seen)", "#2ecc71"),
        (lodo_df, "LODO (no same-drug)",   "#27ae60"),
    ]:
        rhos = df_r["spearman_rho"].dropna()
        ax.hist(rhos, bins=15, alpha=0.6, color=color, label=label,
                edgecolor="white", density=True)
        ax.axvline(rhos.mean(), color=color, linestyle="--", lw=1.5,
                   label=f"{label} mean ρ={rhos.mean():.3f}")

    ax.axvline(0, color="black", lw=1, linestyle="-")
    ax.set_xlabel("Spearman ρ (predicted vs actual |gap| rank, per NCT)")
    ax.set_ylabel("Density")
    ax.set_title("Ranking Quality: LONO vs LODO", fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  Ranking Evaluation — Baselines + LONO vs LODO")
    print("=" * 65)

    df, feat_cols = load_data()

    # ── 8.1 Bootstrap random baseline ────────────────────────────────────────
    print("\n── Phase 1: Bootstrap random baseline ──")
    bootstrap = bootstrap_random_baseline(df)
    print(f"  Random pick: mean={bootstrap['mean']:.3f}  "
          f"95% CI=[{bootstrap['ci_lo']:.3f}, {bootstrap['ci_hi']:.3f}]")

    # ── 8.2 Rule-based baselines ──────────────────────────────────────────────
    print("\n── Phase 2: Rule-based baselines ──")
    rule_results = run_all_rule_baselines(df)

    # ── 8.3 Model selection (LODO family comparison + LONO-Optuna tuning) ───────
    # Runs only when no saved params exist; otherwise loads from disk.
    if PARAMS_PATH.exists():
        print("\n── Phase 3: Load saved model parameters (skip selection) ──")
        family, params = load_or_default_params()
    else:
        print("\n── Phase 3: Model selection — LODO MAE benchmark + LONO tuning ──")
        family, params = run_model_selection(df, feat_cols, n_trials=60)

    # ── 8.4 LONO ranking evaluation ───────────────────────────────────────────
    print("\n── Phase 4: LONO ranking evaluation ──")
    lono_df = lono_ranking(df, feat_cols, family, params)
    lono_df.to_csv(RESULTS_DIR / "lono_ranking_results.csv", index=False)
    lono_agg = aggregate(lono_df)
    rho_lono = lono_df["spearman_rho"].dropna()
    print(f"\n  LONO summary:")
    print(f"    [cell-level]  SQ={lono_agg['cell_mean_status_quo']:.3f}  "
          f"rec={lono_agg['cell_mean_chosen_gap']:.3f}  "
          f"impr={lono_agg['cell_pct_improvement']:.1f}%  "
          f"pct_rank={lono_agg['cell_mean_pct_rank']:.1f}%")
    print(f"    [drug-level]  rec={lono_agg['drug_mean_chosen_gap']:.3f}  "
          f"impr={lono_agg['drug_pct_improvement']:.1f}%  "
          f"pct_rank={lono_agg['drug_mean_pct_rank']:.1f}% (equal)  "
          f"{lono_agg['drug_weighted_pct_rank']:.1f}% (weighted by n_cells)  "
          f"(n_drugs={lono_agg['n_drugs']})")
    print(f"    Mean Spearman ρ:  {rho_lono.mean():.3f}  "
          f"(sig p<0.05: {(lono_df['spearman_p'] < 0.05).sum()}/{len(lono_df)})")
    stratum_summary(lono_df, "LONO")

    # ── 8.5 LODO ranking evaluation (nested CV — per-fold hyperparameter tuning) ──
    print("\n── Phase 5: LODO ranking evaluation (nested CV) ──")
    lodo_df = lodo_ranking(df, feat_cols, family, n_trials=60)
    lodo_df.to_csv(RESULTS_DIR / "lodo_ranking_results.csv", index=False)
    lodo_agg = aggregate(lodo_df)
    rho_lodo = lodo_df["spearman_rho"].dropna()
    drug_summary = per_drug_summary(lodo_df)
    drug_summary.to_csv(RESULTS_DIR / "lodo_per_drug_summary.csv", index=False)

    print(f"\n  LODO summary:")
    print(f"    [cell-level]  SQ={lodo_agg['cell_mean_status_quo']:.3f}  "
          f"rec={lodo_agg['cell_mean_chosen_gap']:.3f}  "
          f"impr={lodo_agg['cell_pct_improvement']:.1f}%  "
          f"pct_rank={lodo_agg['cell_mean_pct_rank']:.1f}%")
    print(f"    [drug-level]  rec={lodo_agg['drug_mean_chosen_gap']:.3f}  "
          f"impr={lodo_agg['drug_pct_improvement']:.1f}%  "
          f"pct_rank={lodo_agg['drug_mean_pct_rank']:.1f}% (equal)  "
          f"{lodo_agg['drug_weighted_pct_rank']:.1f}% (weighted by n_cells)  "
          f"(n_drugs={lodo_agg['n_drugs']})")
    print(f"    Mean Spearman ρ:  {rho_lodo.mean():.3f}  "
          f"(sig p<0.05: {(lodo_df['spearman_p'] < 0.05).sum()}/{len(lodo_df)})")
    stratum_summary(lodo_df, "LODO")
    print(f"\n  Per-drug LODO results:")
    print(drug_summary[["drug", "n_ncts", "n_cells", "status_quo", "chosen_gap",
                         "pct_improvement", "mean_rho", "n_sig",
                         "n_cells_eval"]].to_string(index=False))

    # ── 8.6 Build summary table ───────────────────────────────────────────────
    # Use drug-level aggregation as the primary metric to avoid inflation from
    # data-rich drugs (semaglutide/liraglutide) or noisy small-cell drugs.
    rows = [{"picker": "random_pick",
             "cell_chosen_gap": bootstrap["mean"],
             "drug_chosen_gap": bootstrap["mean"],
             "cell_pct_impr": 0.0,
             "drug_pct_impr": 0.0}]
    for name, res in rule_results.items():
        rows.append({"picker": name,
                     "cell_chosen_gap": res["cell_mean_chosen_gap"],
                     "drug_chosen_gap": res["drug_mean_chosen_gap"],
                     "cell_pct_impr":   res["cell_pct_improvement"],
                     "drug_pct_impr":   res["drug_pct_improvement"]})
    rows.append({"picker": "LONO_model",
                 "cell_chosen_gap": lono_agg["cell_mean_chosen_gap"],
                 "drug_chosen_gap": lono_agg["drug_mean_chosen_gap"],
                 "cell_pct_impr":   lono_agg["cell_pct_improvement"],
                 "drug_pct_impr":   lono_agg["drug_pct_improvement"]})
    rows.append({"picker": "LODO_model",
                 "cell_chosen_gap": lodo_agg["cell_mean_chosen_gap"],
                 "drug_chosen_gap": lodo_agg["drug_mean_chosen_gap"],
                 "cell_pct_impr":   lodo_agg["cell_pct_improvement"],
                 "drug_pct_impr":   lodo_agg["drug_pct_improvement"]})

    summary_df = pd.DataFrame(rows).sort_values("drug_chosen_gap").reset_index(drop=True)
    summary_df.to_csv(RESULTS_DIR / "ranking_summary.csv", index=False)

    print("\n── Summary Table (sorted by drug-level chosen gap) ──")
    print(summary_df[["picker", "cell_chosen_gap", "cell_pct_impr",
                       "drug_chosen_gap", "drug_pct_impr"]].to_string(index=False))

    # ── 8.7 Plots ─────────────────────────────────────────────────────────────
    print("\n── Phase 6: Generating plots ──")
    plot_lodo_by_drug(lodo_df, str(RESULTS_DIR / "R0_lodo_by_drug.png"))
    plot_stratum_analysis(lono_df, lodo_df,
                          str(RESULTS_DIR / "R4_stratum_analysis.png"))
    plot_comparison(summary_df, bootstrap,
                    str(RESULTS_DIR / "R1_baseline_comparison.png"))
    plot_lono_vs_lodo(lono_df, lodo_df,
                      str(RESULTS_DIR / "R2_lono_vs_lodo.png"))
    plot_spearman_distribution(lono_df, lodo_df,
                               str(RESULTS_DIR / "R3_spearman_distribution.png"))

    print(f"\nAll results saved to: {RESULTS_DIR}")
    return summary_df, lono_df, lodo_df


if __name__ == "__main__":
    main()
