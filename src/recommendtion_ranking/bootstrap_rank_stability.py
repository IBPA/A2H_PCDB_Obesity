"""
Per-Drug Bootstrap Ranking Stability Analysis for LODO Recommender
==================================================================
Quantifies how stable the top-ranked preclinical design recommendation is
for each held-out drug under bootstrap resampling of its evaluation cells.

Key design decisions
--------------------
* No model retraining: LODO models are trained once per drug (train on all
  other drugs), then used to predict abs_gap for every row of the held-out
  drug's evaluation data. Predictions are stored in a (cells × designs)
  matrix and reused for all bootstrap replicates.

* Design identity: a "candidate design" is identified by its unique
  combination of all preclinical biological feature values (preclinical_cols:
  species, strain, sex, disease model, administration route, dosage duration,
  animal age/weight/size). design_id is cross-cell consistent — the same
  biological study gets the same design_id in every evaluation cell.
  Bridge columns (dose_translation_ratio, duration_ratio, etc.) are NOT part
  of design_id; rows with the same design_id but different bridge values are
  "dose variants" of the same study, collapsed via min-aggregation.

* Aggregation rule: exactly matches the main pipeline.
    score(design k, bootstrap sample) = mean(min_pred_abs_gap for k across
                                             resampled cells)
  Designs are ranked ascending by score; ties broken by design_id position
  for determinism.

* Bootstrap unit: evaluation cells (NCT × clinical arm combinations) within
  each held-out drug. Resampling is done with replacement at the cell level,
  preserving the within-cell candidate structure intact.

Outputs
-------
  bootstrap_rank_stability_drug.csv     — drug-level summary
  bootstrap_rank_stability_candidate.csv — candidate-level summary
  R6_rank_stability.png                 — bar chart of top-1 bootstrap frequency

Usage
-----
  python app/bootstrap_rank_stability.py              # all drugs, B=2000
  python app/bootstrap_rank_stability.py --drugs liraglutide semaglutide --B 500
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import json

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
REPO_ROOT   = SCRIPT_DIR.parent.parent
DATA_PATH   = REPO_ROOT / "data" / "ml_ready_obesity_dataset.csv"
PARAMS_PATH = REPO_ROOT / "outputs" / "recommender_model_params.json"
RESULTS_DIR = REPO_ROOT / "outputs" / "ranking_evaluation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET       = "translation_outcome"
ABS_TGT      = "abs_gap"
GROUP        = "intervention"
NCT          = "NCT Number"
RANDOM_STATE = 42

# Drugs excluded from ranking evaluation (single preclinical design —
# all variation is dose-level only, not study design; see paper Methods §2.1)
DOSE_ARM_ONLY = {"tirzepatide", "medi0382", "phentermine", "naltrexone"}


# ── Model utilities ────────────────────────────────────────────────────────────

def _make_model(family: str, params: dict):
    if family == "LightGBM":
        return lgb.LGBMRegressor(**params)
    elif family == "XGBoost":
        return xgb.XGBRegressor(**params)
    elif family == "RandomForest":
        return RandomForestRegressor(
            **{k: v for k, v in params.items() if k != "n_jobs"}, n_jobs=-1)
    else:
        raise ValueError(f"Unknown family: {family}")


def load_params():
    """Load tuned model params saved by ranking_evaluation.py Phase 2."""
    if not PARAMS_PATH.exists():
        raise FileNotFoundError(
            f"No saved params at {PARAMS_PATH}. "
            "Run ranking_evaluation.py first to generate them.")
    with open(PARAMS_PATH) as f:
        saved = json.load(f)
    family = saved["model_family"]
    params = saved["params"]
    print(f"  Loaded model: {family}  from {PARAMS_PATH}")
    return family, params


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data():
    """
    Load dataset and add derived columns.
    Filters eval cells with < 2 preclinical options (unrankable).
    Returns (df, feat_cols, preclinical_cols).
    """
    df = pd.read_csv(DATA_PATH)
    df[ABS_TGT] = df[TARGET].abs()

    bridge_cols      = ["dose_translation_ratio", "cumulative_dose_translation_ratio",
                        "duration_ratio", "frequency_ratio", "is_route_match"]
    clinical_cols    = [c for c in df.columns if c.startswith("clinical_")
                        and c != "clinical_arm_id"]
    preclinical_cols = [c for c in df.columns if c.startswith("preclinical_")]
    feat_cols        = preclinical_cols + bridge_cols + clinical_cols

    # Eval cell: unique clinical arm identified by clinical_arm_id within NCT
    df["eval_group"] = (df.groupby([NCT, "clinical_arm_id"])
                          .ngroup()
                          .astype(str))

    # Drop cells with only 1 option — no ranking possible
    cell_sizes    = df.groupby("eval_group").size()
    rankable      = cell_sizes[cell_sizes >= 2].index
    df            = df[df["eval_group"].isin(rankable)].copy()

    return df, feat_cols, preclinical_cols


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
    "DIO-prone Sprague\u2013Dawley": "DIO-SD",
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


def assign_design_ids(drug_df: pd.DataFrame, preclinical_cols: list) -> pd.DataFrame:
    """
    Assign a design_id (human-readable label) to every row of the held-out
    drug's data, identifying the unique preclinical study configuration.

    design_id — string label like "mice/C57BL6/DIO/male/SC/14d/n=16".
    Encodes: species, strain, disease model, sex, administration route,
    dosage duration, and group size. Cross-cell consistent — the same
    biological study gets the same label in every evaluation cell.

    Bridge columns (dose_translation_ratio, duration_ratio, etc.) are NOT
    part of the label. Multiple rows with the same design_id but different
    bridge values are dose variants of the same study, collapsed via
    min-aggregation in build_prediction_matrix.
    """
    pool = drug_df[preclinical_cols].drop_duplicates().reset_index(drop=True)
    pool["design_id"] = pool.apply(make_design_label, axis=1)

    return drug_df.merge(pool, on=preclinical_cols, how="left")


# ── LODO inference (no retraining inside bootstrap) ───────────────────────────

def run_lodo_inference(df: pd.DataFrame, feat_cols: list,
                       family: str, params: dict,
                       drugs_to_evaluate: list) -> dict:
    """
    For each held-out drug, train on all other drugs once and predict
    abs_gap for every row of the held-out drug.

    Returns
    -------
    dict: drug -> DataFrame of held-out rows with column "pred_abs_gap" added.
    """
    X      = df[feat_cols].values
    y      = df[ABS_TGT].values
    groups = df[GROUP].values

    predictions = {}
    for drug in drugs_to_evaluate:
        tr_mask = groups != drug
        te_mask = groups == drug

        # Sanity check: must have training data
        assert tr_mask.sum() > 0, f"No training rows for drug={drug}"
        assert te_mask.sum() > 0, f"No test rows for drug={drug}"

        sc = StandardScaler()
        m  = _make_model(family, params)
        m.fit(sc.fit_transform(X[tr_mask]), y[tr_mask])

        df_te = df[te_mask].copy()
        df_te["pred_abs_gap"] = m.predict(sc.transform(df_te[feat_cols].values))

        n_cells = df_te["eval_group"].nunique()
        print(f"  [{drug:20s}]  {te_mask.sum():4d} rows  {n_cells:2d} cells  "
              f"mean_pred={df_te['pred_abs_gap'].mean():.2f}")
        predictions[drug] = df_te

    return predictions


# ── Prediction matrix builder ──────────────────────────────────────────────────

def build_prediction_matrix(drug_df_pred: pd.DataFrame):
    """
    Build a (cells × designs) prediction matrix for one drug.

    Aggregation: for each (cell, design_id) pair, take the minimum
    pred_abs_gap across any dose-ratio variants present in that cell.
    This treats each unique preclinical study (identified by biological
    features) as a single candidate, using its best-ranked row as its
    representative score — consistent with how the main pipeline would
    select among dose variants within a larger design choice.

    Parameters
    ----------
    drug_df_pred : DataFrame with columns eval_group, design_id, pred_abs_gap.

    Returns
    -------
    pivot : DataFrame, shape (n_cells, n_designs)
            Index = eval_group, Columns = design_id (int).
            Values = min pred_abs_gap for that design in that cell.
            Raises ValueError if any cell is missing a design.
    """
    pivot = drug_df_pred.pivot_table(
        index="eval_group",
        columns="design_id",
        values="pred_abs_gap",
        aggfunc="min",   # take best (lowest) pred for this design in this cell
    )
    # Sanity: every design must appear in every cell
    if pivot.isnull().any().any():
        n_missing = pivot.isnull().sum().sum()
        raise ValueError(
            f"Prediction matrix has {n_missing} NaN entries — "
            "some designs are missing from certain cells. "
            "The preclinical pool may not be fully shared across NCTs.")
    return pivot


# ── Core bootstrap ─────────────────────────────────────────────────────────────

def bootstrap_drug(pivot: pd.DataFrame,
                   actual_gaps: pd.Series,
                   B: int,
                   rng: np.random.Generator) -> dict:
    """
    Bootstrap ranking stability for one drug.

    Parameters
    ----------
    pivot        : (n_cells × n_candidates) prediction matrix.
    actual_gaps  : Series indexed by candidate_id, mean actual abs_gap
                   per candidate across cells (for reference only).
    B            : number of bootstrap replicates.
    rng          : numpy Generator.

    Returns
    -------
    dict with keys:
        ranks       : (B × n_candidates) array of bootstrap ranks (1-based).
        candidates  : list of candidate_id strings (column order).
    """
    mat         = pivot.values                 # shape (C, K)
    candidates  = pivot.columns.tolist()
    C, K        = mat.shape

    # Tie-breaking order: use candidate position in the sorted candidate list
    # (deterministic — same candidate always wins ties).
    tiebreak_order = np.arange(K, dtype=float) * 1e-9  # tiny nudge

    ranks = np.empty((B, K), dtype=np.int32)
    for b in range(B):
        cell_idx   = rng.integers(0, C, size=C)        # resample cells
        boot_mat   = mat[cell_idx, :]                   # (C, K)
        scores     = boot_mat.mean(axis=0) + tiebreak_order
        # Rank ascending: lowest score = rank 1 (best recommendation)
        ranks[b]   = stats.rankdata(scores, method="ordinal").astype(np.int32)

    return {"ranks": ranks, "candidates": candidates}


# ── Summary builders ───────────────────────────────────────────────────────────

def original_ranking(pivot: pd.DataFrame) -> pd.Series:
    """
    Compute original (non-bootstrapped) aggregate score and rank for
    each candidate, exactly matching the bootstrap aggregation rule.

    Returns Series of ranks indexed by candidate_id (rank 1 = best).
    """
    scores       = pivot.mean(axis=0)
    tiebreak     = pd.Series(np.arange(len(scores)) * 1e-9, index=scores.index)
    adj_scores   = scores + tiebreak
    return adj_scores.rank(method="first").astype(int)


def drug_level_summary(drug: str,
                       pivot: pd.DataFrame,
                       boot_result: dict,
                       original_ranks: pd.Series) -> dict:
    """One row for the drug-level output table."""
    ranks      = boot_result["ranks"]        # (B, K)
    B, K       = ranks.shape
    candidates = boot_result["candidates"]

    top1_cand  = original_ranks.idxmin()
    orig_rank1 = int(original_ranks[top1_cand])  # should be 1

    # Column index of the original top-1 candidate
    col_idx    = candidates.index(top1_cand)
    cand_ranks = ranks[:, col_idx]              # shape (B,)

    return {
        "held_out_drug":            drug,
        "n_eval_cells":             pivot.shape[0],
        "n_candidate_designs":      K,
        "original_top_design_id":   top1_cand,
        "original_top_score":       round(float(pivot.mean(axis=0)[top1_cand]), 4),
        "top1_frequency":           round((cand_ranks == 1).mean(), 4),
        "top2_frequency":           round((cand_ranks <= 2).mean(), 4),
        "top3_frequency":           round((cand_ranks <= 3).mean(), 4),
        "median_rank":              round(float(np.median(cand_ranks)), 2),
        "rank_q1":                  round(float(np.percentile(cand_ranks, 25)), 2),
        "rank_q3":                  round(float(np.percentile(cand_ranks, 75)), 2),
    }


def candidate_level_summary(drug: str,
                             pivot: pd.DataFrame,
                             boot_result: dict,
                             original_ranks: pd.Series) -> pd.DataFrame:
    """One row per candidate for the candidate-level output table."""
    ranks      = boot_result["ranks"]        # (B, K)
    candidates = boot_result["candidates"]
    scores     = pivot.mean(axis=0)          # original aggregate scores

    rows = []
    for k, cid in enumerate(candidates):
        crank = ranks[:, k]
        rows.append({
            "held_out_drug":        drug,
            "design_id":            cid,
            "original_score":       round(float(scores[cid]), 4),
            "original_rank":        int(original_ranks[cid]),
            "boot_mean_rank":       round(float(crank.mean()), 3),
            "boot_median_rank":     round(float(np.median(crank)), 2),
            "boot_rank_q1":         round(float(np.percentile(crank, 25)), 2),
            "boot_rank_q3":         round(float(np.percentile(crank, 75)), 2),
            "top1_frequency":       round(float((crank == 1).mean()), 4),
            "top2_frequency":       round(float((crank <= 2).mean()), 4),
            "top3_frequency":       round(float((crank <= 3).mean()), 4),
        })
    return pd.DataFrame(rows).sort_values("original_rank").reset_index(drop=True)


# ── Figure ─────────────────────────────────────────────────────────────────────

def plot_stability(drug_summary_df: pd.DataFrame, save_path: Path):
    """
    Horizontal bar chart: top-1 bootstrap frequency for the original
    top-ranked design, one bar per held-out drug.
    Bars colored by frequency (green ≥ 0.5, amber 0.3–0.5, red < 0.3).
    Annotated with n_eval_cells and n_candidate_designs.
    """
    df = drug_summary_df.sort_values("top1_frequency", ascending=True).copy()
    n  = len(df)

    fig, ax = plt.subplots(figsize=(9, max(4, n * 0.6 + 1.5)))

    bar_colors = [
        "#27ae60" if f >= 0.50 else "#f39c12" if f >= 0.30 else "#e74c3c"
        for f in df["top1_frequency"]
    ]
    bars = ax.barh(df["held_out_drug"], df["top1_frequency"],
                   color=bar_colors, edgecolor="white", height=0.6)

    # Frequency label inside/outside bar
    for bar, row in zip(bars, df.itertuples()):
        x_val = bar.get_width()
        label = (f"{row.top1_frequency:.0%}  "
                 f"(cells={row.n_eval_cells}, cands={row.n_candidate_designs})")
        ax.text(min(x_val + 0.01, 0.98), bar.get_y() + bar.get_height() / 2,
                label, va="center", ha="left", fontsize=8.5)

    ax.axvline(0.50, color="#e74c3c", linestyle="--", lw=1.5,
               label="50% — coin-flip stability")
    ax.axvline(1.0 / df["n_candidate_designs"].max(), color="#888888",
               linestyle=":", lw=1.2,
               label=f"Random chance (≈ 1/K)")

    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Bootstrap top-1 frequency\n"
                  "(proportion of replicates where original top design ranks #1)",
                  fontsize=10)
    ax.set_title("Ranking Stability: How Often Does the Top-Ranked\n"
                 "Preclinical Design Stay #1 Under Cell Resampling?",
                 fontweight="bold", fontsize=11)

    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor="#27ae60", label="≥ 50% — stable"),
        Patch(facecolor="#f39c12", label="30–50% — moderate"),
        Patch(facecolor="#e74c3c", label="< 30% — unstable"),
    ]
    ax.legend(handles=legend_els, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(B: int = 2000, seed: int = RANDOM_STATE,
         drugs_filter: list = None, exclude_dose_arm_only: bool = True):
    """
    Run bootstrap rank stability analysis.

    Parameters
    ----------
    B                    : Number of bootstrap replicates (default 2000).
    seed                 : Random seed for reproducibility.
    drugs_filter         : If provided, restrict to this list of drugs.
    exclude_dose_arm_only: If True (default), skip drugs where all preclinical
                           variation is dose-level only (tirzepatide, medi0382).
    """
    print("=" * 65)
    print("  Bootstrap Rank Stability — LODO Recommender")
    print(f"  B={B:,}  seed={seed}")
    print("=" * 65)

    rng = np.random.default_rng(seed)

    # ── 1. Load data and model ─────────────────────────────────────────────────
    df, feat_cols, preclinical_cols = load_data()
    family, params = load_params()

    # Determine which drugs to evaluate
    all_drugs = sorted(df[GROUP].unique())
    if exclude_dose_arm_only:
        all_drugs = [d for d in all_drugs if d not in DOSE_ARM_ONLY]
    if drugs_filter:
        all_drugs = [d for d in all_drugs if d in drugs_filter]

    print(f"\n  Evaluating {len(all_drugs)} drugs: {all_drugs}")

    # ── 2. LODO inference (train once per drug, no retraining in bootstrap) ────
    print(f"\n── LODO inference (no model retraining) ──")
    drug_preds = run_lodo_inference(df, feat_cols, family, params, all_drugs)

    # ── 3. Bootstrap per drug ──────────────────────────────────────────────────
    print(f"\n── Bootstrap stability  (B={B:,}) ──")
    drug_rows      = []
    candidate_rows = []

    for drug in all_drugs:
        df_pred = drug_preds[drug].copy()

        # Assign design_id and variant_id
        df_pred = assign_design_ids(df_pred, preclinical_cols)

        # Sanity: every cell should have the same set of design_ids.
        # Dose variants (same design_id, different bridge-feature rows) are
        # already collapsed via min-aggregation in build_prediction_matrix.
        cand_sets = df_pred.groupby("eval_group")["design_id"].apply(set)
        if not all(s == cand_sets.iloc[0] for s in cand_sets):
            missing = max(len(cand_sets.iloc[0].symmetric_difference(s))
                          for s in cand_sets)
            if missing > 0:
                print(f"  WARNING [{drug}]: {missing} design_id(s) absent from "
                      "some cells — check for genuinely non-shared preclinical pool.")

        # Build prediction matrix (cells × designs, min-aggregated)
        pivot         = build_prediction_matrix(df_pred)
        orig_ranks    = original_ranking(pivot)

        # Mean actual gap per design (for reference)
        actual_mean   = (df_pred.groupby("design_id")[ABS_TGT]
                         .mean()
                         .reindex(pivot.columns))

        n_cells, n_cands = pivot.shape
        print(f"  [{drug:20s}]  cells={n_cells}  designs={n_cands}  "
              f"top1=design_{orig_ranks.idxmin()}  "
              f"top1_score={pivot.mean(axis=0).min():.2f}")

        # Bootstrap
        boot = bootstrap_drug(pivot, actual_mean, B=B, rng=rng)

        # Summaries
        drug_rows.append(
            drug_level_summary(drug, pivot, boot, orig_ranks))
        candidate_rows.append(
            candidate_level_summary(drug, pivot, boot, orig_ranks))

    # ── 4. Combine and save ────────────────────────────────────────────────────
    drug_df      = pd.DataFrame(drug_rows).sort_values(
        "top1_frequency", ascending=False).reset_index(drop=True)
    candidate_df = pd.concat(candidate_rows, ignore_index=True)

    drug_csv  = RESULTS_DIR / "bootstrap_rank_stability_drug.csv"
    cand_csv  = RESULTS_DIR / "bootstrap_rank_stability_candidate.csv"
    drug_df.to_csv(drug_csv, index=False)
    candidate_df.to_csv(cand_csv, index=False)
    print(f"\n  Saved drug summary:      {drug_csv}")
    print(f"  Saved candidate summary: {cand_csv}")

    # ── 5. Print summary ───────────────────────────────────────────────────────
    print("\n── Drug-level ranking stability ──")
    print(drug_df[["held_out_drug", "n_eval_cells", "n_candidate_designs",
                   "original_top_design_id", "original_top_score",
                   "top1_frequency", "top2_frequency", "top3_frequency",
                   "median_rank", "rank_q1", "rank_q3"]].to_string(index=False))

    # ── 6. Figure ──────────────────────────────────────────────────────────────
    fig_path = RESULTS_DIR / "R6_rank_stability.png"
    plot_stability(drug_df, fig_path)

    return drug_df, candidate_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bootstrap ranking stability for LODO recommender.")
    parser.add_argument("--B", type=int, default=2000,
                        help="Number of bootstrap replicates (default 2000).")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE,
                        help="Random seed (default 42).")
    parser.add_argument("--drugs", nargs="+", default=None,
                        help="Restrict to specific drug(s). Default: all evaluable.")
    parser.add_argument("--include-dose-arm-only", action="store_true",
                        help="Include dose-arm-only drugs (tirzepatide, medi0382).")
    args = parser.parse_args()

    main(
        B=args.B,
        seed=args.seed,
        drugs_filter=args.drugs,
        exclude_dose_arm_only=not args.include_dose_arm_only,
    )
