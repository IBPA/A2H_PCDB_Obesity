"""
Baseline Comparison & Study-Level Aggregation Analysis
=======================================================
Two questions this script answers:

1. BASELINES: How does LightGBM_tuned compare against proper statistical baselines?
   - Null (grand mean), per-drug mean, per-NCT mean, median, oracle drug mean

2. STUDY-LEVEL AGGREGATION: Should we evaluate at NCT level, not row level?
   - There are 36 unique NCT Numbers but 1430 rows (39.7x inflation)
   - Within-NCT variance = 93.3% of total variance (all from preclinical variation)
   - Row-level RMSE may be dominated by within-NCT noise
   - Study-level: aggregate predictions by NCT, compute metrics once per trial

Results saved to: app/results/
"""

import os
import warnings
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import lightgbm as lgb

warnings.filterwarnings("ignore")

SCRIPT_DIR  = Path(__file__).parent
REPO_ROOT   = SCRIPT_DIR.parent
DATA_PATH   = REPO_ROOT / "data" / "ml_ready_obesity_dataset.csv"
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL   = "translation_outcome"
GROUP_COL    = "intervention"
NCT_COL      = "NCT Number"
EXCLUDE_COLS = {GROUP_COL, NCT_COL, TARGET_COL}
RANDOM_STATE = 42


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def metrics(y_true, y_pred, label=""):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    bias = float(np.mean(y_pred - y_true))
    return {"label": label, "rmse": rmse, "mae": mae, "r2": r2, "bias": bias,
            "n": len(y_true)}


def lodo_predict(df, feat_cols, model_factory):
    """Run LODO CV; returns array of out-of-fold predictions."""
    drugs    = df[GROUP_COL].values
    X        = df[feat_cols].values
    y        = df[TARGET_COL].values
    preds    = np.full(len(df), np.nan)
    for drug in np.unique(drugs):
        te = drugs == drug
        tr = ~te
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        m = model_factory()
        m.fit(Xtr, y[tr])
        preds[te] = m.predict(Xte)
    return preds


def study_level_metrics(df, preds_col, label=""):
    """Aggregate predictions and actuals to NCT level, then compute metrics."""
    tmp = df[[NCT_COL, GROUP_COL, TARGET_COL]].copy()
    tmp["predicted"] = preds_col
    agg = tmp.groupby(NCT_COL).agg(
        actual=  (TARGET_COL, "mean"),
        predicted=("predicted", "mean"),
        drug=    (GROUP_COL, "first"),
        n_rows=  (TARGET_COL, "count"),
    ).reset_index()
    return metrics(agg["actual"], agg["predicted"], label=label), agg


# ──────────────────────────────────────────────────────────────────────────────
# Part 0: Dataset Structure Report
# ──────────────────────────────────────────────────────────────────────────────

def report_dataset_structure(df):
    print("\n── Dataset Structure ──")
    print(f"  Total rows       : {len(df)}")
    print(f"  Unique NCT #s    : {df[NCT_COL].nunique()}")
    print(f"  Unique drugs     : {df[GROUP_COL].nunique()}")
    print(f"  Inflation factor : {len(df)/df[NCT_COL].nunique():.1f}x rows per NCT")

    # Variance decomposition
    grand_mean   = df[TARGET_COL].mean()
    total_var    = df[TARGET_COL].var()
    nct_means    = df.groupby(NCT_COL)[TARGET_COL].mean()
    between_var  = nct_means.var()
    within_var   = df.groupby(NCT_COL)[TARGET_COL].var().mean()
    drug_means   = df.groupby(GROUP_COL)[TARGET_COL].mean()
    between_drug = drug_means.var()

    print(f"\n  Variance decomposition (target={TARGET_COL}):")
    print(f"    Total variance       : {total_var:8.3f}")
    print(f"    Between-drug var     : {between_drug:8.3f}  ({100*between_drug/total_var:.1f}%)")
    print(f"    Between-NCT var      : {between_var:8.3f}  ({100*between_var/total_var:.1f}%)")
    print(f"    Within-NCT var (avg) : {within_var:8.3f}  ({100*within_var/total_var:.1f}%)")
    print(f"\n  Note: within-NCT variance is 100% from preclinical design variation")
    print(f"  (clinical_outcome is constant within an NCT; all gap variation = preclinical)")

    nct_sizes = df.groupby(NCT_COL).size()
    print(f"\n  Rows per NCT Number: median={nct_sizes.median():.0f}, "
          f"mean={nct_sizes.mean():.0f}, max={nct_sizes.max()}")

    return total_var, between_var, within_var, between_drug


# ──────────────────────────────────────────────────────────────────────────────
# Part 1: Baselines
# ──────────────────────────────────────────────────────────────────────────────

def compute_baselines(df):
    """
    LODO-correct baselines (all use only training-set information).

    B1: Grand mean (train mean predicted for all test)
    B2: Drug-stratified mean — predict per-drug mean from training set
         (not valid in strict LODO since the test drug was unseen, so we use
          grand mean for the held-out drug — this IS the grand mean baseline)
    B3: NCT-level oracle mean — cheat: predict the mean actual value of the
         test NCT (this shows the upper bound of what NCT-level aggregation can achieve)
    B4: Nearest-drug mean — find the most similar drug in training set by
         target mean distance and predict its mean for all test rows
    """
    drugs   = df[GROUP_COL].values
    y       = df[TARGET_COL].values
    nct_ids = df[NCT_COL].values

    b1_preds = np.full(len(df), np.nan)   # grand mean
    b2_preds = np.full(len(df), np.nan)   # nearest-drug mean
    b3_preds = np.full(len(df), np.nan)   # NCT oracle mean
    b4_preds = np.full(len(df), np.nan)   # per-drug median (LODO)

    for drug in np.unique(drugs):
        te = drugs == drug
        tr = ~te

        train_mean   = float(np.mean(y[tr]))
        train_median = float(np.median(y[tr]))

        # B1: grand mean from training
        b1_preds[te] = train_mean

        # B4: training median
        b4_preds[te] = train_median

        # B2: find nearest drug mean in training set
        drug_means_train = {d: np.mean(y[(drugs == d)]) for d in np.unique(drugs[tr])}
        # nearest = training drug whose mean is closest to overall train mean
        # (we don't know test drug's mean — use the drug with mean closest to grand train mean)
        nearest_mean = min(drug_means_train.values(),
                           key=lambda m: abs(m - train_mean))
        b2_preds[te] = nearest_mean

    # B3: NCT oracle (not LODO-valid — cheat baseline to show ceiling)
    for nct in np.unique(nct_ids):
        mask = nct_ids == nct
        b3_preds[mask] = np.mean(y[mask])

    return b1_preds, b2_preds, b3_preds, b4_preds


# ──────────────────────────────────────────────────────────────────────────────
# Part 2: Best Model Predictions (reload from file or re-run)
# ──────────────────────────────────────────────────────────────────────────────

def get_best_model_preds(df):
    feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS]

    # Load best params from previous run
    params_path = RESULTS_DIR / "best_model_params.json"
    if params_path.exists():
        with open(params_path) as f:
            saved = json.load(f)
        params = {k: v for k, v in saved["params"].items()}
        params["verbose"] = -1
        print(f"  Loaded best params from {params_path}")
        model_factory = lambda: lgb.LGBMRegressor(**params)
    else:
        print("  No saved params found — using default LightGBM")
        model_factory = lambda: lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
            verbose=-1)

    # Try loading existing predictions
    pred_path = RESULTS_DIR / "lodo_best_model_predictions.csv"
    if pred_path.exists():
        saved_preds = pd.read_csv(pred_path)
        # Verify alignment
        if len(saved_preds) == len(df):
            print(f"  Loaded existing predictions from {pred_path}")
            return saved_preds["predicted"].values, feat_cols

    print("  Re-running LODO CV ...", end="", flush=True)
    preds = lodo_predict(df, feat_cols, model_factory)
    print(" done")
    return preds, feat_cols


# ──────────────────────────────────────────────────────────────────────────────
# Part 3: Row-level vs Study-level Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(df, model_preds, b1_preds, b2_preds, b3_preds, b4_preds):
    y = df[TARGET_COL].values

    # ── Row-level ──────────────────────────────────────────────────────────────
    row_results = [
        metrics(y, model_preds,    "LightGBM_tuned (LODO)"),
        metrics(y, b1_preds,       "B1: Grand mean (LODO-correct)"),
        metrics(y, b4_preds,       "B2: Grand median (LODO-correct)"),
        metrics(y, b2_preds,       "B3: Nearest-drug mean (LODO-correct)"),
        metrics(y, b3_preds,       "B4: NCT oracle mean [CHEATING - ceiling]"),
    ]
    row_df = pd.DataFrame(row_results)

    # ── Study-level (aggregate by NCT) ────────────────────────────────────────
    study_results = []
    study_aggs = {}

    pairs = [
        ("LightGBM_tuned (LODO)", model_preds),
        ("B1: Grand mean",        b1_preds),
        ("B2: Grand median",      b4_preds),
        ("B3: Nearest-drug mean", b2_preds),
        ("B4: NCT oracle [CHEAT]",b3_preds),
    ]
    for label, preds in pairs:
        m, agg_df = study_level_metrics(df, preds, label)
        study_results.append(m)
        study_aggs[label] = agg_df

    study_df = pd.DataFrame(study_results)

    return row_df, study_df, study_aggs


# ──────────────────────────────────────────────────────────────────────────────
# Part 4: Per-Drug Study-Level Analysis
# ──────────────────────────────────────────────────────────────────────────────

def per_drug_study_level(df, model_preds):
    """
    For the best model: per-drug study-level vs row-level comparison.
    Shows how much row-level RMSE is inflated by within-NCT noise.
    """
    y      = df[TARGET_COL].values
    drugs  = df[GROUP_COL].values
    ncts   = df[NCT_COL].values

    records = []
    for drug in sorted(np.unique(drugs)):
        mask = drugs == drug
        y_d  = y[mask]
        p_d  = model_preds[mask]
        n_d  = df[NCT_COL].values[mask]

        row_rmse = mean_squared_error(y_d, p_d, squared=False)
        row_r2   = r2_score(y_d, p_d)

        # Study-level for this drug
        tmp = pd.DataFrame({"nct": n_d, "actual": y_d, "pred": p_d})
        agg = tmp.groupby("nct").agg(actual=("actual","mean"),
                                     pred=("pred","mean")).reset_index()
        n_trials = len(agg)
        if n_trials > 1:
            study_rmse = mean_squared_error(agg["actual"], agg["pred"], squared=False)
            study_r2   = r2_score(agg["actual"], agg["pred"])
        else:
            study_rmse = abs(agg["actual"].iloc[0] - agg["pred"].iloc[0])
            study_r2   = float("nan")

        # How much variance is within-NCT for this drug?
        nct_means = tmp.groupby("nct")["actual"].mean()
        between_v = nct_means.var() if len(nct_means) > 1 else 0.0
        within_v  = tmp.groupby("nct")["actual"].var().mean()
        total_v   = tmp["actual"].var()

        records.append({
            "drug": drug,
            "n_rows": int(mask.sum()),
            "n_trials": n_trials,
            "row_rmse": round(row_rmse, 3),
            "study_rmse": round(study_rmse, 3),
            "rmse_ratio": round(row_rmse / study_rmse, 2) if study_rmse > 0 else float("nan"),
            "row_r2":   round(row_r2, 3),
            "study_r2": round(study_r2, 3) if not np.isnan(study_r2) else float("nan"),
            "within_nct_pct": round(100 * within_v / total_v, 1) if total_v > 0 else float("nan"),
        })

    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
# Part 5: Plots
# ──────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "liraglutide":    "#1f77b4",
    "semaglutide":    "#ff7f0e",
    "tirzepatide":    "#2ca02c",
    "metformin":      "#d62728",
    "orlistat":       "#9467bd",
    "canagliflozin":  "#8c564b",
    "survodutide":    "#e377c2",
    "exenatide":      "#7f7f7f",
    "phentermine":    "#bcbd22",
    "medi0382":       "#17becf",
    "naltrexone":     "#aec7e8",
}


def plot_variance_decomposition(df, save_path):
    """Stacked bar showing between-NCT vs within-NCT variance per drug."""
    records = []
    for drug in sorted(df[GROUP_COL].unique()):
        sub = df[df[GROUP_COL] == drug]
        total_v = sub[TARGET_COL].var()
        nct_means = sub.groupby(NCT_COL)[TARGET_COL].mean()
        between_v = nct_means.var() if len(nct_means) > 1 else 0.0
        within_v  = sub.groupby(NCT_COL)[TARGET_COL].var().mean()
        records.append({"drug": drug, "total": total_v,
                        "between_nct": between_v, "within_nct": within_v,
                        "n_trials": sub[NCT_COL].nunique(),
                        "n_rows": len(sub)})
    vd = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: absolute variance
    ax = axes[0]
    x = range(len(vd))
    bars_w = ax.bar(x, vd["within_nct"], label="Within-NCT (preclinical variation)",
                    color="#e74c3c", alpha=0.8)
    bars_b = ax.bar(x, vd["between_nct"], bottom=vd["within_nct"],
                    label="Between-NCT (trial variation)", color="#3498db", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(vd["drug"], rotation=30, ha="right")
    ax.set_ylabel("Variance of translation_outcome")
    ax.set_title("Target Variance Decomposition by Drug", fontweight="bold")
    ax.legend(fontsize=8)
    for i, row in vd.iterrows():
        ax.text(i, row["within_nct"] + row["between_nct"] + 2,
                f'{row["n_trials"]}T/{row["n_rows"]}R',
                ha="center", fontsize=7, color="gray")

    # Right: % within NCT
    ax2 = axes[1]
    within_pct = 100 * vd["within_nct"] / vd["total"]
    colors = ["#e74c3c" if p > 80 else "#f39c12" if p > 50 else "#2ecc71"
              for p in within_pct.fillna(0)]
    ax2.barh(vd["drug"], within_pct.fillna(0), color=colors)
    ax2.axvline(50, color="black", linestyle="--", lw=1, label="50% threshold")
    ax2.set_xlabel("Within-NCT variance (%)")
    ax2.set_title("% Variance Within-NCT per Drug\n(red = dominated by preclinical noise)",
                  fontweight="bold")
    ax2.set_xlim(0, 110)
    ax2.legend(fontsize=8)

    plt.suptitle("Variance Decomposition: Why Study-Level Metrics Matter",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_baseline_comparison(row_df, study_df, save_path):
    """Side-by-side: row-level vs study-level RMSE and R² for all baselines."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics_info = [
        ("rmse", "RMSE (lower=better)", False),
        ("r2",   "R² (higher=better)",  True),
    ]
    titles = ["Row-Level (1430 rows)", "Study-Level (36 NCT trials)"]
    dfs    = [row_df, study_df]

    for col, (metric, ylabel, higher_better) in enumerate(metrics_info):
        for row, (df_m, title) in enumerate(zip(dfs, titles)):
            ax = axes[row][col]
            sub = df_m.sort_values(metric, ascending=not higher_better)
            colors = []
            for lbl in sub["label"]:
                if "CHEAT" in lbl or "oracle" in lbl.lower():
                    colors.append("#95a5a6")   # grey = ceiling/invalid
                elif "LightGBM" in lbl:
                    colors.append("#2ecc71")   # green = our model
                else:
                    colors.append("#3498db")   # blue = baseline
            bars = ax.barh(sub["label"], sub[metric], color=colors)
            ax.set_title(f"{title}\n{ylabel}", fontweight="bold", fontsize=9)
            ax.set_xlabel(metric.upper())
            if metric == "r2":
                ax.axvline(0, color="black", lw=1, linestyle="--")
            for bar, val in zip(bars, sub[metric]):
                ax.text(bar.get_width() + abs(sub[metric].max()) * 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center", fontsize=8)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", label="LightGBM_tuned (our model)"),
        Patch(facecolor="#3498db", label="Statistical baseline"),
        Patch(facecolor="#95a5a6", label="Oracle/ceiling [invalid]"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle("Baseline Comparison: Row-Level vs Study-Level Evaluation",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_per_drug_row_vs_study(per_drug_df, save_path):
    """For each drug: row-level RMSE vs study-level RMSE, colored by within-NCT %."""
    valid = per_drug_df.dropna(subset=["study_rmse", "within_nct_pct"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Left: RMSE comparison
    ax = axes[0]
    x = range(len(valid))
    width = 0.35
    ax.bar([i - width/2 for i in x], valid["row_rmse"],
           width, label="Row-level RMSE", color="#e74c3c", alpha=0.8)
    ax.bar([i + width/2 for i in x], valid["study_rmse"],
           width, label="Study-level RMSE", color="#3498db", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(valid["drug"], rotation=30, ha="right")
    ax.set_ylabel("RMSE")
    ax.set_title("Row-Level vs Study-Level RMSE\n(per drug)", fontweight="bold")
    ax.legend(fontsize=9)

    # Middle: RMSE ratio (inflation)
    ax2 = axes[1]
    colors = ["#e74c3c" if r > 2 else "#f39c12" if r > 1.2 else "#2ecc71"
              for r in valid["rmse_ratio"].fillna(1)]
    ax2.barh(valid["drug"], valid["rmse_ratio"].fillna(1), color=colors)
    ax2.axvline(1.0, color="black", lw=1.5, linestyle="--")
    ax2.set_xlabel("Row RMSE / Study RMSE")
    ax2.set_title("RMSE Inflation Factor\n(row ÷ study level)", fontweight="bold")
    for i, (_, row) in enumerate(valid.iterrows()):
        if not np.isnan(row["rmse_ratio"]):
            ax2.text(row["rmse_ratio"] + 0.05, i, f'{row["rmse_ratio"]:.2f}x',
                     va="center", fontsize=8)

    # Right: scatter within-NCT % vs RMSE ratio
    ax3 = axes[2]
    ax3.scatter(valid["within_nct_pct"], valid["rmse_ratio"].fillna(1),
                s=valid["n_rows"] / 5 + 30,
                c=[PALETTE.get(d, "grey") for d in valid["drug"]],
                alpha=0.8, edgecolors="black", linewidths=0.5)
    for _, row in valid.iterrows():
        ax3.annotate(row["drug"], (row["within_nct_pct"], row["rmse_ratio"]),
                     fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax3.axhline(1.0, color="red", lw=1, linestyle="--")
    ax3.set_xlabel("Within-NCT variance (%)")
    ax3.set_ylabel("RMSE inflation (row / study)")
    ax3.set_title("Within-NCT Noise → RMSE Inflation", fontweight="bold")

    plt.suptitle("Necessity of Study-Level Aggregation per Drug",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_study_level_scatter(study_aggs, save_path):
    """
    Study-level actual vs predicted scatter for LightGBM_tuned,
    colored by drug, sized by n_rows contributing to each NCT mean.
    """
    agg_df = study_aggs["LightGBM_tuned (LODO)"]
    fig, ax = plt.subplots(figsize=(8, 7))

    for drug in sorted(agg_df["drug"].unique()):
        sub = agg_df[agg_df["drug"] == drug]
        ax.scatter(sub["actual"], sub["predicted"],
                   s=sub["n_rows"] * 1.5 + 40,
                   color=PALETTE.get(drug, "grey"),
                   alpha=0.7, edgecolors="black", linewidths=0.5,
                   label=drug)

    lo = min(agg_df["actual"].min(), agg_df["predicted"].min()) - 3
    hi = max(agg_df["actual"].max(), agg_df["predicted"].max()) + 3
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, label="Perfect prediction")

    r2   = r2_score(agg_df["actual"], agg_df["predicted"])
    rmse = mean_squared_error(agg_df["actual"], agg_df["predicted"], squared=False)
    mae  = mean_absolute_error(agg_df["actual"], agg_df["predicted"])
    ax.set_title(f"Study-Level LODO Predictions (n=36 NCT trials)\n"
                 f"RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.3f}",
                 fontweight="bold")
    ax.set_xlabel("Actual mean translation gap per NCT (%BW)")
    ax.set_ylabel("Predicted mean translation gap per NCT (%BW)")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_prediction_improvement(row_df, study_df, save_path):
    """
    Show the improvement of LightGBM over best baseline at both eval levels.
    Includes a summary stats panel.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, df_m, title in zip(axes, [row_df, study_df],
                                ["Row-Level (1430 rows)", "Study-Level (36 trials)"]):
        lgbm_rmse = df_m.loc[df_m["label"].str.contains("LightGBM"), "rmse"].values[0]
        lgbm_r2   = df_m.loc[df_m["label"].str.contains("LightGBM"), "r2"].values[0]

        # Best valid baseline (exclude oracle)
        valid_bl = df_m[~df_m["label"].str.contains("CHEAT|oracle", case=False)]
        valid_bl = valid_bl[~valid_bl["label"].str.contains("LightGBM")]
        best_bl_rmse = valid_bl["rmse"].min()
        best_bl_r2   = valid_bl.loc[valid_bl["rmse"].idxmin(), "r2"]
        best_bl_name = valid_bl.loc[valid_bl["rmse"].idxmin(), "label"]

        rmse_reduc = 100 * (best_bl_rmse - lgbm_rmse) / best_bl_rmse
        r2_gain    = lgbm_r2 - best_bl_r2

        categories = ["RMSE reduction\nvs best baseline (%)",
                      "R² absolute gain\nvs best baseline"]
        values = [rmse_reduc, r2_gain * 100]
        colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in values]

        bars = ax.bar(categories, values, color=colors, width=0.5)
        ax.axhline(0, color="black", lw=1)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (1 if val >= 0 else -3),
                    f"{val:.1f}", ha="center", fontweight="bold", fontsize=12)
        ax.set_title(f"{title}\nLightGBM vs best baseline ({best_bl_name.split(':')[0].strip()})",
                     fontweight="bold", fontsize=9)
        ax.set_ylabel("% improvement")

    plt.suptitle("Model Gain Over Best Statistical Baseline",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Baseline Comparison & Study-Level Aggregation Analysis")
    print("=" * 65)

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} cols")

    # ── 0. Dataset structure ──────────────────────────────────────────────────
    total_var, between_var, within_var, between_drug = report_dataset_structure(df)

    # ── 1. Best model predictions ─────────────────────────────────────────────
    print("\n── Getting best model predictions ──")
    model_preds, feat_cols = get_best_model_preds(df)

    # ── 2. Baselines ──────────────────────────────────────────────────────────
    print("\n── Computing baselines ──")
    b1_preds, b2_preds, b3_preds, b4_preds = compute_baselines(df)
    print("  B1: Grand mean (LODO-correct)")
    print("  B2: Grand median (LODO-correct)")
    print("  B3: Nearest-drug mean (LODO-correct)")
    print("  B4: NCT oracle mean [CHEATING — ceiling]")

    # ── 3. Row-level vs study-level metrics ───────────────────────────────────
    print("\n── Computing row-level and study-level metrics ──")
    row_df, study_df, study_aggs = compute_all_metrics(
        df, model_preds, b1_preds, b2_preds, b3_preds, b4_preds)

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n══ ROW-LEVEL METRICS (1430 rows) ══")
    print(row_df[["label","rmse","mae","r2","bias","n"]].to_string(index=False))

    print("\n══ STUDY-LEVEL METRICS (36 NCT trials) ══")
    print(study_df[["label","rmse","mae","r2","bias","n"]].to_string(index=False))

    # Per-drug analysis
    print("\n── Per-drug row vs study level ──")
    per_drug_df = per_drug_study_level(df, model_preds)
    print(per_drug_df.to_string(index=False))

    # ── Key findings printout ─────────────────────────────────────────────────
    lgbm_row_rmse   = row_df.loc[row_df["label"].str.contains("LightGBM"), "rmse"].values[0]
    lgbm_study_rmse = study_df.loc[study_df["label"].str.contains("LightGBM"), "rmse"].values[0]
    lgbm_row_r2     = row_df.loc[row_df["label"].str.contains("LightGBM"), "r2"].values[0]
    lgbm_study_r2   = study_df.loc[study_df["label"].str.contains("LightGBM"), "r2"].values[0]

    valid_bl_row   = row_df[~row_df["label"].str.contains("CHEAT|LightGBM", case=False)]
    valid_bl_study = study_df[~study_df["label"].str.contains("CHEAT|LightGBM", case=False)]
    best_bl_row_rmse   = valid_bl_row["rmse"].min()
    best_bl_study_rmse = valid_bl_study["rmse"].min()

    print(f"\n══ KEY FINDINGS ══")
    print(f"  Row-level:   LightGBM RMSE={lgbm_row_rmse:.3f}, R²={lgbm_row_r2:.3f}  |  "
          f"Best baseline RMSE={best_bl_row_rmse:.3f}  |  "
          f"Gain={100*(best_bl_row_rmse-lgbm_row_rmse)/best_bl_row_rmse:.1f}%")
    print(f"  Study-level: LightGBM RMSE={lgbm_study_rmse:.3f}, R²={lgbm_study_r2:.3f}  |  "
          f"Best baseline RMSE={best_bl_study_rmse:.3f}  |  "
          f"Gain={100*(best_bl_study_rmse-lgbm_study_rmse)/best_bl_study_rmse:.1f}%")
    print(f"\n  RMSE at row level    : {lgbm_row_rmse:.3f}  (inflated by within-NCT noise)")
    print(f"  RMSE at study level  : {lgbm_study_rmse:.3f}  (trial-level accuracy)")
    print(f"  RMSE reduction row→study: {100*(lgbm_row_rmse-lgbm_study_rmse)/lgbm_row_rmse:.1f}%")
    print(f"  R² at study level    : {lgbm_study_r2:.3f}  (trial-level predictability)")

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    row_df.to_csv(RESULTS_DIR / "baseline_row_level_metrics.csv", index=False)
    study_df.to_csv(RESULTS_DIR / "baseline_study_level_metrics.csv", index=False)
    per_drug_df.to_csv(RESULTS_DIR / "per_drug_row_vs_study_metrics.csv", index=False)
    study_aggs["LightGBM_tuned (LODO)"].to_csv(
        RESULTS_DIR / "study_level_predictions.csv", index=False)
    print(f"\nCSVs saved to {RESULTS_DIR}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n── Generating plots ──")

    plot_variance_decomposition(df,
        str(RESULTS_DIR / "08_variance_decomposition.png"))

    plot_baseline_comparison(row_df, study_df,
        str(RESULTS_DIR / "09_baseline_comparison.png"))

    plot_per_drug_row_vs_study(per_drug_df,
        str(RESULTS_DIR / "10_per_drug_row_vs_study.png"))

    plot_study_level_scatter(study_aggs,
        str(RESULTS_DIR / "11_study_level_scatter.png"))

    plot_prediction_improvement(row_df, study_df,
        str(RESULTS_DIR / "12_model_gain_over_baseline.png"))

    # ── Write research notes ──────────────────────────────────────────────────
    write_findings_note(row_df, study_df, per_drug_df, total_var, between_var,
                        within_var, between_drug, lgbm_row_rmse, lgbm_study_rmse,
                        lgbm_row_r2, lgbm_study_r2, best_bl_row_rmse, best_bl_study_rmse)

    print("\nDone.")


def write_findings_note(row_df, study_df, per_drug_df,
                        total_var, between_var, within_var, between_drug,
                        lgbm_row_rmse, lgbm_study_rmse,
                        lgbm_row_r2, lgbm_study_r2,
                        best_bl_row_rmse, best_bl_study_rmse):
    notes_dir = REPO_ROOT / "files" / "research_notes"
    notes_dir.mkdir(parents=True, exist_ok=True)

    row_gain   = 100 * (best_bl_row_rmse - lgbm_row_rmse) / best_bl_row_rmse
    study_gain = 100 * (best_bl_study_rmse - lgbm_study_rmse) / best_bl_study_rmse

    lines = [
        "# ML Analysis Findings — Baseline & Study-Level Evaluation\n",
        "## 1. Dataset Structure\n",
        f"- 1430 rows, but only **36 unique NCT Numbers** (clinical trials) — **39.7× inflation**",
        f"- Each clinical trial is paired with many preclinical study designs",
        f"- `translation_outcome = clinical_outcome − preclinical_outcome`",
        f"- Since clinical_outcome is **fixed per NCT**, all within-NCT variance = preclinical variation\n",
        "## 2. Variance Decomposition\n",
        f"| Component | Variance | % of Total |",
        f"|---|---|---|",
        f"| Total | {total_var:.2f} | 100% |",
        f"| Between-NCT (trial-level) | {between_var:.2f} | {100*between_var/total_var:.1f}% |",
        f"| Within-NCT (preclinical noise) | {within_var:.2f} | {100*within_var/total_var:.1f}% |",
        f"| Between-drug | {between_drug:.2f} | {100*between_drug/total_var:.1f}% |\n",
        f"**Key implication**: Row-level RMSE is dominated by within-NCT (preclinical) noise. "
        f"Study-level aggregation by NCT gives a cleaner signal of trial-level predictability.\n",
        "## 3. Baseline Comparison — Row-Level\n",
        row_df[["label","rmse","mae","r2","bias"]].sort_values("rmse").to_markdown(index=False),
        "\n## 4. Baseline Comparison — Study-Level (36 NCT trials)\n",
        study_df[["label","rmse","mae","r2","bias"]].sort_values("rmse").to_markdown(index=False),
        "\n## 5. Key Findings\n",
        f"- **Row-level**: LightGBM RMSE={lgbm_row_rmse:.3f}, R²={lgbm_row_r2:.3f} "
        f"(vs best baseline {best_bl_row_rmse:.3f}, **{row_gain:.1f}% better**)",
        f"- **Study-level**: LightGBM RMSE={lgbm_study_rmse:.3f}, R²={lgbm_study_r2:.3f} "
        f"(vs best baseline {best_bl_study_rmse:.3f}, **{study_gain:.1f}% better**)",
        f"- Study-level R² is {'higher' if lgbm_study_r2 > lgbm_row_r2 else 'lower'} than row-level R²",
        f"  → The model {'better' if lgbm_study_r2 > lgbm_row_r2 else 'does not better'} predicts mean trial-level gaps than individual arm gaps\n",
        "## 6. Per-Drug Row vs Study Level\n",
        per_drug_df.sort_values("rmse_ratio", ascending=False).to_markdown(index=False),
        "\n## 7. Necessity of Study-Level Evaluation\n",
        "**YES, study-level aggregation is necessary** because:",
        "- Within-NCT variance is 93.3% of total — row-level RMSE is heavily inflated by preclinical noise",
        "- The scientific question is *trial-level*: 'Can we predict how well a drug will translate?'",
        "- 36 trials is the true effective sample size, not 1430",
        "- Row-level metrics give false precision by treating correlated rows as independent",
        "- However, row-level metrics remain useful for measuring *within-trial* ranking of preclinical designs",
    ]

    out_path = notes_dir / "ml_analysis_findings.md"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Research notes: {out_path}")


if __name__ == "__main__":
    main()
