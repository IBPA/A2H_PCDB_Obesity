"""
Bootstrap Uncertainty Quantification for LODO Recommender Evaluation
=====================================================================
Two bootstrap strategies for the drug-level recommendation claim:

Bootstrap 1 — Drug-level cluster bootstrap:
    Treats each drug as the resampling unit.
    Samples drugs with replacement, recomputes cross-drug means.
    Directly quantifies uncertainty at the level of the paper claim.

Bootstrap 2 — Within-drug cell bootstrap:
    Keeps the drug set fixed.
    Within each drug, resamples evaluation cells with replacement.
    Recomputes per-drug summaries, then averages across drugs.
    Addresses sensitivity to which clinical contexts were observed
    within each drug (relevant for data-rich drugs like semaglutide
    and liraglutide that contribute many cells).

Four metrics tracked:
    mean_pct_rank         — lower is better (50% = random)
    mean_improvement_pp   — absolute gap reduction in percentage points
    mean_pct_improvement  — % reduction relative to status quo
    n_drugs_improving     — count of drugs where improvement_pp > 0

Output:
    app/results/ranking_evaluation/bootstrap_results.csv
    app/results/ranking_evaluation/R5_bootstrap_distributions.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Config ────────────────────────────────────────────────────────────────────
N_BOOTSTRAP   = 10_000
RANDOM_STATE  = 42
CI_LOW, CI_HIGH = 2.5, 97.5

RESULTS_DIR = Path(__file__).parent.parent.parent / "outputs" / "ranking_evaluation"
LODO_CSV    = RESULTS_DIR / "lodo_ranking_results.csv"
OUT_CSV     = RESULTS_DIR / "bootstrap_results.csv"
OUT_FIG     = RESULTS_DIR / "R5_bootstrap_distributions.png"

# Drugs excluded from the primary claim (single preclinical design — dose arms
# only, not distinct study designs; see Methods for rationale)
DOSE_ARM_ONLY = {"tirzepatide", "medi0382", "phentermine", "naltrexone"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def drug_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-drug summary metrics from cell-level LODO results.
    Returns one row per drug with columns:
        mean_pct_rank, mean_improvement_pp, mean_pct_improvement, improves
    """
    rows = []
    for drug, grp in df.groupby("drug"):
        rows.append({
            "drug":               drug,
            "mean_pct_rank":      grp["chosen_pct_rank"].mean(),
            "mean_improvement_pp": grp["improvement"].mean(),
            "mean_pct_improvement": (grp["improvement"] / grp["status_quo"] * 100).mean(),
            "improves":           int(grp["improvement"].mean() > 0),
        })
    return pd.DataFrame(rows).set_index("drug")


def cross_drug_means(drug_df: pd.DataFrame) -> dict:
    """Average per-drug summaries across the drug sample."""
    return {
        "mean_pct_rank":          drug_df["mean_pct_rank"].mean(),
        "mean_improvement_pp":    drug_df["mean_improvement_pp"].mean(),
        "mean_pct_improvement":   drug_df["mean_pct_improvement"].mean(),
        "n_drugs_improving":      drug_df["improves"].sum(),
    }


# ── Bootstrap 1: Drug-level cluster bootstrap ────────────────────────────────

def bootstrap_drug_level(drug_df: pd.DataFrame, n_boot: int, rng) -> pd.DataFrame:
    """
    Resample drugs with replacement. Each iteration draws len(drug_df) drugs
    (with replacement) from the drug pool and computes cross-drug means.
    """
    drugs = drug_df.index.tolist()
    n = len(drugs)
    records = []
    for _ in range(n_boot):
        sampled = drug_df.loc[rng.choice(drugs, size=n, replace=True)]
        records.append(cross_drug_means(sampled))
    return pd.DataFrame(records)


# ── Bootstrap 2: Within-drug cell bootstrap ──────────────────────────────────

def bootstrap_within_drug(cell_df: pd.DataFrame, n_boot: int, rng) -> pd.DataFrame:
    """
    Keep the drug set fixed. Within each drug, resample its evaluation cells
    with replacement. Recompute per-drug summaries, then average across drugs.
    """
    records = []
    for _ in range(n_boot):
        resampled_parts = []
        for drug, grp in cell_df.groupby("drug"):
            idx = rng.choice(len(grp), size=len(grp), replace=True)
            resampled_parts.append(grp.iloc[idx])
        resampled_df = pd.concat(resampled_parts)
        d_sum = drug_summary(resampled_df)
        records.append(cross_drug_means(d_sum))
    return pd.DataFrame(records)


# ── CI computation ────────────────────────────────────────────────────────────

def compute_ci(boot_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in boot_df.columns:
        rows.append({
            "metric":   col,
            "mean":     boot_df[col].mean(),
            "ci_low":   np.percentile(boot_df[col], CI_LOW),
            "ci_high":  np.percentile(boot_df[col], CI_HIGH),
        })
    return pd.DataFrame(rows).set_index("metric")


# ── Plotting ──────────────────────────────────────────────────────────────────

METRIC_META = {
    "mean_pct_rank": {
        "label":    "Mean pct_rank (%)",
        "ref":       50,
        "ref_label": "Random (50%)",
        "lower_better": True,
    },
    "mean_improvement_pp": {
        "label":    "Mean improvement (pp)",
        "ref":       0,
        "ref_label": "No improvement",
        "lower_better": False,
    },
    "mean_pct_improvement": {
        "label":    "Mean % improvement",
        "ref":       0,
        "ref_label": "No improvement",
        "lower_better": False,
    },
    "n_drugs_improving": {
        "label":    "Drugs with improvement > 0",
        "ref":       None,
        "ref_label": None,
        "lower_better": False,
    },
}

COLOR_DRUG = "#2166ac"   # blue — drug-level bootstrap
COLOR_CELL = "#d6604d"   # red  — within-drug cell bootstrap


def plot_distributions(boot1: pd.DataFrame, boot2: pd.DataFrame,
                       point_est: dict, save_path: Path, subtitle: str = ""):
    metrics = list(METRIC_META.keys())
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    fig.suptitle(
        "Bootstrap Uncertainty — LODO Recommender" + (f"\n{subtitle}" if subtitle else ""),
        fontsize=13, fontweight="bold", y=1.01
    )

    for ax, metric in zip(axes, metrics):
        meta = METRIC_META[metric]
        v1 = boot1[metric].values
        v2 = boot2[metric].values
        ci1 = (np.percentile(v1, CI_LOW), np.percentile(v1, CI_HIGH))
        ci2 = (np.percentile(v2, CI_LOW), np.percentile(v2, CI_HIGH))
        pt  = point_est[metric]

        ax.hist(v1, bins=60, alpha=0.55, color=COLOR_DRUG, density=True, label="Drug-level")
        ax.hist(v2, bins=60, alpha=0.55, color=COLOR_CELL, density=True, label="Cell (within-drug)")

        # Point estimate
        ax.axvline(pt, color="black", linewidth=2, linestyle="-", label=f"Obs. = {pt:.1f}")

        # CIs as shaded spans
        ax.axvspan(ci1[0], ci1[1], alpha=0.15, color=COLOR_DRUG)
        ax.axvspan(ci2[0], ci2[1], alpha=0.15, color=COLOR_CELL)

        # Reference line
        if meta["ref"] is not None:
            ax.axvline(meta["ref"], color="gray", linewidth=1.2,
                       linestyle="--", label=meta["ref_label"])

        ax.set_xlabel(meta["label"], fontsize=10)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_title(
            f'95% CI\nDrug: [{ci1[0]:.1f}, {ci1[1]:.1f}]\n'
            f'Cell: [{ci2[0]:.1f}, {ci2[1]:.1f}]',
            fontsize=8.5
        )
        ax.legend(fontsize=7.5, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(exclude_drugs=None, label=""):
    """
    Run both bootstrap analyses on the LODO results.

    Parameters
    ----------
    exclude_drugs : set or None
        Drug names to exclude before bootstrapping.
        Pass DOSE_ARM_ONLY to restrict to 7 true study-design drugs.
    label : str
        Suffix for output file names and figure subtitle.
    """
    if exclude_drugs is None:
        exclude_drugs = set()

    rng = np.random.default_rng(RANDOM_STATE)

    # Load and filter
    cell_df = pd.read_csv(LODO_CSV)
    cell_df = cell_df[~cell_df["drug"].isin(exclude_drugs)].copy()
    drugs_used = sorted(cell_df["drug"].unique())
    n_drugs = len(drugs_used)
    n_cells = cell_df["eval_cell"].nunique()
    print(f"\n{'='*60}")
    print(f"  Bootstrap analysis  {label}")
    print(f"  Drugs ({n_drugs}): {drugs_used}")
    print(f"  Eval cells: {n_cells}")
    print(f"  N bootstrap iterations: {N_BOOTSTRAP:,}")

    # Point estimates
    d_sum = drug_summary(cell_df)
    point = cross_drug_means(d_sum)
    print(f"\n  Point estimates:")
    for k, v in point.items():
        print(f"    {k:30s}: {v:.2f}")

    # Bootstrap 1
    print(f"\n  Running drug-level cluster bootstrap...")
    boot1 = bootstrap_drug_level(d_sum, N_BOOTSTRAP, rng)
    ci1   = compute_ci(boot1)

    # Bootstrap 2
    print(f"  Running within-drug cell bootstrap...")
    boot2 = bootstrap_within_drug(cell_df, N_BOOTSTRAP, rng)
    ci2   = compute_ci(boot2)

    # Combine results
    result = pd.DataFrame({
        "point_estimate":    pd.Series(point),
        "drug_boot_mean":    ci1["mean"],
        "drug_boot_ci_low":  ci1["ci_low"],
        "drug_boot_ci_high": ci1["ci_high"],
        "cell_boot_mean":    ci2["mean"],
        "cell_boot_ci_low":  ci2["ci_low"],
        "cell_boot_ci_high": ci2["ci_high"],
    }).round(2)

    suffix = f"_{label}" if label else ""
    csv_path = RESULTS_DIR / f"bootstrap_results{suffix}.csv"
    fig_path = RESULTS_DIR / f"R5_bootstrap_distributions{suffix}.png"

    result.to_csv(csv_path)
    print(f"\n  Results:\n{result.to_string()}")
    print(f"\n  Saved: {csv_path}")

    subtitle = f"{label}  |  {n_drugs} drugs, {n_cells} eval cells"
    plot_distributions(boot1, boot2, point, fig_path, subtitle=subtitle)

    return result, boot1, boot2


if __name__ == "__main__":
    # Analysis A: all 9 evaluable drugs in LODO results (primary table)
    res9, b9_drug, b9_cell = run(exclude_drugs=set(), label="all9drugs")

    # Analysis B: 7 study-design-distinct drugs (stratified analysis)
    # Tirzepatide and medi0382 are excluded because their preclinical "arms"
    # are dose levels of a single study design, not distinct designs.
    res7, b7_drug, b7_cell = run(exclude_drugs=DOSE_ARM_ONLY, label="7study_design_drugs")

    # Combined summary table
    combined = pd.concat(
        [res9.add_suffix("_all9"), res7.add_suffix("_7design")], axis=1
    )
    combined_path = RESULTS_DIR / "bootstrap_results_combined.csv"
    combined.to_csv(combined_path)
    print(f"\nCombined table saved: {combined_path}")
    print(combined.round(2).to_string())
