"""Scoring a chosen preclinical arm, and rolling those scores up to drug level."""

import numpy as np
import pandas as pd


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
