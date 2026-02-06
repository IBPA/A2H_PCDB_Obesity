"""Study-level (NCT) performance metrics computation."""

import pandas as pd
import numpy as np
from typing import Optional
import warnings


def compute_nct_level_metrics(
    eval_table: pd.DataFrame,
    min_variance_threshold: float = 1e-6
) -> pd.DataFrame:
    """Compute NCT-level performance metrics.

    For each (model, test_drug, nct_id) combination, computes:
    - MAE_nct: Mean absolute error
    - R2_nct: R-squared (with handling for low-variance cases)
    - MAE_reduction_pct_nct: Percentage MAE reduction vs baseline

    Args:
        eval_table: Evaluation table with columns:
            model, test_drug, nct_id, y_true, y_pred, y_baseline, distance.
        min_variance_threshold: Minimum variance for R2 computation.
            NCTs with lower variance will have R2 set to NaN.

    Returns:
        NCT-level DataFrame with columns:
            model, test_drug, nct_id, distance, n_samples,
            MAE_nct, R2_nct, MAE_reduction_pct_nct, R2_valid.
    """
    results = []

    grouped = eval_table.groupby(['model', 'test_drug', 'nct_id'])

    for (model, test_drug, nct_id), group in grouped:
        y_true = group['y_true'].values
        y_pred = group['y_pred'].values
        y_baseline = group['y_baseline'].values[0]  # Same for all rows in group
        distance_both = group['distance_both'].values[0]  # Same for all rows in group
        distance_ec50_only = group['distance_ec50_only'].values[0]  # Same for all rows in group
        distance_halflife_only = group['distance_halflife_only'].values[0]  # Same for all rows in group
        n_samples = len(y_true)

        # MAE
        mae_nct = np.mean(np.abs(y_true - y_pred))

        # Baseline MAE
        mae_baseline_nct = np.mean(np.abs(y_true - y_baseline))

        # MAE reduction percentage
        # Negative means model is better (lower MAE)
        if mae_baseline_nct > 0:
            mae_reduction_pct = 100 * (mae_nct - mae_baseline_nct) / mae_baseline_nct
        else:
            mae_reduction_pct = np.nan

        # R2 with variance check
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_res = np.sum((y_true - y_pred) ** 2)

        r2_valid = True
        if ss_tot < min_variance_threshold or n_samples < 2:
            # Low variance or single sample - R2 is unreliable
            r2_nct = np.nan
            r2_valid = False
        else:
            r2_nct = 1 - ss_res / ss_tot

        results.append({
            'model': model,
            'test_drug': test_drug,
            'nct_id': nct_id,
            'distance_both': distance_both,
            'distance_ec50_only': distance_ec50_only,
            'distance_halflife_only': distance_halflife_only,
            'n_samples': n_samples,
            'MAE_nct': mae_nct,
            'MAE_baseline_nct': mae_baseline_nct,
            'R2_nct': r2_nct,
            'MAE_reduction_pct_nct': mae_reduction_pct,
            'R2_valid': r2_valid
        })

    return pd.DataFrame(results)


def summarize_nct_metrics(nct_table: pd.DataFrame) -> pd.DataFrame:
    """Summarize NCT-level metrics by model and drug.

    Args:
        nct_table: NCT-level metrics table.

    Returns:
        Summary DataFrame with mean metrics per (model, test_drug).
    """
    summary = nct_table.groupby(['model', 'test_drug']).agg({
        'distance_both': 'first',
        'distance_ec50_only': 'first',
        'distance_halflife_only': 'first',
        'n_samples': 'sum',
        'MAE_nct': 'mean',
        'R2_nct': lambda x: x.dropna().mean() if x.dropna().size > 0 else np.nan,
        'MAE_reduction_pct_nct': 'mean',
        'nct_id': 'nunique'
    }).rename(columns={'nct_id': 'n_ncts'}).reset_index()

    return summary
