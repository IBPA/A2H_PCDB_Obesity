"""Stratified cluster bootstrap inference for cross-drug performance analysis."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings


@dataclass
class InferenceResult:
    """Results from bootstrap inference."""
    model: str
    outcome_name: str
    n_drugs: int
    n_ncts: int
    n_obs: int
    beta0_hat: float
    beta1_hat: float
    beta0_boot_median: float
    beta1_boot_median: float
    beta0_boot_ci_low: float
    beta0_boot_ci_high: float
    beta1_boot_ci_low: float
    beta1_boot_ci_high: float
    p_boot: float
    r_hat: float
    r_boot_median: float
    r_boot_ci_low: float
    r_boot_ci_high: float
    p_boot_r: float
    leave_one_out: Dict[str, float]


def fit_ols(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Fit simple OLS regression y = beta0 + beta1 * x.

    Args:
        x: Predictor variable (distance).
        y: Response variable (performance metric).

    Returns:
        Tuple of (beta0, beta1, pearson_r).
    """
    # Remove NaN values
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x = x[valid_mask]
    y = y[valid_mask]

    if len(x) < 2:
        return np.nan, np.nan, np.nan

    # OLS via least squares
    X = np.column_stack([np.ones(len(x)), x])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        beta0, beta1 = coeffs
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan

    # Pearson correlation
    if np.std(x) > 0 and np.std(y) > 0:
        r = np.corrcoef(x, y)[0, 1]
    else:
        r = np.nan

    return beta0, beta1, r


def stratified_cluster_bootstrap(
    nct_table: pd.DataFrame,
    outcome_col: str,
    distance_mode: str,
    n_bootstrap: int = 10000,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform stratified cluster bootstrap by drug.

    Resamples NCTs within each drug stratum to preserve drug composition
    while accounting for within-drug clustering.

    Args:
        nct_table: NCT-level metrics table for a single model.
            Must have columns: test_drug, nct_id, distance, and outcome_col.
        outcome_col: Name of the outcome column to analyze.
        n_bootstrap: Number of bootstrap replicates.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (beta0_boots, beta1_boots, r_boots) arrays of length n_bootstrap.
    """
    rng = np.random.default_rng(random_state)

    # Get unique drugs and their NCTs
    drugs = nct_table['test_drug'].unique()
    drug_ncts = {}
    drug_data = {}

    for drug in drugs:
        drug_df = nct_table[nct_table['test_drug'] == drug]
        ncts = drug_df['nct_id'].unique()
        drug_ncts[drug] = ncts
        # Store data indexed by NCT for fast lookup
        drug_data[drug] = {
            nct: drug_df[drug_df['nct_id'] == nct].iloc[0]
            for nct in ncts
        }

    beta0_boots = np.zeros(n_bootstrap)
    beta1_boots = np.zeros(n_bootstrap)
    r_boots = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        boot_distances = []
        boot_outcomes = []

        for drug in drugs:
            ncts = drug_ncts[drug]
            k = len(ncts)

            # Sample NCTs with replacement
            sampled_ncts = rng.choice(ncts, size=k, replace=True)

            # Collect data for sampled NCTs, then aggregate to drug-level mean
            drug_distances = []
            drug_outcomes = []
            for nct in sampled_ncts:
                row = drug_data[drug][nct]
                drug_distances.append(row[f'distance_{distance_mode}'])
                drug_outcomes.append(row[outcome_col])
            boot_distances.append(np.mean(drug_distances))
            boot_outcomes.append(np.mean(drug_outcomes))

        boot_distances = np.array(boot_distances)
        boot_outcomes = np.array(boot_outcomes)

        beta0, beta1, r = fit_ols(boot_distances, boot_outcomes)
        beta0_boots[b] = beta0
        beta1_boots[b] = beta1
        r_boots[b] = r

    return beta0_boots, beta1_boots, r_boots


def compute_bootstrap_pvalue(boot_estimates: np.ndarray, null_value: float = 0) -> float:
    """Compute two-sided bootstrap p-value for H0: estimate = null_value.

    Args:
        boot_estimates: Array of bootstrap estimates.
        null_value: Value under null hypothesis.

    Returns:
        Two-sided p-value.
    """
    valid = boot_estimates[~np.isnan(boot_estimates)]
    n = len(valid)
    if n == 0:
        return np.nan

    # Plus-one correction to avoid exact-zero p-values
    # (Phipson & Smyth, 2010; Davison & Hinkley, 1997)
    count_leq = np.sum(valid <= null_value)
    count_geq = np.sum(valid >= null_value)
    prop_leq = (count_leq + 1) / (n + 1)
    prop_geq = (count_geq + 1) / (n + 1)
    p_value = 2 * min(prop_leq, prop_geq)

    return min(p_value, 1.0)


def leave_one_drug_out_sensitivity(
    nct_table: pd.DataFrame,
    outcome_col: str,
    distance_mode: str
) -> Dict[str, float]:
    """Perform leave-one-drug-out sensitivity analysis.

    For each drug, drops all its NCTs and refits OLS to check
    if the slope sign is stable.

    Args:
        nct_table: NCT-level metrics table for a single model.
        outcome_col: Name of the outcome column.

    Returns:
        Dict mapping dropped drug name to beta1 when that drug is excluded.
    """
    drugs = nct_table['test_drug'].unique()
    results = {}

    for drop_drug in drugs:
        subset = nct_table[nct_table['test_drug'] != drop_drug]
        if len(subset) < 2:
            results[drop_drug] = np.nan
            continue

        drug_means = subset.groupby('test_drug').agg({
            f'distance_{distance_mode}': 'mean',
            outcome_col: 'mean'
        }).dropna()
        x = drug_means[f'distance_{distance_mode}'].values
        y = drug_means[outcome_col].values
        _, beta1, _ = fit_ols(x, y)
        results[drop_drug] = beta1

    return results


def run_inference_for_model(
    nct_table: pd.DataFrame,
    model_name: str,
    outcome_col: str,
    distance_mode: str,
    n_bootstrap: int = 10000,
    random_state: int = 42
) -> InferenceResult:
    """Run full inference for a single model and outcome.

    Args:
        nct_table: NCT-level metrics table for the model.
        model_name: Name of the model.
        outcome_col: Name of the outcome column.
        n_bootstrap: Number of bootstrap replicates.
        random_state: Random seed.

    Returns:
        InferenceResult with all estimates and CIs.
    """
    # Filter to valid outcome values
    valid_table = nct_table[~nct_table[outcome_col].isna()].copy()

    if len(valid_table) < 2:
        return InferenceResult(
            model=model_name,
            outcome_name=outcome_col,
            n_drugs=0,
            n_ncts=0,
            n_obs=0,
            beta0_hat=np.nan,
            beta1_hat=np.nan,
            beta0_boot_median=np.nan,
            beta1_boot_median=np.nan,
            beta0_boot_ci_low=np.nan,
            beta0_boot_ci_high=np.nan,
            beta1_boot_ci_low=np.nan,
            beta1_boot_ci_high=np.nan,
            p_boot=np.nan,
            r_hat=np.nan,
            r_boot_median=np.nan,
            r_boot_ci_low=np.nan,
            r_boot_ci_high=np.nan,
            p_boot_r=np.nan,
            leave_one_out={}
        )

    n_drugs = valid_table['test_drug'].nunique()
    n_ncts = valid_table['nct_id'].nunique()
    n_obs = len(valid_table)

    # Original estimates (drug-level means, consistent with bootstrap)
    drug_means = valid_table.groupby('test_drug').agg({
        f'distance_{distance_mode}': 'mean',
        outcome_col: 'mean'
    }).dropna()
    x = drug_means[f'distance_{distance_mode}'].values
    y = drug_means[outcome_col].values
    beta0_hat, beta1_hat, r_hat = fit_ols(x, y)

    # Bootstrap
    beta0_boots, beta1_boots, r_boots = stratified_cluster_bootstrap(
        valid_table, outcome_col, distance_mode,
        n_bootstrap=n_bootstrap, random_state=random_state
    )

    # Bootstrap statistics for beta0
    valid_beta0 = beta0_boots[~np.isnan(beta0_boots)]
    if len(valid_beta0) > 0:
        beta0_boot_median = np.median(valid_beta0)
        beta0_boot_ci_low = np.percentile(valid_beta0, 2.5)
        beta0_boot_ci_high = np.percentile(valid_beta0, 97.5)
    else:
        beta0_boot_median = np.nan
        beta0_boot_ci_low = np.nan
        beta0_boot_ci_high = np.nan

    # Bootstrap statistics for beta1
    valid_beta1 = beta1_boots[~np.isnan(beta1_boots)]
    if len(valid_beta1) > 0:
        beta1_boot_median = np.median(valid_beta1)
        beta1_boot_ci_low = np.percentile(valid_beta1, 2.5)
        beta1_boot_ci_high = np.percentile(valid_beta1, 97.5)
        p_boot = compute_bootstrap_pvalue(valid_beta1, 0)
    else:
        beta1_boot_median = np.nan
        beta1_boot_ci_low = np.nan
        beta1_boot_ci_high = np.nan
        p_boot = np.nan

    # Bootstrap statistics for r
    valid_r = r_boots[~np.isnan(r_boots)]
    if len(valid_r) > 0:
        r_boot_median = np.median(valid_r)
        r_boot_ci_low = np.percentile(valid_r, 2.5)
        r_boot_ci_high = np.percentile(valid_r, 97.5)
        p_boot_r = compute_bootstrap_pvalue(valid_r, 0)
    else:
        r_boot_median = np.nan
        r_boot_ci_low = np.nan
        r_boot_ci_high = np.nan
        p_boot_r = np.nan

    # Leave-one-drug-out sensitivity
    leave_one_out = leave_one_drug_out_sensitivity(valid_table, outcome_col,
                                                   distance_mode)

    return InferenceResult(
        model=model_name,
        outcome_name=outcome_col,
        n_drugs=n_drugs,
        n_ncts=n_ncts,
        n_obs=n_obs,
        beta0_hat=beta0_hat,
        beta1_hat=beta1_hat,
        beta0_boot_median=beta0_boot_median,
        beta1_boot_median=beta1_boot_median,
        beta0_boot_ci_low=beta0_boot_ci_low,
        beta0_boot_ci_high=beta0_boot_ci_high,
        beta1_boot_ci_low=beta1_boot_ci_low,
        beta1_boot_ci_high=beta1_boot_ci_high,
        p_boot=p_boot,
        r_hat=r_hat,
        r_boot_median=r_boot_median,
        r_boot_ci_low=r_boot_ci_low,
        r_boot_ci_high=r_boot_ci_high,
        p_boot_r=p_boot_r,
        leave_one_out=leave_one_out
    )


def run_full_inference(
    nct_table: pd.DataFrame,
    distance_mode: str,
    outcome_cols: List[str] = None,
    n_bootstrap: int = 10000,
    random_state: int = 42
) -> List[InferenceResult]:
    """Run inference for all models and outcomes.

    Args:
        nct_table: Full NCT-level metrics table with all models.
        outcome_cols: List of outcome columns to analyze.
            Default: ['MAE_nct', 'MAE_reduction_pct_nct', 'R2_nct']
        n_bootstrap: Number of bootstrap replicates.
        random_state: Random seed.

    Returns:
        List of InferenceResult objects.
    """
    if outcome_cols is None:
        outcome_cols = ['MAE_nct', 'MAE_reduction_pct_nct', 'R2_nct']

    models = nct_table['model'].unique()
    results = []

    for model in models:
        model_table = nct_table[nct_table['model'] == model]
        for outcome in outcome_cols:
            print(f"Running inference for {model}, {outcome}...")
            result = run_inference_for_model(
                model_table, model, outcome, distance_mode,
                n_bootstrap=n_bootstrap, random_state=random_state
            )
            results.append(result)

    return results


def results_to_dataframe(results: List[InferenceResult]) -> pd.DataFrame:
    """Convert inference results to DataFrame.

    Args:
        results: List of InferenceResult objects.

    Returns:
        DataFrame with inference results.
    """
    rows = []
    for r in results:
        rows.append({
            'model': r.model,
            'outcome': r.outcome_name,
            'n_drugs': r.n_drugs,
            'n_ncts': r.n_ncts,
            'n_obs': r.n_obs,
            'beta0_hat': r.beta0_hat,
            'beta1_hat': r.beta1_hat,
            'beta0_boot_median': r.beta0_boot_median,
            'beta1_boot_median': r.beta1_boot_median,
            'beta0_boot_CI_low': r.beta0_boot_ci_low,
            'beta0_boot_CI_high': r.beta0_boot_ci_high,
            'beta1_boot_CI_low': r.beta1_boot_ci_low,
            'beta1_boot_CI_high': r.beta1_boot_ci_high,
            'p_boot': r.p_boot,
            'r_hat': r.r_hat,
            'r_boot_median': r.r_boot_median,
            'r_boot_CI_low': r.r_boot_ci_low,
            'r_boot_CI_high': r.r_boot_ci_high,
            'p_boot_r': r.p_boot_r
        })
    return pd.DataFrame(rows)


def leave_one_out_to_dataframe(results: List[InferenceResult]) -> pd.DataFrame:
    """Convert leave-one-out results to DataFrame.

    Args:
        results: List of InferenceResult objects.

    Returns:
        DataFrame with leave-one-out slopes.
    """
    rows = []
    for r in results:
        for drug, beta1_drop in r.leave_one_out.items():
            rows.append({
                'model': r.model,
                'outcome': r.outcome_name,
                'dropped_drug': drug,
                'beta1_drop': beta1_drop,
                'beta1_original': r.beta1_hat,
                'sign_stable': (
                    np.sign(beta1_drop) == np.sign(r.beta1_hat)
                    if not (np.isnan(beta1_drop) or np.isnan(r.beta1_hat))
                    else None
                )
            })
    return pd.DataFrame(rows)
