#!/usr/bin/env python3
"""Run cross-drug performance analysis.

This script analyzes whether PK/PD distance to the training drug predicts
cross-drug generalization performance, using stratified cluster bootstrap.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

from .data_preparation import create_single_drug_evaluation_table
from .distance_computation import compute_pkpd_distances, attach_distances_to_evaluation_table
from .study_level_metrics import compute_nct_level_metrics, summarize_nct_metrics
from .bootstrap_inference import (
    run_full_inference,
    results_to_dataframe,
    leave_one_out_to_dataframe
)


def run_analysis_single_drug(
    training_drug: str,
    predictions_path: str,
    test_data_path: str,
    train_data_path: str,
    pkpd_path: str,
    output_dir: str,
    n_bootstrap: int = 10000,
    random_state: int = 42,
    exclude_drugs: list = None,
    distance_mode: str = 'both'
):
    """Run cross-drug performance analysis for a single training drug.

    Args:
        training_drug: Name of the training drug (e.g., 'liraglutide').
        predictions_path: Path to model predictions TSV.
        test_data_path: Path to model test data CSV.
        train_data_path: Path to model training data CSV.
        pkpd_path: Path to PK/PD profiles CSV.
        output_dir: Directory to save results.
        n_bootstrap: Number of bootstrap replicates.
        random_state: Random seed.
        exclude_drugs: List of drug names to exclude from analysis.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"CROSS-DRUG PERFORMANCE ANALYSIS - {training_drug.upper()} MODEL")
    print("Analyzing PK/PD distance as predictor of cross-drug generalization")
    print("=" * 80)
    print()

    # Step 1: Create evaluation table for single drug
    print("Step 1: Creating evaluation table...")
    eval_table = create_single_drug_evaluation_table(
        training_drug,
        predictions_path,
        test_data_path,
        train_data_path
    )
    print(f"  Created table with {len(eval_table)} rows")
    print(f"  Model: {eval_table['model'].unique()}")
    print(f"  Test drugs: {eval_table['test_drug'].unique()}")

    # Filter out excluded drugs if specified
    if exclude_drugs:
        print(f"  Excluding drugs: {exclude_drugs}")
        eval_table = eval_table[~eval_table['test_drug'].str.lower().isin([d.lower() for d in exclude_drugs])]
        print(f"  After filtering: {len(eval_table)} rows")
        print(f"  Remaining test drugs: {eval_table['test_drug'].unique()}")
    print()

    # Step 2: Compute PK/PD distances
    print("Step 2: Computing PK/PD distances...")
    pkpd_distances = compute_pkpd_distances(pkpd_path)
    print("  PK/PD feature space (z-scored log features):")
    print(pkpd_distances[['Drug', 'z_log_EC50_ratio', 'z_log_half_life_hr',
                          'distance_lira_trained_both', 'distance_sema_trained_both',
                          'distance_lira_trained_ec50_only', 'distance_sema_trained_ec50_only',
                          'distance_lira_trained_halflife_only',
                          'distance_sema_trained_halflife_only']].to_string(index=False))
    print()

    # Step 3: Attach distances to evaluation table
    print("Step 3: Attaching distances to evaluation table...")
    eval_table = attach_distances_to_evaluation_table(eval_table, pkpd_distances)
    print()

    # Step 4: Compute NCT-level metrics
    print("Step 4: Computing NCT-level metrics...")
    nct_table = compute_nct_level_metrics(eval_table)
    print(f"  Created NCT-level table with {len(nct_table)} rows")
    print()

    # Save intermediate tables
    eval_table.to_csv(output_path / 'evaluation_table.csv', index=False)
    nct_table.to_csv(output_path / 'nct_level_metrics.csv', index=False)
    pkpd_distances.to_csv(output_path / 'pkpd_distances.csv', index=False)

    # Print NCT-level summary
    print("NCT-level metrics summary:")
    print("-" * 80)
    nct_summary = summarize_nct_metrics(nct_table)
    print(nct_summary.to_string(index=False))
    print()
    nct_summary.to_csv(output_path / 'nct_summary_by_drug.csv', index=False)

    # Step 5: Run bootstrap inference
    print("Step 5: Running stratified cluster bootstrap inference...")
    print(f"  Bootstrap replicates: {n_bootstrap}")
    print(f"  Random state: {random_state}")
    print()

    if distance_mode == 'both':
        distance_name = 'EC50_and_half-life'
        print(f"Analyzing distance mode: {distance_mode}, using both EC50 and half-life distances")
    else:
        distance_name = distance_mode.replace(' ', '_')
        print(f"Analyzing distance mode: {distance_mode}, using only {distance_mode} distance")
    
    results = run_full_inference(
        nct_table,
        distance_mode,
        outcome_cols=['MAE_nct', 'MAE_reduction_pct_nct', 'R2_nct'],
        n_bootstrap=n_bootstrap,
        random_state=random_state
    )

    # Convert to DataFrames
    results_df = results_to_dataframe(results)
    loo_df = leave_one_out_to_dataframe(results)

    # Save results
    results_df.to_csv(output_path / f'inference_results_{distance_name}.csv', index=False)
    loo_df.to_csv(output_path / f'leave_one_out_sensitivity_{distance_name}.csv', index=False)

    # Print results
    print()
    print("=" * 80)
    print("INFERENCE RESULTS")
    print("=" * 80)
    print()

    for _, row in results_df.iterrows():
        print(f"Model: {row['model']}, Outcome: {row['outcome']}")
        print(f"  N drugs: {row['n_drugs']}, N NCTs: {row['n_ncts']}, N obs: {row['n_obs']}")
        print(f"  beta1 (original): {row['beta1_hat']:.4f}")
        print(f"  beta1 (bootstrap median): {row['beta1_boot_median']:.4f}")
        print(f"  beta1 95% CI: [{row['beta1_boot_CI_low']:.4f}, {row['beta1_boot_CI_high']:.4f}]")
        print(f"  p-value (bootstrap, beta1): {row['p_boot']:.4e}")
        print(f"  Pearson r (original): {row['r_hat']:.4f}")
        print(f"  r 95% CI: [{row['r_boot_CI_low']:.4f}, {row['r_boot_CI_high']:.4f}]")
        print(f"  p-value (bootstrap, r): {row['p_boot_r']:.4e}")
        print()

    # Print leave-one-out sensitivity
    print("=" * 80)
    print("LEAVE-ONE-DRUG-OUT SENSITIVITY ANALYSIS")
    print("=" * 80)
    print()

    for model in results_df['model'].unique():
        for outcome in results_df['outcome'].unique():
            subset = loo_df[(loo_df['model'] == model) & (loo_df['outcome'] == outcome)]
            if len(subset) == 0:
                continue

            print(f"Model: {model}, Outcome: {outcome}")
            original_beta1 = subset['beta1_original'].iloc[0]
            print(f"  Original beta1: {original_beta1:.4f}")
            print(f"  {'Dropped Drug':<15} {'beta1_drop':>12} {'Sign Stable':>12}")
            print("  " + "-" * 40)

            for _, row in subset.iterrows():
                sign_str = "Yes" if row['sign_stable'] else "No" if row['sign_stable'] is not None else "N/A"
                beta1_str = f"{row['beta1_drop']:.4f}" if not np.isnan(row['beta1_drop']) else "N/A"
                print(f"  {row['dropped_drug']:<15} {beta1_str:>12} {sign_str:>12}")

            # Summary
            all_stable = subset['sign_stable'].dropna().all() if not subset['sign_stable'].isna().all() else None
            beta1_range = subset['beta1_drop'].dropna()
            if len(beta1_range) > 0:
                print(f"  Range of beta1: [{beta1_range.min():.4f}, {beta1_range.max():.4f}]")
                print(f"  Sign stability: {'All stable' if all_stable else 'Some sign changes'}")
            print()

    print("=" * 80)
    print(f"Results saved to: {output_path}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Run cross-drug performance analysis.'
    )
    parser.add_argument(
        '--training-drug', type=str, required=True,
        help='Training drug to analyze (e.g., liraglutide, semaglutide)'
    )
    parser.add_argument(
        '--predictions', type=str, required=True,
        help='Path to model predictions TSV'
    )
    parser.add_argument(
        '--test-data', type=str, required=True,
        help='Path to model test data CSV'
    )
    parser.add_argument(
        '--train-data', type=str, required=True,
        help='Path to model training data CSV'
    )
    parser.add_argument(
        '--pkpd-path', type=str, required=True,
        help='Path to PK/PD profiles CSV'
    )
    parser.add_argument(
        '--output-dir', type=str, required=True,
        help='Directory to save results'
    )
    parser.add_argument(
        '--n-bootstrap', type=int, default=10000,
        help='Number of bootstrap replicates (default: 10000)'
    )
    parser.add_argument(
        '--random-state', type=int, default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--exclude-drugs', type=str, default=None,
        help='Comma-separated list of drugs to exclude'
    )

    args = parser.parse_args()

    exclude_drugs = None
    if args.exclude_drugs:
        exclude_drugs = [d.strip() for d in args.exclude_drugs.split(',')]

    run_analysis_single_drug(
        args.training_drug,
        args.predictions,
        args.test_data,
        args.train_data,
        args.pkpd_path,
        args.output_dir,
        n_bootstrap=args.n_bootstrap,
        random_state=args.random_state,
        exclude_drugs=exclude_drugs
    )


if __name__ == '__main__':
    main()
