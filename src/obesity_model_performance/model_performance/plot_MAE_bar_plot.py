#!/usr/bin/env python3
"""Generate bar plot comparing baseline vs liraglutide-trained model MAE for GLP-1 drugs."""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def bootstrap_group_mae(y_true: np.ndarray, y_pred: np.ndarray, groups: pd.Series,
                        n_bootstrap: int = 10000, random_state: int = 42) -> dict:
    """Calculate bootstrap CI for group-level MAE."""
    rng = np.random.RandomState(random_state)
    df = pd.DataFrame({
        'ae': np.abs(y_true - y_pred),
        'group': groups.values
    })
    group_mae = df.groupby('group')['ae'].mean()
    unique_groups = df['group'].unique()
    n_groups = len(unique_groups)

    observed_gmae = group_mae.mean()
    bootstrap_values = [
        group_mae.loc[rng.choice(unique_groups, size=n_groups, replace=True)].mean()
        for _ in range(n_bootstrap)
    ]
    bootstrap_values = np.array(bootstrap_values)

    return {
        'mean': observed_gmae,
        'ci_lower': np.percentile(bootstrap_values, 2.5),
        'ci_upper': np.percentile(bootstrap_values, 97.5),
        'n_samples': len(y_true)
    }


def load_and_compute_mae(predictions_path: str, test_data_path: str, train_data_path: str,
                         drugs: list, n_bootstrap: int = 10000, random_state: int = 42) -> dict:
    """Load data and compute bootstrap MAE for each drug (model and baseline)."""
    predictions = pd.read_csv(predictions_path, sep='\t')
    test_data = pd.read_csv(test_data_path)
    train_data = pd.read_csv(train_data_path)

    if 'intervention' not in predictions.columns:
        predictions['intervention'] = test_data['intervention'].values

    mean_train_label = np.mean(train_data["result_translation"].values)

    results = {}
    for drug in drugs:
        target_test = test_data[test_data.intervention == drug].copy()
        target_pred = predictions[predictions.intervention == drug].copy()

        if len(target_test) == 0:
            continue

        groups = target_test['NCT Number'].reset_index(drop=True)
        y_true = target_pred["result_translation"].values
        y_pred_model = target_pred["prediction"].values
        y_pred_baseline = np.full(len(y_true), mean_train_label)

        results[drug] = {
            'model': bootstrap_group_mae(y_true, y_pred_model, groups, n_bootstrap, random_state),
            'baseline': bootstrap_group_mae(y_true, y_pred_baseline, groups, n_bootstrap, random_state)
        }
    return results


def create_bar_plot(results: dict, output_path: str):
    """Create bar plot with baseline (red) and model (green) bars, sorted by sample size."""
    drug_labels = {
        'semaglutide': 'Semaglutide', 'liraglutide': 'Liraglutide',
        'tirzepatide': 'Tirzepatide', 'survodutide': 'Survodutide',
        'medi0382': 'MEDI0382', 'exenatide': 'Exenatide'
    }

    # Sort drugs by sample size (descending)
    sorted_drugs = sorted(results.keys(), key=lambda d: results[d]['model']['n_samples'], reverse=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(sorted_drugs))
    width = 0.3

    baseline_means = [results[d]['baseline']['mean'] for d in sorted_drugs]
    baseline_errs = [[results[d]['baseline']['mean'] - results[d]['baseline']['ci_lower'] for d in sorted_drugs],
                     [results[d]['baseline']['ci_upper'] - results[d]['baseline']['mean'] for d in sorted_drugs]]
    model_means = [results[d]['model']['mean'] for d in sorted_drugs]
    model_errs = [[results[d]['model']['mean'] - results[d]['model']['ci_lower'] for d in sorted_drugs],
                  [results[d]['model']['ci_upper'] - results[d]['model']['mean'] for d in sorted_drugs]]

    ax.bar(x - width/2, baseline_means, width, yerr=baseline_errs, label='Baseline',
           color='#C44E52', capsize=4, error_kw={'elinewidth': 1.5, 'capthick': 1.5})
    ax.bar(x + width/2, model_means, width, yerr=model_errs, label='Liraglutide-trained Model',
           color='#55A868', capsize=4, error_kw={'elinewidth': 1.5, 'capthick': 1.5})

    # Add mean values in the middle of bars
    for i, (b_mean, m_mean) in enumerate(zip(baseline_means, model_means)):
        ax.text(x[i] - width/2, b_mean/2, f'{b_mean:.1f}', ha='center', va='center',
                fontsize=12)
        ax.text(x[i] + width/2, m_mean/2, f'{m_mean:.1f}', ha='center', va='center',
                fontsize=12)

    # X-axis labels with sample size
    x_labels = [f"{drug_labels.get(d, d)}\n(n={results[d]['model']['n_samples']})" for d in sorted_drugs]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10)

    ax.set_ylabel('Mean absolute error in translation gap predictions\n(Δ in preclinical and clinical body weight % change)', fontsize=11, fontweight='medium')
    ax.legend(loc='upper right', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    fig.text(0.5, 0.01, 'Error bars: 95% CI (bootstrap resampling by clinical study)',
             ha='center', fontsize=8, color='#888888', style='italic')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    for ext in ['pdf']:
        plt.savefig(f"{output_path}.{ext}", bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Bar plot saved to {output_path}.pdf")

def main():
    parser = argparse.ArgumentParser(description='Generate bar plot comparing baseline vs model MAE.')

    parser.add_argument('--predictions',
                        type=str,
                        required=True,
                        help='Path to predictions TSV')

    parser.add_argument('--test-data',
                        type=str,
                        required=True,
                        help='Path to test data CSV')

    parser.add_argument('--train-data',
                        type=str,
                        required=True,
                        help='Path to training data CSV')

    parser.add_argument('--output-path',
                        type=str,
                        required=True,
                        help='Output path (without extension)')

    parser.add_argument('--drugs',
                        type=str,
                        default='semaglutide,tirzepatide,survodutide,medi0382,exenatide',
                        help='Comma-separated list of drugs to evaluate')

    parser.add_argument('--n-bootstrap',
                        type=int,
                        default=10000,
                        help='Number of bootstrap iterations')

    parser.add_argument('--random-state',
                        type=int,
                        default=42,
                        help='Random seed')

    args = parser.parse_args()

    drugs = [d.strip() for d in args.drugs.split(',')]

    print("Computing bootstrap MAE for each drug...")
    results = load_and_compute_mae(args.predictions, args.test_data, args.train_data,
                                   drugs, args.n_bootstrap, args.random_state)

    print("\nResults:")
    print(f"{'Drug':<15} {'Baseline MAE':>15} {'Model MAE':>15} {'n':>8}")
    print("-" * 55)
    for drug in sorted(results.keys(), key=lambda d: results[d]['model']['n_samples'], reverse=True):
        r = results[drug]
        print(f"{drug:<15} {r['baseline']['mean']:>14.2f} {r['model']['mean']:>14.2f} {r['model']['n_samples']:>8}")

    print("\nCreating bar plot...")
    create_bar_plot(results, args.output_path)
    print("Done!")


if __name__ == '__main__':
    main()
