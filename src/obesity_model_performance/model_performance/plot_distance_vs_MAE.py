#!/usr/bin/env python3
"""Visualize PK/PD distance vs cross-drug performance.

Creates scatter plots showing the relationship between PK/PD distance
to training drug and model performance (MAE, R²) across test drugs.
Generates separate plots for MAE and R², one training drug at a time.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from scipy import stats


# Colors for each drug
DRUG_COLORS = {
    'Liraglutide': '#17BECF',   # Muted teal
    'Semaglutide': '#FF7F0E',   # Muted orange
    'Tirzepatide': '#D62728',   # Saddle brown
    'Survodutide': '#E377C2',   # Dark slate gray
    'Medi0382': '#8C564B',      # Dark orchid
    'Exenatide': '#2CA02C'       # Gold
}

MODEL_LABELS = {
    'lira_trained': 'Liraglutide',
    'sema_trained': 'Semaglutide',
}

DISTANCE_MODE_LABELS = {
    'both': 'PK/PD distance',
    'ec50_only': 'EC50 ratio distance',
    'halflife_only': 'Half-life distance',
}


def compute_drug_level_stats(
    nct_table: pd.DataFrame,
    outcome_col: str,
    distance_mode: str = 'both'
) -> pd.DataFrame:
    """Compute drug-level mean and 95% CI for an outcome.

    Args:
        nct_table: NCT-level metrics table.
        outcome_col: Name of the outcome column.

    Returns:
        DataFrame with drug-level statistics.
    """
    results = []

    for (model, drug), group in nct_table.groupby(['model', 'test_drug']):
        values = group[outcome_col].dropna().values
        distance = group[f'distance_{distance_mode}'].iloc[0]

        if len(values) == 0 or np.isnan(distance):
            continue

        mean_val = np.mean(values)
        n = len(values)

        if n >= 2:
            se = stats.sem(values)
            ci_low = mean_val - 1.96 * se
            ci_high = mean_val + 1.96 * se
        else:
            ci_low = mean_val
            ci_high = mean_val

        results.append({
            'model': model,
            'test_drug': drug,
            'distance': distance,
            'mean': mean_val,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'n_ncts': n
        })

    return pd.DataFrame(results)


def fit_linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Fit linear regression and return parameters for plotting.

    Args:
        x: Independent variable (distance).
        y: Dependent variable (performance metric).

    Returns:
        Tuple of (slope, intercept, x_line, y_line) for plotting.
    """
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]

    if len(x_valid) < 2:
        return np.nan, np.nan, np.array([]), np.array([])

    slope, intercept, r_value, p_value, std_err = stats.linregress(x_valid, y_valid)

    # Generate line points
    x_line = np.linspace(x_valid.min() - 0.1, x_valid.max() + 0.1, 100)
    y_line = intercept + slope * x_line

    return slope, intercept, x_line, y_line


def get_bootstrap_stats(inference_results: pd.DataFrame, model: str, outcome: str):
    """Get bootstrap beta0, beta1, CIs and p-value from inference results if available."""
    if inference_results is None:
        return None, None, None, None, None, None
    mask = (inference_results['model'] == model) & (inference_results['outcome'] == outcome)
    if mask.sum() > 0:
        row = inference_results.loc[mask].iloc[0]
        beta0 = row.get('beta0_hat', None)
        beta1 = row['beta1_hat']
        beta1_ci_low = row['beta1_boot_CI_low']
        beta1_ci_high = row['beta1_boot_CI_high']
        beta0_ci_low = row.get('beta0_boot_CI_low', None)
        beta0_ci_high = row.get('beta0_boot_CI_high', None)
        # Get p-value from p_boot column
        p_value = row.get('p_boot', None)
        return beta0, beta1, beta1_ci_low, beta1_ci_high, (beta0_ci_low, beta0_ci_high), p_value
    return None, None, None, None, None, None


def plot_points(ax, stats_df: pd.DataFrame):
    """Plot points with error bars using DRUG_COLORS for each drug."""
    for _, row in stats_df.iterrows():
        drug = row['test_drug']
        color = DRUG_COLORS.get(drug, '#333333')
        ax.errorbar(
            row['distance'], row['mean'],
            yerr=[[row['mean'] - row['ci_low']], [row['ci_high'] - row['mean']]],
            fmt='o', color=color, markersize=10,
            markeredgecolor='gray',
            capsize=4, capthick=1.5, elinewidth=1,
            alpha=1, zorder=3
        )


def plot_regression(ax, stats_df: pd.DataFrame, model: str, outcome: str,
                    inference_results: pd.DataFrame = None):
    """Plot regression line using bootstrap results.

    Line is always plotted in dark gray. Returns statistics for text annotation.

    Returns:
        dict with slope, ci_low, ci_high, p_value, x_line, y_line for annotation.
    """
    x = stats_df['distance'].values
    y = stats_df['mean'].values

    if len(x) < 2:
        return None

    # Get bootstrap stats (including beta0 and p-value)
    bootstrap_beta0, bootstrap_beta1, beta1_ci_low, beta1_ci_high, beta0_ci, p_value = get_bootstrap_stats(
        inference_results, model, outcome
    )

    if bootstrap_beta1 is not None:
        slope = bootstrap_beta1
        if bootstrap_beta0 is not None and not np.isnan(bootstrap_beta0):
            intercept = bootstrap_beta0
        else:
            x_mean, y_mean = np.mean(x), np.mean(y)
            intercept = y_mean - slope * x_mean
    else:
        slope, intercept, _, _ = fit_linear_regression(x, y)
        beta1_ci_low, beta1_ci_high = None, None
        p_value = None

    if slope is None or np.isnan(slope):
        return None

    x_line = np.linspace(x.min() - 0.1, x.max() + 0.1, 100)
    y_line = intercept + slope * x_line

    # Dark gray fitted line
    ax.plot(x_line, y_line, color='#444444', linestyle='--', linewidth=1,
            alpha=0.8, zorder=2)

    return {
        'slope': slope,
        'ci_low': beta1_ci_low,
        'ci_high': beta1_ci_high,
        'p_value': p_value,
        'x_line': x_line,
        'y_line': y_line
    }


def add_drug_labels(ax, stats_df: pd.DataFrame, y_offset: float = 0.02):
    """Add drug name labels above each point.

    Args:
        ax: Matplotlib axis.
        stats_df: DataFrame with drug statistics (must have test_drug, distance, mean columns).
        y_offset: Vertical offset for labels as fraction of y-axis range.
    """
    # Get y-axis range for offset calculation
    y_min, y_max = ax.get_ylim()
    offset = (y_max - y_min) * y_offset

    for _, row in stats_df.iterrows():
        drug = row['test_drug']
        ax.text(
            row['distance'],
            row['ci_high'] + offset,
            drug,
            ha='center', va='bottom',
            fontsize=9, fontweight='medium',
            color='#333333'
        )


def add_regression_stats_text(ax, reg_stats: dict, position: str = 'upper right'):
    """Add regression statistics text annotation near the fitted line.

    Args:
        ax: Matplotlib axis.
        reg_stats: Dictionary with slope, ci_low, ci_high, p_value from plot_regression.
        position: Position for the text ('upper right', 'upper left', 'lower right', 'lower left').
    """
    if reg_stats is None:
        return

    slope = reg_stats['slope']
    ci_low = reg_stats.get('ci_low')
    ci_high = reg_stats.get('ci_high')
    p_value = reg_stats.get('p_value')

    # Format slope with sign
    sign = '+' if slope >= 0 else ''
    text = f'β = {sign}{slope:.2f}'

    # Add CI if available
    if ci_low is not None and ci_high is not None:
        text += f'\n95% CI [{ci_low:.2f}, {ci_high:.2f}]'

    # Add p-value if available
    if p_value is not None:
        if p_value < 0.001:
            text += '\n(p < 0.001)'
        else:
            text += f'\n(p = {p_value:.3f})'

    # Determine position coordinates
    if position == 'upper right':
        x_pos, y_pos = 0.95, 0.95
        ha, va = 'right', 'top'
    elif position == 'upper left':
        x_pos, y_pos = 0.05, 0.95
        ha, va = 'left', 'top'
    elif position == 'lower right':
        x_pos, y_pos = 0.95, 0.05
        ha, va = 'right', 'bottom'
    else:  # lower left
        x_pos, y_pos = 0.05, 0.05
        ha, va = 'left', 'bottom'

    ax.text(
        x_pos, y_pos, text,
        transform=ax.transAxes,
        ha=ha, va=va,
        fontsize=9,
        color='#333333',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8)
    )


def style_axis(ax, xlabel: str, ylabel: str):
    """Style axis with consistent formatting (no legend)."""
    ax.set_xlabel(xlabel, fontsize=9, fontweight='medium')
    ax.set_ylabel(ylabel, fontsize=9, fontweight='medium')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.tick_params(axis='both', labelsize=9.5)
    ax.grid(True, linestyle='--', alpha=0.3, zorder=0)


def create_mae_plot(
    nct_table: pd.DataFrame,
    output_path: str,
    model: str,
    inference_results: pd.DataFrame = None,
    distance_mode: str = 'both'
):
    """Create MAE vs PK/PD distance plot for a single training drug model.

    Args:
        nct_table: NCT-level metrics table with columns:
            model, test_drug, nct_id, distance, MAE_nct.
        output_path: Path to save the plot (without extension).
        model: Model identifier (e.g., 'lira_trained' or 'sema_trained').
        exclude_drugs: List of drugs to exclude from the plot.
        inference_results: DataFrame with bootstrap inference results.
        distance_mode: Which distance metric is used ('both', 'ec50_only', 'halflife_only').
    """

    # Filter to target model and valid distances
    nct_table = nct_table[(nct_table['model'] == model) & (~nct_table[f'distance_{distance_mode}'].isna())]

    if len(nct_table) == 0:
        print(f"No data found for model '{model}'")
        return

    mae_stats = compute_drug_level_stats(nct_table, 'MAE_nct', distance_mode=distance_mode)
    label = MODEL_LABELS.get(model, model)
    distance_label = DISTANCE_MODE_LABELS.get(distance_mode, 'PK/PD distance')

    fig, ax = plt.subplots(figsize=(6, 5))

    plot_points(ax, mae_stats)
    reg_stats = plot_regression(ax, mae_stats, model, 'MAE_nct', inference_results)

    style_axis(ax, f'{distance_label} to {label}',
               'Mean absolute error in translation gap predictions\n(Δ in preclinical and clinical body weight % change)')

    # Add drug name labels above points
    add_drug_labels(ax, mae_stats)

    # Add regression statistics text
    add_regression_stats_text(ax, reg_stats, position='upper left')

    plt.tight_layout()

    for ext in ['pdf']:
        plt.savefig(f"{output_path}.{ext}", bbox_inches='tight', dpi=300)
    plt.close()

    print(f"MAE plot saved to {output_path}.pdf")

def main():
    parser = argparse.ArgumentParser(
        description='Visualize PK/PD distance vs cross-drug performance. '
                    'Generates separate MAE, MAE reduction, and R² plots for a single training drug.'
    )
    parser.add_argument(
        '--nct-table', type=str, required=True,
        help='Path to NCT-level metrics CSV'
    )
    parser.add_argument(
        '--output-path', type=str, required=True,
        help='Output path prefix for plots (without extension). '
             'Will generate <output_path>_mae and <output_path>_r2 files.'
    )
    parser.add_argument(
        '--training-drug', type=str, required=True,
        choices=['liraglutide', 'semaglutide'],
        help='Training drug to analyze (liraglutide or semaglutide)'
    )
    parser.add_argument(
        '--inference-results', type=str, default=None,
        help='Path to bootstrap inference results CSV (for beta values)'
    )
    parser.add_argument(
        '--distance-mode', type=str, default='both',
        choices=['both', 'ec50_only', 'halflife_only'],
        help='Distance mode to use (default: both)'
    )

    args = parser.parse_args()

    # Map training drug name to model identifier
    model_map = {
        'liraglutide': 'lira_trained'
    }
    model = model_map[args.training_drug]

    nct_table = pd.read_csv(args.nct_table)

    inference_results = None
    if args.inference_results:
        inference_results = pd.read_csv(args.inference_results)

    # Generate MAE plot
    print(f"Creating MAE plot for {args.training_drug}-trained model...")
    create_mae_plot(
        nct_table,
        f"{args.output_path}_mae",
        model,
        inference_results,
        distance_mode=args.distance_mode,
    )
    print("Done!")


if __name__ == '__main__':
    main()
