#!/usr/bin/env python3
"""
Plot grid search results for regression models.

This script loads grid search results from cross-validation and creates a box plot
comparing MAE across different regressors against a baseline (mean prediction).

Output:
- grid_search_performance.svg: Box plot comparing regressor performance
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import click

from pathlib import Path


def load_grid_search_results(rgs_dir) -> pd.DataFrame:
    """Load grid search results from CSV file.

    Args:
        rgs_dir: Path to the RGS output directory.

    Returns:
        DataFrame containing grid search results.
    """
    gs_result = pd.read_csv(f"{rgs_dir}/regressors/grid_search_results.csv")
    return gs_result


def load_train_data(data_dir) -> pd.DataFrame:
    """Load training data with labels.

    Args:
        data_dir: Path to the data processing splits directory.

    Returns:
        DataFrame containing training data with labels.
    """
    train_data = pd.read_csv(f"{data_dir}/data_train_with_labels.csv")
    return train_data


def group_mae(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> float:
    """Calculate mean of per-group MAE values.

    Args:
        y_true: Array of true target values.
        y_pred: Array of predicted values.
        groups: Array of group identifiers (e.g., NCT Number).

    Returns:
        Mean of per-group MAE values.
    """
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'group': groups})
    df['ae'] = np.abs(df['y_true'] - df['y_pred'])
    group_maes = df.groupby('group')['ae'].mean()
    return float(group_maes.mean())


def calculate_baseline_mae(train_data: pd.DataFrame) -> float:
    """Calculate baseline MAE using mean prediction.

    The baseline predicts the mean of the training target for all samples.

    Args:
        train_data: Training DataFrame with 'result_translation' and 'NCT Number' columns.

    Returns:
        Baseline MAE value.
    """
    y_pred = np.array([train_data["result_translation"].mean()] * len(train_data))
    y_true = train_data['result_translation'].values
    groups = train_data['NCT Number'].values
    baseline = group_mae(y_true, y_pred, groups)
    return baseline


def plot_grid_search_performance(
    gs_result: pd.DataFrame,
    baseline_mae: float,
    title: str,
    output_path: Path,
) -> None:
    """Create box plot comparing regressor performance.

    Args:
        gs_result: DataFrame with grid search results.
        baseline_mae: Baseline MAE value for comparison.
        title: Plot title.
        output_path: Path to save the output figure.
    """
    # Select and rename columns
    gs_result = gs_result[['model', 'os_method', 'mean_test_group_mae']].copy()
    gs_result = gs_result.rename(
        columns={
            'mean_test_group_mae': 'Negative MAE',
            'model': 'Regressor',
            'os_method': 'Oversampling',
        }
    )

    # Convert negative MAE to positive
    gs_result['MAE'] = -gs_result['Negative MAE']

    # Map model abbreviations to full names
    gs_result['Regressor'] = gs_result['Regressor'].map({
        'xg': 'XGBoost',
        'ada': 'ADA',
        'mlp': 'MLP',
        'rf': 'RF',
        'gb': 'GBoost',
    })

    # Create plot
    sns.set_theme(style='whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6))

    g = sns.boxplot(
        data=gs_result,
        x='Regressor',
        y='MAE',
        hue="Regressor",
        order=['XGBoost', 'RF', 'GBoost', 'MLP', 'ADA'],
        showfliers=False,
        ax=ax,
    )

    # Add baseline line
    g.axhline(baseline_mae, ls='--', color='black', label='Baseline')

    # Labels and title
    g.set_title(title)
    g.set_ylabel('Mean Absolute Error (MAE)')
    g.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")
    plt.close()

@click.command()
@click.argument(
    'path-rgs-dir',
    type=click.Path(exists=True),
)
@click.argument(
    'path-data-dir',
    type=click.Path(exists=True),
)
@click.option(
    '--training-drug',
    default='liraglutide',
    help='Drug used for training the model. Default: liraglutide.',
)

def main(path_rgs_dir, path_data_dir, training_drug):
    """Main function to plot grid search results."""
    rgs_dir = path_rgs_dir
    data_dir = path_data_dir
    print(f"Loading grid search results from: {rgs_dir}")
    gs_result = load_grid_search_results(rgs_dir)
    print(f"Grid search results: {len(gs_result)} rows")

    print(f"\nLoading training data from: {data_dir}")
    train_data = load_train_data(data_dir)
    print(f"Training data: {len(train_data)} rows")

    print("\nCalculating baseline MAE...")
    baseline_mae = calculate_baseline_mae(train_data)
    print(f"Baseline MAE: {baseline_mae:.4f}")

    print("\nGenerating grid search performance plot...")
    os.makedirs(f"outputs/ml_training_{training_drug}/visualizations", exist_ok=True)
    output_path = f"outputs/ml_training_{training_drug}/visualizations/grid_search_performance.svg"
    plot_grid_search_performance(
        gs_result,
        baseline_mae,
        title='3-Fold Cross-Validation MAE for Regressors',
        output_path=output_path,
    )

    print("\n" + "=" * 50)
    print("Done!")
    print("=" * 50)


if __name__ == '__main__':
    main()
