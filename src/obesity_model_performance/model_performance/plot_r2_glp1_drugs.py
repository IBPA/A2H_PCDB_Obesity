#!/usr/bin/env python3
"""Generate R² scatter plots for GLP-1 test drugs.

Creates a two-panel plot showing R² performance:
- Left panel: Semaglutide
- Right panel: Other GLP-1 drugs (using liraglutide-trained model predictions)
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score


# Color map for each drug
DRUG_COLORS = {
    'exenatide': [0.17254902, 0.62745098, 0.17254902, 1.0],
    'tirzepatide': [0.83921569, 0.15294118, 0.15686275, 1.0],
    'semaglutide': [1.0, 0.49803922, 0.05490196, 1.0],
    'survodutide': [0.89019608, 0.46666667, 0.76078431, 1.0],
    'medi0382': [0.54901961, 0.3372549, 0.29411765, 1.0],
    'liraglutide': [0.09019608, 0.74509804, 0.81176471, 1.0]
}


def plot_r2_two_panels(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    drug_col: str,
    left_drug: str,
    right_drugs: list,
    left_title: str = None,
    right_title: str = None,
    output_path: str = None,
    base_width: float = 6.0,
    height: float = 8.0,
    dot_size: int = 120,
):
    """Create R² scatter plots with two panels.

    Args:
        df: DataFrame with predictions.
        y_true_col: Column name for actual values.
        y_pred_col: Column name for predicted values.
        drug_col: Column name for drug/intervention.
        left_drug: Drug to show in left panel.
        right_drugs: List of drugs to show in right panel.
        left_title: Title for left panel.
        right_title: Title for right panel.
        output_path: Base path for saving plots (without extension).
        base_width: Width per subplot in inches.
        height: Figure height in inches.
        dot_size: Scatter point size.

    Returns:
        Tuple of (figure, axes).
    """
    sns.set_theme(style="whitegrid", context="talk")

    all_drugs = [left_drug] + right_drugs

    # Compute global axis limits across all drugs
    all_vals = np.concatenate([
        df[df[drug_col].isin(all_drugs)][y_true_col].dropna().values,
        df[df[drug_col].isin(all_drugs)][y_pred_col].dropna().values
    ])

    if len(all_vals) == 0:
        print("No data found for the specified drugs.")
        return None, None

    global_min = np.nanmin(all_vals)
    global_max = np.nanmax(all_vals)
    rng = global_max - global_min if global_max > global_min else 1.0
    margin = 0.05 * rng
    global_min -= margin
    global_max += margin

    # Create figure with 2 panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(base_width * 2, height),
                                    sharex=True, sharey=True)

    def plot_panel(ax, drugs, title):
        """Plot a single panel."""
        dsub = df[df[drug_col].isin(drugs)].dropna(subset=[y_true_col, y_pred_col])

        # Calculate R² for all drugs in this panel combined
        if len(dsub) >= 2 and dsub[y_true_col].nunique() > 1:
            r2 = r2_score(dsub[y_true_col], dsub[y_pred_col])
        else:
            r2 = np.nan

        # Plot each drug with its color
        for drug in drugs:
            drug_data = dsub[dsub[drug_col] == drug]
            if len(drug_data) > 0 and drug in DRUG_COLORS:
                sns.scatterplot(
                    data=drug_data,
                    x=y_true_col,
                    y=y_pred_col,
                    ax=ax,
                    s=dot_size,
                    marker='o',
                    color=DRUG_COLORS[drug],
                    edgecolor="#4D4D4D",
                    linewidth=0.7,
                    alpha=0.8,
                )

        # Identity line
        ax.plot([global_min, global_max], [global_min, global_max],
                linestyle="--", color="gray", linewidth=1)

        ax.set_xlim(global_min, global_max)
        ax.set_ylim(global_min, global_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Observed translation gap\n(Δ in preclinical and clinical body weight % change)", fontsize=14)
        ax.set_ylabel("Predicted translation gap\n(Δ in preclinical and clinical body weight % change)", fontsize=14)

        n_points = len(dsub)
        ax.set_title(f"{title} (n={n_points})", fontsize=12, pad=12)

    # Plot left panel (single drug)
    plot_panel(ax1, [left_drug], left_title or left_drug.capitalize())

    # Plot right panel (multiple drugs)
    plot_panel(ax2, right_drugs, right_title or "Other GLP-1 Drugs")

    # Create legend with all drugs
    legend_handles = []
    legend_labels = []

    for drug in all_drugs:
        if drug in DRUG_COLORS:
            handle = plt.Line2D(
                [], [], linestyle="", marker='o',
                markersize=14, markerfacecolor=DRUG_COLORS[drug],
                markeredgecolor='black', markeredgewidth=1.0, label=drug
            )
            legend_handles.append(handle)
            legend_labels.append(drug)

    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc='center right',
        bbox_to_anchor=(0.99, 0.5),
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        edgecolor="gray",
        facecolor="white",
        fontsize=14
    )

    plt.subplots_adjust(right=0.85)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    plt.savefig(f"{output_path}.pdf", bbox_inches='tight', dpi=300)
    print(f"R² plot saved to {output_path}.pdf")

    return fig, (ax1, ax2)


def main():
    parser = argparse.ArgumentParser(
        description='Generate R² scatter plots for GLP-1 test drugs.'
    )
    parser.add_argument(
        '--predictions', type=str, required=True,
        help='Path to predictions TSV file'
    )
    parser.add_argument(
        '--test-data', type=str, required=True,
        help='Path to test data CSV file'
    )
    parser.add_argument(
        '--output-path', type=str, required=True,
        help='Output path for plot (without extension)'
    )
    parser.add_argument(
        '--left-drug', type=str, default='semaglutide',
        help='Drug to show in left panel (default: semaglutide)'
    )
    parser.add_argument(
        '--right-drugs', type=str, default='tirzepatide,survodutide,medi0382,exenatide',
        help='Comma-separated list of drugs for right panel'
    )
    parser.add_argument(
        '--left-title', type=str, default='Semaglutide',
        help='Title for left panel'
    )
    parser.add_argument(
        '--right-title', type=str, default='Other GLP-1 Drugs',
        help='Title for right panel'
    )
    parser.add_argument(
        '--dot-size', type=int, default=70,
        help='Size of scatter points (default: 120)'
    )

    args = parser.parse_args()

    # Load data
    print("Loading data...")
    predictions = pd.read_csv(args.predictions, sep='\t')
    test_data = pd.read_csv(args.test_data)

    # Add intervention column to predictions if not present
    if 'intervention' not in predictions.columns:
        predictions['intervention'] = test_data['intervention'].values

    # Parse right panel drugs
    right_drugs = [d.strip() for d in args.right_drugs.split(',')]

    # Create output directory if needed
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Generate plot
    print("Generating R² plot...")
    plot_r2_two_panels(
        df=predictions,
        y_true_col="result_translation",
        y_pred_col="prediction",
        drug_col="intervention",
        left_drug=args.left_drug,
        right_drugs=right_drugs,
        left_title=args.left_title,
        right_title=args.right_title,
        output_path=args.output_path,
        dot_size=args.dot_size,
    )

    plt.close()
    print("Done!")


if __name__ == '__main__':
    main()
