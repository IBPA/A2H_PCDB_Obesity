"""
Script to generate 2D plot of log(half life) vs log(EC50 ratio to native GLP-1)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_drug_colors():
    """Return drug-specific colors (RGBA -> RGB)."""
    drugs = ["Tirzepatide", "Semaglutide",
             "Survodutide", "Medi0382",
             "Liraglutide", 'Exenatide']
    colors = np.array([
        [0.83921569, 0.15294118, 0.15686275, 1.0],
        [1.0, 0.4980392156862745, 0.054901960784313725, 1.0],
        [0.8901960784313725, 0.4666666666666667, 0.7607843137254902, 1.0],
        [0.54901961, 0.3372549, 0.29411765, 1.0],
        [0.09019608, 0.74509804, 0.81176471, 1.0],
        [0.17254902, 0.62745098, 0.17254902, 1.0]
    ])
    return {drug: tuple(rgba[:3]) for drug, rgba in zip(drugs, colors)}


def plot_pkpd_profile(data_path: str, output_path: str):
    """Generate 2D plot of log(half life) vs log(EC50 ratio)."""
    color_dict = get_drug_colors()
    default_color = (0.5, 0.5, 0.5)

    df = pd.read_csv(data_path)

    _, ax = plt.subplots(figsize=(7, 5))

    for _, row in df.iterrows():
        drug = row['Drug']
        log_half_life = np.log10(row['half_life(hrs)'])
        log_ec50_ratio = np.log10(row['EC50_ratio'])
        color = color_dict.get(drug, default_color)

        ax.scatter(log_half_life, log_ec50_ratio, c=[color], s=200,
                   label=drug, edgecolors='black', linewidths=0.5)
        ax.annotate(drug, (log_half_life, log_ec50_ratio),
                    textcoords="offset points", xytext=(-38, -18), fontsize=9)

    ax.set_xlabel('log$_{10}$(Half Life (hours))', fontsize=12)
    ax.set_ylabel('log$_{10}$(EC50 ratio to native GLP-1)', fontsize=12)
    ax.set_title('PK/PD Profile: Half Life vs EC50 Ratio', fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--')
    # ax.legend(loc='best', framealpha=0.5)
    plt.tight_layout()

    # Save in multiple formats
    output_base = Path(output_path)
    for ext in ['.png', '.svg']:
        save_path = output_base.with_suffix(ext)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate 2D plot of log(half life) vs log(EC50 ratio)'
    )
    parser.add_argument('--data-path', required=True,
                        help='Path to pkpd_profiles.csv')
    parser.add_argument('--output-path', required=True,
                        help='Output path (without extension)')
    args = parser.parse_args()

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    plot_pkpd_profile(args.data_path, args.output_path)


if __name__ == '__main__':
    main()
