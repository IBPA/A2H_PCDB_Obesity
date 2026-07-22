#!/usr/bin/env python3
"""
Script to generate figures and statistics for the A2H Obesity dataset analysis.

This script generates:
- Preclinical trial outcomes distribution plot
- Clinical trial outcomes distribution plot
- A2H translation outcome (delta) distribution plot
- Combined preclinical/clinical distribution plot
- Bar chart for A2H obesity preclinical treatments

It also prints statistics for preclinical/clinical trial arms and studies.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def load_data(data_dir: Path):
    """Load the preclinical, clinical, and A2H datasets."""
    a2h_preclinical_only = pd.read_csv(
        data_dir / "preclinical_arms.csv", dtype=str, keep_default_na=False
    )
    a2h_clinical_only = pd.read_csv(
        data_dir / "clinical_arms.csv", dtype=str, keep_default_na=False
    )
    a2h_dataset = pd.read_csv(
        data_dir / "obesity_a2h_dataset.csv", dtype=str, keep_default_na=False
    )
    return a2h_preclinical_only, a2h_clinical_only, a2h_dataset


def plot_one_dist(
    data,
    title,
    output_dir: Path,
    hist_color="green",
    line_color="red",
    bin_n=50,
    x_label="change of body weight",
):
    """Plot a single distribution with histogram and KDE."""
    sns.set_theme(style="whitegrid")

    data = [float(d) for d in data]
    plt.subplots(figsize=(8, 6))
    g = sns.histplot(
        data,
        color=hist_color,
        alpha=0.5,
        bins=bin_n,
    )
    g.set(
        xlabel=x_label,
        ylabel="Count",
    )

    g2 = plt.twinx()
    sns.kdeplot(
        data,
        color=line_color,
        linestyle="--",
        legend=False,
        ax=g2,
    )
    g2.set(
        ylabel="Density",
    )
    ax = plt.gca()

    ticks = ax.get_xticks()
    ax.set_xticklabels([f"{int(t)}%" for t in ticks])
    plt.title(title)
    plt.savefig(output_dir / f"{title}.svg")
    plt.close()


def plot_delta_dist(delta_data, output_dir: Path, bin_n=50):
    """Plot the translation outcome (delta) distribution."""
    delta_data = [float(d) for d in delta_data]

    fig, ax1 = plt.subplots(figsize=(7, 5))
    sns.histplot(delta_data, bins=bin_n, ax=ax1, color="skyblue")
    ax1.set_ylabel("Count")

    ax2 = ax1.twinx()
    kde_line = sns.kdeplot(delta_data, color="red", ax=ax2, linestyle="-")
    ax2.set_ylabel("Density")

    ax = plt.gca()
    ticks = ax.get_xticks()
    ax.set_xticklabels([f"{int(t)}%" for t in ticks], fontsize=12)

    sigma = np.array(delta_data).std()
    mean = np.array(delta_data).mean()
    print(f"A2H Translation Outcome - Mean: {mean:.2f}")
    print(f"A2H Translation Outcome - Std: {sigma:.2f}")

    plt.tight_layout(pad=2.0)
    plt.savefig(output_dir / "a2h_preclinical_clinical_delta_dist.svg")
    plt.close()


def plot_combined_dist(preclinical_data, clinical_data, output_dir: Path):
    """Plot combined preclinical and clinical distributions."""
    preclinical_data = [float(d) for d in preclinical_data]
    clinical_data = [float(d) for d in clinical_data]

    combined_min = min(min(preclinical_data), min(clinical_data))
    combined_max = max(max(preclinical_data), max(clinical_data))
    bins = np.linspace(combined_min, combined_max, 11)

    sns.set_theme(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    sns.histplot(
        preclinical_data,
        bins=bins,
        color="#0066CC",
        alpha=0.8,
        label="Preclinical Trials",
        ax=ax1,
        stat="count",
    )
    sns.histplot(
        clinical_data,
        bins=bins,
        color="#66CC00",
        alpha=0.6,
        label="Clinical Trials",
        ax=ax1,
        stat="count",
    )

    ax2.set_ylabel("Density")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax = plt.gca()
    ticks = ax.get_xticks()
    ax.set_xticklabels([f"{int(t)}%" for t in ticks])
    ax1.set_xlabel("Change of body weight (%)")
    plt.title("Preclinical Trials & Clinical Trials Outcome Measure")
    plt.tight_layout()
    plt.savefig(output_dir / "a2h_preclinical_clinical_combined_dist.svg")
    plt.close()


def draw_barchart(values, labels, title, output_dir: Path, figsize=(5, 4), colors=None):
    """Draw a horizontal bar chart for treatment distributions."""
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=figsize)

    if colors is None:
        colors = sns.color_palette("Set2", len(labels))
        sns.barplot(x=values, y=labels, palette=colors, ax=ax)
    else:
        sns.barplot(x=values, y=labels, color=colors, ax=ax)

    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_xlabel("Percentage of animal-to-human arms (%)", fontsize=12)
    ax.set_ylabel("Treatment", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, v in enumerate(values):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=11)

    plt.savefig(output_dir / f"{title}.svg")
    plt.close()


def plot_treatment_barchart(a2h_dataset: pd.DataFrame, output_dir: Path):
    """Plot bar chart showing treatment distributions in A2H dataset."""
    drugs = a2h_dataset["intervention"].value_counts().index.to_list()
    drugs = [drug.capitalize() for drug in drugs]

    counts = a2h_dataset["intervention"].value_counts().to_list()
    percentages = np.array(counts) / len(a2h_dataset) * 100

    top_values = percentages[:5].tolist()
    top_labels = drugs[:5]
    others_value = 100 - sum(top_values)
    values_bar = top_values + [others_value]
    labels_bar = top_labels + ["Others"]

    draw_barchart(
        values_bar,
        labels_bar,
        title="a2h_obesity_treatment_dist",
        output_dir=output_dir,
    )


def print_statistics(
    a2h_preclinical_only: pd.DataFrame,
    a2h_clinical_only: pd.DataFrame,
    a2h_dataset: pd.DataFrame,
):
    """Print statistics for preclinical, clinical, and A2H datasets."""
    print("=" * 50)
    print("PRECLINICAL TRIAL STATISTICS")
    print("=" * 50)
    print(f"Total preclinical arms: {len(a2h_preclinical_only)}")
    print(f"Total preclinical articles: {a2h_preclinical_only['pmcid'].nunique()}")

    print()
    print("=" * 50)
    print("CLINICAL TRIAL STATISTICS")
    print("=" * 50)
    print(f"Total clinical arms: {len(a2h_clinical_only)}")
    print(f"Total clinical studies: {a2h_clinical_only['NCT Number'].nunique()}")

    print()
    print("=" * 50)
    print("A2H DATASET STATISTICS")
    print("=" * 50)
    print(f"Total A2H trial arms: {len(a2h_dataset)}")
    print()


def main():
    """Main function to generate all figures and print statistics."""
    script_dir = Path(__file__).parent.resolve()
    data_dir = script_dir / "../../data/obesity"
    output_dir = script_dir / "../../outputs/obesity_eda"
    
    if not output_dir.exists():
        output_dir.mkdir(exist_ok=True)

    print("Loading data...")
    a2h_preclinical_only, a2h_clinical_only, a2h_dataset = load_data(data_dir)

    a2h_preclinical_only["preclinical_outcome"] = a2h_preclinical_only[
        "preclinical_outcome"
    ].astype(float)
    a2h_clinical_only["clinical_outcome"] = a2h_clinical_only[
        "clinical_outcome"
    ].astype(float)
    a2h_dataset["translation_outcome"] = a2h_dataset["translation_outcome"].astype(
        float
    )

    print_statistics(a2h_preclinical_only, a2h_clinical_only, a2h_dataset)

    print("Generating preclinical distribution plot...")
    preclinical_data = a2h_preclinical_only["preclinical_outcome"].to_list()
    plot_one_dist(
        preclinical_data,
        "Obesity Preclinical Trial Outcomes (%)",
        output_dir=output_dir,
        bin_n=10,
        hist_color="blue",
        line_color="darkblue",
    )

    print("Generating clinical distribution plot...")
    clinical_data = a2h_clinical_only["clinical_outcome"].to_list()
    plot_one_dist(
        clinical_data,
        "Obesity clinical Trial Outcomes (%)",
        output_dir=output_dir,
        bin_n=10,
        hist_color="green",
        line_color="green",
    )

    print("Generating A2H delta distribution plot...")
    delta_data = a2h_dataset["translation_outcome"].to_list()
    plot_delta_dist(delta_data, output_dir=output_dir, bin_n=10)

    print("Generating combined distribution plot...")
    plot_combined_dist(preclinical_data, clinical_data, output_dir=output_dir)

    print("Generating treatment bar chart...")
    plot_treatment_barchart(a2h_dataset, output_dir=output_dir)

    print()
    print("=" * 50)
    print("All figures have been saved to:", output_dir)
    print("=" * 50)
    print("Generated files:")
    print("  - Obesity Preclinical Trial Outcomes (%).svg")
    print("  - Obesity clinical Trial Outcomes (%).svg")
    print("  - a2h_preclinical_clinical_delta_dist.svg")
    print("  - a2h_preclinical_clinical_combined_dist.svg")
    print("  - a2h_obesity_treatment_dist.svg")


if __name__ == "__main__":
    main()
