#!/usr/bin/env python3
"""
Plot translation_outcome vs preclinical_dosage_duration for liraglutide and semaglutide in DIO mice.

Duration is grouped into four categories: Short (<4 weeks), Medium (4-8 weeks),
Long (8-12 weeks), and Very Long (>12 weeks).

Output:
- liraglutide_mice_plot.svg: Scatter plot for liraglutide in DIO mice
- semaglutide_mice_plot.svg: Scatter plot for semaglutide in DIO mice (optional)
- combined_mice_plot.svg: Combined scatter plot for both interventions (optional)
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Define colors for each intervention
COLORS = {
    'liraglutide': sns.color_palette("Set2")[0],
    'semaglutide': sns.color_palette("Set2")[1]
}

# Define category order
CATEGORY_ORDER = [
    "≤2 weeks",
    "2–4 weeks",
    "4–6 weeks",
    "6–8 weeks",
    ">8 weeks",
]


def load_data(data_path: Path) -> pd.DataFrame:
    """Load the obesity A2H dataset."""
    df = pd.read_csv(data_path, dtype=str, keep_default_na=False)
    return df


def categorize_duration(days: float) -> str:
    """Categorize treatment duration into groups."""
    if days <= 14:
        return "≤2 weeks"
    elif days <= 28:
        return "2–4 weeks"
    elif days <= 42:
        return "4–6 weeks"
    elif days <= 56:
        return "6–8 weeks"
    else:
        return ">8 weeks"


def create_plot(df: pd.DataFrame, intervention: str, output_dir: Path) -> None:
    """Create scatter plot for a specific intervention.

    Args:
        df: The obesity A2H dataset DataFrame.
        intervention: The intervention name ('liraglutide' or 'semaglutide').
        output_dir: Directory to save the output plot.
    """
    # Filter for intervention, mice, and diet-induced disease models
    mask = (
        (df['intervention'] == intervention) &
        (df['preclinical_animal_species'] == 'mice') &
        (df['preclinical_disease_model'].str.startswith('diet:', na=False))
    )
    filtered_df = df[mask].copy()

    # Convert numeric columns
    filtered_df['preclinical_dosage_duration(days)'] = pd.to_numeric(
        filtered_df['preclinical_dosage_duration(days)'], errors='coerce'
    )
    filtered_df['translation_outcome'] = pd.to_numeric(
        filtered_df['translation_outcome'], errors='coerce'
    )
    # Remove rows with missing values in the columns we need
    filtered_df = filtered_df.dropna(subset=[
        'preclinical_dosage_duration(days)',
        'translation_outcome',
    ])

    print(f"\n{intervention.capitalize()} - Number of data points: {len(filtered_df)}")

    # Create duration categories
    filtered_df['duration_category'] = filtered_df['preclinical_dosage_duration(days)'].apply(categorize_duration)

    # Convert to categorical with order
    filtered_df['duration_category'] = pd.Categorical(
        filtered_df['duration_category'],
        categories=CATEGORY_ORDER,
        ordered=True
    )

    # Map categories to numeric x positions for plotting
    category_to_x = {cat: i for i, cat in enumerate(CATEGORY_ORDER)}
    filtered_df['x_pos'] = filtered_df['duration_category'].map(category_to_x).astype(float)

    # Add jitter to x position for better visualization
    np.random.seed(42)
    jitter = np.random.uniform(-0.25, 0.25, len(filtered_df))
    x = filtered_df['x_pos'].values + jitter

    y = filtered_df['translation_outcome']

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 7))

    # Create scatter plot with solid circles
    ax.scatter(
        x, y,
        color=COLORS[intervention],
        s=80,
        marker='o',
        edgecolors='black',
        linewidths=0.5,
        alpha=0.8
    )

    # Set x-axis ticks and labels
    ax.set_xticks(range(len(CATEGORY_ORDER)))
    ax.set_xticklabels(CATEGORY_ORDER)
    ax.tick_params(axis='x', labelsize=14)
    ax.set_xlim(-0.5, len(CATEGORY_ORDER) - 0.5)

    # Labels and title
    ax.set_xlabel('Preclinical treatment duration', fontsize=14)
    ax.set_ylabel('Translation gap\n(δ in preclinical and clinical body weight % change)', fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_title(f'DIO mice treated with {intervention}', fontsize=14)

    # Grid for readability (only horizontal)
    ax.yaxis.grid(True, alpha=0.3)

    # Add gray dashed vertical lines to separate groups
    for i in range(1, len(CATEGORY_ORDER)):
        ax.axvline(x=i - 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    # Print category counts
    print("Points per category:")
    print(filtered_df['duration_category'].value_counts().sort_index())

    plt.tight_layout()

    # Save the plot
    output_path = output_dir / f'{intervention}_mice_plot.svg'
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")

    plt.close()


def create_combined_plot(df: pd.DataFrame, output_dir: Path) -> None:
    """Create combined scatter plot for both interventions.

    Args:
        df: The obesity A2H dataset DataFrame.
        output_dir: Directory to save the output plot.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    for intervention in ['liraglutide', 'semaglutide']:
        # Filter for intervention, mice, and diet-induced disease models
        mask = (
            (df['intervention'] == intervention) &
            (df['preclinical_animal_species'] == 'mice') &
            (df['preclinical_disease_model'].str.startswith('diet:', na=False))
        )
        filtered_df = df[mask].copy()

        # Convert numeric columns
        filtered_df['preclinical_dosage_duration(days)'] = pd.to_numeric(
            filtered_df['preclinical_dosage_duration(days)'], errors='coerce'
        )
        filtered_df['translation_outcome'] = pd.to_numeric(
            filtered_df['translation_outcome'], errors='coerce'
        )

        # Remove rows with missing values
        filtered_df = filtered_df.dropna(subset=[
            'preclinical_dosage_duration(days)',
            'translation_outcome',
        ])

        print(f"\n{intervention.capitalize()} - Number of data points: {len(filtered_df)}")

        # Create duration categories
        filtered_df['duration_category'] = filtered_df['preclinical_dosage_duration(days)'].apply(categorize_duration)

        # Convert to categorical with order
        filtered_df['duration_category'] = pd.Categorical(
            filtered_df['duration_category'],
            categories=CATEGORY_ORDER,
            ordered=True
        )

        # Map categories to numeric x positions
        category_to_x = {cat: i for i, cat in enumerate(CATEGORY_ORDER)}
        filtered_df['x_pos'] = filtered_df['duration_category'].map(category_to_x).astype(float)

        # Add jitter - offset liraglutide left, semaglutide right
        np.random.seed(42)
        jitter = np.random.uniform(-0.15, 0.15, len(filtered_df))
        offset = -0.2 if intervention == 'liraglutide' else 0.2
        x = filtered_df['x_pos'].values + jitter + offset

        y = filtered_df['translation_outcome']

        # Create scatter plot
        ax.scatter(
            x, y,
            color=COLORS[intervention],
            s=70,
            marker='o',
            edgecolors='black',
            linewidths=0.5,
            alpha=0.8,
            label=intervention.capitalize()
        )

    # Set x-axis ticks and labels
    ax.set_xticks(range(len(CATEGORY_ORDER)))
    ax.set_xticklabels(CATEGORY_ORDER)
    ax.tick_params(axis='x', labelsize=14)
    ax.set_xlim(-0.5, len(CATEGORY_ORDER) - 0.5)

    # Labels and title
    ax.set_xlabel('Preclinical treatment duration', fontsize=14)
    ax.set_ylabel('Translation gap\n(δ in preclinical and clinical body weight % change)', fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_title('Liraglutide vs Semaglutide in DIO-Induced Mice:\nTreatment Duration & Translation Gap', fontsize=14)

    # Grid for readability
    ax.yaxis.grid(True, alpha=0.3)

    # Add gray dashed vertical lines to separate groups
    for i in range(1, len(CATEGORY_ORDER)):
        ax.axvline(x=i - 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    # Add legend
    ax.legend(loc='upper left', fontsize=11)

    plt.tight_layout()

    # Save the plot
    output_path = output_dir / 'combined_mice_plot.svg'
    plt.savefig(output_path, dpi=150)
    print(f"\nCombined plot saved to: {output_path}")

    output_path_png = output_dir / 'combined_mice_plot.png'
    plt.savefig(output_path_png, dpi=150)
    print(f"Combined plot saved to: {output_path_png}")

    plt.close()


def main():
    """Main function to generate liraglutide/semaglutide plots for DIO mice."""
    script_dir = Path(__file__).parent.resolve()
    data_dir = script_dir / "../../data/obesity"
    output_dir = script_dir / "../../outputs/obesity_eda"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data from: {data_dir}")
    df = load_data(data_dir / "obesity_a2h_dataset.csv")
    print(f"Total rows: {len(df)}")

    # Generate plot for liraglutide
    print("\nGenerating liraglutide plot...")
    create_plot(df, 'liraglutide', output_dir)

    print("\nGenerating semaglutide plot...")
    create_plot(df, 'semaglutide', output_dir)

    print("\nGenerating combined plot...")
    create_combined_plot(df, output_dir)

    print("\n" + "=" * 50)
    print("All figures have been saved to:", output_dir)
    print("=" * 50)


if __name__ == '__main__':
    main()
