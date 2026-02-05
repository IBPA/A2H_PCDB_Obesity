#!/usr/bin/env python3
"""
Plot translation_outcome vs preclinical_dosage_duration for liraglutide and semaglutide in DIO mice.

Points are colored by preclinical_animal_weight_before_treatment (darker = higher weight).
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
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path

# Define colors for each intervention
COLORS = {
    'liraglutide': sns.color_palette("Set2")[0],
    'semaglutide': sns.color_palette("Set2")[1]
}

# Define category order
CATEGORY_ORDER = [
    "Short\n(<4 weeks)",
    "Medium\n(4-8 weeks)",
    "Long\n(8-12 weeks)",
    "Very Long\n(>12 weeks)"
]


def load_data(data_path: Path) -> pd.DataFrame:
    """Load the obesity A2H dataset."""
    df = pd.read_csv(data_path, dtype=str, keep_default_na=False)
    return df


def categorize_duration(days: float) -> str:
    """Categorize treatment duration into groups."""
    if days < 28:
        return "Short\n(<4 weeks)"
    elif days < 56:
        return "Medium\n(4-8 weeks)"
    elif days < 84:
        return "Long\n(8-12 weeks)"
    else:
        return "Very Long\n(>12 weeks)"


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
    filtered_df['preclinical_animal_weight_before_treatment(grams)'] = pd.to_numeric(
        filtered_df['preclinical_animal_weight_before_treatment(grams)'], errors='coerce'
    )

    # Remove rows with missing values in the columns we need
    filtered_df = filtered_df.dropna(subset=[
        'preclinical_dosage_duration(days)',
        'translation_outcome',
        'preclinical_animal_weight_before_treatment(grams)'
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
    weight = filtered_df['preclinical_animal_weight_before_treatment(grams)']

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Normalize weights for colormap (dark = high, light = low)
    norm = plt.Normalize(vmin=weight.min(), vmax=weight.max())

    # Create custom colormap from light to dark version of intervention color
    base_color = np.array(COLORS[intervention])
    light_color = base_color + (1 - base_color) * 0.7  # Lighten
    dark_color = base_color * 0.3  # Darken significantly
    cmap = mcolors.LinearSegmentedColormap.from_list(
        f'{intervention}_cmap',
        [light_color, base_color, dark_color]
    )

    # Create scatter plot with solid circles, colored by weight
    scatter = ax.scatter(
        x, y,
        c=weight,
        cmap=cmap,
        norm=norm,
        s=80,
        marker='o',
        edgecolors='black',
        linewidths=0.5,
        alpha=0.8
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Animal weight before treatment (grams)', fontsize=11)

    # Set x-axis ticks and labels
    ax.set_xticks(range(len(CATEGORY_ORDER)))
    ax.set_xticklabels(CATEGORY_ORDER)
    ax.tick_params(axis='x', labelsize=12)
    ax.set_xlim(-0.5, len(CATEGORY_ORDER) - 0.5)

    # Labels and title
    ax.set_xlabel('Preclinical treatment duration', fontsize=12)
    ax.set_ylabel('Translation gap\n(Δ in preclinical and clinical body weight % change)', fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
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
    fig, ax = plt.subplots(figsize=(12, 8))

    scatters = {}
    norms = {}
    cmaps = {}

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
        filtered_df['preclinical_animal_weight_before_treatment(grams)'] = pd.to_numeric(
            filtered_df['preclinical_animal_weight_before_treatment(grams)'], errors='coerce'
        )

        # Remove rows with missing values
        filtered_df = filtered_df.dropna(subset=[
            'preclinical_dosage_duration(days)',
            'translation_outcome',
            'preclinical_animal_weight_before_treatment(grams)'
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
        weight = filtered_df['preclinical_animal_weight_before_treatment(grams)']

        # Normalize weights for colormap
        norm = plt.Normalize(vmin=weight.min(), vmax=weight.max())

        # Create custom colormap
        base_color = np.array(COLORS[intervention])
        light_color = base_color + (1 - base_color) * 0.7
        dark_color = base_color * 0.3
        cmap = mcolors.LinearSegmentedColormap.from_list(
            f'{intervention}_cmap',
            [light_color, base_color, dark_color]
        )

        # Create scatter plot
        scatter = ax.scatter(
            x, y,
            c=weight,
            cmap=cmap,
            norm=norm,
            s=70,
            marker='o',
            edgecolors='black',
            linewidths=0.5,
            alpha=0.8,
            label=intervention.capitalize()
        )

        # Store for colorbars
        scatters[intervention] = scatter
        norms[intervention] = norm
        cmaps[intervention] = cmap

    # Set x-axis ticks and labels
    ax.set_xticks(range(len(CATEGORY_ORDER)))
    ax.set_xticklabels(CATEGORY_ORDER)
    ax.set_xlim(-0.5, len(CATEGORY_ORDER) - 0.5)

    # Labels and title
    ax.set_xlabel('Preclinical Dosage Duration', fontsize=12)
    ax.set_ylabel('Translation Gap', fontsize=12)
    ax.set_title('Liraglutide vs Semaglutide in DIO-Induced Mice:\nTreatment Duration, Baseline Weight & Translation Gap', fontsize=14)

    # Grid for readability
    ax.yaxis.grid(True, alpha=0.3)

    # Add gray dashed vertical lines to separate groups
    for i in range(1, len(CATEGORY_ORDER)):
        ax.axvline(x=i - 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    # Add legend
    ax.legend(loc='upper left', fontsize=11)

    # Add a single black-white colorbar for weight reference
    # Get combined weight range
    all_weights = []
    for intervention in ['liraglutide', 'semaglutide']:
        mask = (
            (df['intervention'] == intervention) &
            (df['preclinical_animal_species'] == 'mice') &
            (df['preclinical_disease_model'].str.startswith('diet:', na=False))
        )
        filtered = df[mask].copy()
        filtered['preclinical_animal_weight_before_treatment(grams)'] = pd.to_numeric(
            filtered['preclinical_animal_weight_before_treatment(grams)'], errors='coerce'
        )
        filtered = filtered.dropna(subset=['preclinical_animal_weight_before_treatment(grams)'])
        all_weights.extend(filtered['preclinical_animal_weight_before_treatment(grams)'].values)

    # Create a ScalarMappable for black-white colorbar
    sm = plt.cm.ScalarMappable(
        cmap='Greys',
        norm=plt.Normalize(vmin=min(all_weights), vmax=max(all_weights))
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, aspect=30, shrink=0.7)
    cbar.set_label('Animal Weight Before Treatment (g)', fontsize=11)

    plt.tight_layout()

    # Save the plot
    output_path = output_dir / 'combined_mice_plot.svg'
    plt.savefig(output_path, dpi=150)
    print(f"\nCombined plot saved to: {output_path}")

    plt.close()


def main():
    """Main function to generate liraglutide/semaglutide plots for DIO mice."""
    script_dir = Path(__file__).parent.resolve()
    data_dir = script_dir / "../../data/obesity"
    output_dir = script_dir / "../../outputs/obesity_eda"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data from: {data_dir}")
    df = load_data(data_dir / "obesity_a2h.csv")
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
