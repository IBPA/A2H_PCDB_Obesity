import os

import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

def calculate_species_coverage(df):
    """Calculate species coverage and DIO-induced coverage for each species."""
    total_count = len(df)
    species_counts = df['preclinical_animal_species'].value_counts()

    # Calculate total DIO coverage in entire dataset
    total_dio = df[df['preclinical_disease_model'].str.startswith('diet:', na=False)]
    total_dio_pct = (len(total_dio) / total_count) * 100

    results = []
    for species in species_counts.index:
        species_data = df[df['preclinical_animal_species'] == species]
        species_count = len(species_data)
        species_pct = (species_count / total_count) * 100

        # DIO-induced = any disease model starting with "diet:"
        dio_data = species_data[
            species_data['preclinical_disease_model'].str.startswith('diet:', na=False)
        ]
        dio_count = len(dio_data)

        if species_count > 0:
            dio_pct_within_species = (dio_count / species_count) * 100
        else:
            dio_pct_within_species = 0

        # The second bar is DIO% of species * species% of total
        dio_pct_of_total = (dio_pct_within_species / 100) * species_pct

        results.append({
            'species': species,
            'species_count': species_count,
            'species_pct': species_pct,
            'dio_count': dio_count,
            'dio_pct_within_species': dio_pct_within_species,
            'dio_pct_of_total': dio_pct_of_total,
        })

    return pd.DataFrame(results), total_dio_pct


def format_species_name(name):
    """Format species name for display."""
    name_map = {
        'mice': 'Mice',
        'rats': 'Rats',
        'hamsters': 'Hamsters',
        'pigs': 'Pigs',
        'cynomolgus monkeys': 'Cynomolgus\nmonkeys',
    }
    return name_map.get(name.lower(), name.capitalize())


def plot_species_coverage(df_results, output_path):
    """Plot horizontal bar chart with species coverage and DIO-induced coverage."""
    # Sort by species coverage descending
    df_results = df_results.sort_values('species_pct', ascending=True).reset_index(drop=True)

    species_names = df_results['species'].values
    species = [format_species_name(s) for s in species_names]
    species_pct = df_results['species_pct'].values
    species_count = df_results['species_count'].values
    dio_pct = df_results['dio_pct_of_total'].values
    dio_count = df_results['dio_count'].values

    # Species used in formal statistical analysis
    analysis_species = ['mice', 'rats']

    # Set up the plot
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.yaxis.grid(False)  # Remove horizontal grid lines
    ax.xaxis.grid(True)   # Keep vertical grid lines

    y_pos = np.arange(len(species))
    bar_height = 0.35

    # Plot species coverage bars (All arms)
    bars1 = ax.barh(
        y_pos + bar_height/2,
        species_pct,
        height=bar_height,
        label='All arms',
        color='#4C72B0',
        edgecolor='none',
    )

    # Plot DIO-induced bars (DIO models only)
    bars2 = ax.barh(
        y_pos - bar_height/2,
        dio_pct,
        height=bar_height,
        label='DIO models only',
        color='#55A868',
        edgecolor='none',
    )

    # Add value labels with sample sizes
    for i, (bar, val, n) in enumerate(zip(bars1, species_pct, species_count)):
        ax.text(
            val + 1,
            bar.get_y() + bar.get_height()/2,
            f'{val:.1f}% (n={n})',
            va='center',
            ha='left',
            fontsize=10,
        )

    for i, (bar, val, n) in enumerate(zip(bars2, dio_pct, dio_count)):
        ax.text(
            val + 1,
            bar.get_y() + bar.get_height()/2,
            f'{val:.1f}% (n={n})',
            va='center',
            ha='left',
            fontsize=10,
        )

    # Create y-axis labels with asterisk for species in formal analysis
    y_labels = []
    for s, orig_name in zip(species, species_names):
        # if orig_name.lower() in analysis_species:
        #     y_labels.append(f'{s} *')
        # else:
        #     y_labels.append(s)
        y_labels.append(s)

    # Customize axes
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=12)
    ax.set_xlabel('Percentage of dataset (%)', fontsize=12)
    ax.set_xlim(0, max(species_pct) * 1.25)

    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    ax.legend(loc='lower right', frameon=True, fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def main():
    script_dir = Path(__file__).parent.resolve()
    data_dir = script_dir / "../../data/obesity"
    output_dir = script_dir / "../../outputs/obesity_eda"
    print(f"Loading data from: {data_dir}")

    df = pd.read_csv(data_dir / "obesity_a2h_dataset.csv", dtype=str, keep_default_na=False)
    print(f"Total rows: {len(df)}")

    # Calculate coverage
    df_results, total_dio_pct = calculate_species_coverage(df)
    print(f"\nDIO-induced disease model in entire dataset: {total_dio_pct:.1f}%")
    print("\nSpecies coverage:")
    print(df_results.to_string(index=False))

    # Set output directory
    if output_dir is None:
        output_dir = '../../outputs/obesity_eda'
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, 'a2h_obesity_species_coverage.svg')
    plot_species_coverage(df_results, output_path)


if __name__ == '__main__':
    main()
