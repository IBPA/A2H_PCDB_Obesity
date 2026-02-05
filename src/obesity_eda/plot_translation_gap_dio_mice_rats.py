import os

import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def load_and_filter_data(data_path):
    """Load data and filter for DIO models in mice/rats with liraglutide/semaglutide."""
    df = pd.read_csv(data_path, dtype=str, keep_default_na=False)

    # Filter for DIO models (diet-induced obesity)
    dio_df = df[df['preclinical_disease_model'].str.startswith('diet:', na=False)].copy()

    # Filter for mice and rats
    dio_df = dio_df[dio_df['preclinical_animal_species'].isin(['mice', 'rats'])]

    # Filter for liraglutide and semaglutide
    dio_df = dio_df[dio_df['intervention'].isin(['liraglutide', 'semaglutide'])]

    # Convert translation_outcome to numeric
    dio_df['translation_outcome'] = pd.to_numeric(dio_df['translation_outcome'], errors='coerce')

    return dio_df


def calculate_stats(data):
    """Calculate mean, SEM, 95% CI, and sample size."""
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)
    ci_95 = sem * stats.t.ppf((1 + 0.95) / 2, n - 1)

    return {
        'n': n,
        'mean': mean,
        'sem': sem,
        'ci_95': ci_95,
        'ci_lower': mean - ci_95,
        'ci_upper': mean + ci_95,
    }


def perform_ttest(group1, group2):
    """Perform independent t-test between two groups."""
    t_stat, p_value = stats.ttest_ind(group1, group2)
    return p_value


def plot_translation_gap(dio_df, output_path, figsize=(7, 6)):
    """Create translation gap plot for DIO mice vs rats."""

    # Prepare data for each group
    groups = {
        ('mice', 'liraglutide'): dio_df[(dio_df['preclinical_animal_species'] == 'mice') &
                                         (dio_df['intervention'] == 'liraglutide')]['translation_outcome'].values,
        ('mice', 'semaglutide'): dio_df[(dio_df['preclinical_animal_species'] == 'mice') &
                                         (dio_df['intervention'] == 'semaglutide')]['translation_outcome'].values,
        ('rats', 'liraglutide'): dio_df[(dio_df['preclinical_animal_species'] == 'rats') &
                                         (dio_df['intervention'] == 'liraglutide')]['translation_outcome'].values,
        ('rats', 'semaglutide'): dio_df[(dio_df['preclinical_animal_species'] == 'rats') &
                                         (dio_df['intervention'] == 'semaglutide')]['translation_outcome'].values,
    }

    # Calculate statistics for each group
    group_stats = {key: calculate_stats(data) for key, data in groups.items()}

    # Print statistics
    print("\nGroup Statistics:")
    for (species, drug), stat in group_stats.items():
        print(f"  {species.capitalize()} + {drug.capitalize()}: "
              f"mean={stat['mean']:.2f}%, n={stat['n']}, 95% CI=[{stat['ci_lower']:.2f}, {stat['ci_upper']:.2f}]")

    # Set up the plot
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=figsize)

    # Colors
    colors = {
        'liraglutide': sns.color_palette("Set2")[0],
        'semaglutide': sns.color_palette("Set2")[1]
    }

    # X positions
    species_positions = {'mice': 0, 'rats': 1}
    drug_offsets = {'liraglutide': -0.2, 'semaglutide': 0.2}

    # Plot individual data points with jitter
    np.random.seed(42)
    
    for (species, drug), data in groups.items():
        x_base = species_positions[species] + drug_offsets[drug]
        jitter = np.random.uniform(-0.08, 0.08, len(data))
        x_jittered = x_base + jitter

        ax.scatter(
            x_jittered, data,
            c=colors[drug],
            alpha=0.6,
            s=20,
            edgecolors='black',
            linewidths=0.2,
            zorder=2,
        )

    # Plot means and error bars
    for (species, drug), stat in group_stats.items():
        x_pos = species_positions[species] + drug_offsets[drug]

        # Mean value label with 95% CI
        ax.annotate(
            # f'{stat["mean"]:.1f}%\n[{stat["ci_lower"]:.1f}, {stat["ci_upper"]:.1f}]',
            f'{stat["mean"]:.1f}%',
            xy=(x_pos, stat['mean']),
            xytext=(x_pos + 0.06, stat['mean']),
            fontsize=9,
            color='black',
            va='center',
            ha='left',
            weight='medium',
        )
        
        
        # Error bar (95% CI)
        ax.errorbar(
            x_pos, stat['mean'],
            yerr=stat['ci_95'],
            fmt='none',
            color=colors[drug],
            capsize=4,
            capthick=2,
            elinewidth=2,
            zorder=4,
        )

        # Mean marker
        ax.scatter(
            x_pos, stat['mean'],
            c=colors[drug],
            s=40,
            marker='D',
            edgecolors='white',
            linewidths=0.5,
            zorder=5,
            label=drug.capitalize() if species == 'mice' else None,
        )
        

    # Calculate y-axis limits based on actual data points
    all_data = np.concatenate([data for data in groups.values()])
    data_min = np.min(all_data)
    data_max = np.max(all_data)
    padding = (data_max - data_min) * 0.1  # 10% padding
    y_min = data_min - padding - 8  # Extra space for sample size labels
    y_max = data_max + padding + 3

    # Set y-axis limits
    ax.set_ylim(y_min, y_max)

    # Add sample size labels below each group
    for (species, drug), stat in group_stats.items():
        x_pos = species_positions[species] + drug_offsets[drug]
        ax.text(
            x_pos, y_min + 2,
            f'n={stat["n"]}',
            ha='center',
            va='bottom',
            fontsize=9,
            color='gray',
        )

    # Customize axes
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['DIO mice', 'DIO rats'], fontsize=12)
    ax.set_ylabel('Translation gap\n(Δ in preclinical and clinical body weight % change)', fontsize=12)

    # Add gridlines for y-axis only
    ax.yaxis.grid(True, linestyle='-', alpha=0.4)
    ax.xaxis.grid(False)

    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper right', frameon=True, fontsize=10, title='Drug')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"\nPlot saved to: {output_path}")
    plt.close()


@click.command()
@click.option(
    '--data-path',
    type=click.Path(exists=True),
    default='../../data/obesity/obesity_a2h.csv',
    help='Path to the obesity A2H dataset CSV.'
)
@click.option(
    '--output-dir',
    type=str,
    default=None,
    help='Directory to save the plot. Defaults to src/obesity_a2h_dataset/visualization.'
)
@click.option(
    '--width',
    type=float,
    default=7.0,
    help='Figure width in inches.'
)
@click.option(
    '--height',
    type=float,
    default=6.0,
    help='Figure height in inches.'
)
def main(data_path, output_dir, width, height):
    print(f"Loading data from: {data_path}")

    dio_df = load_and_filter_data(data_path)
    print(f"Filtered data: {len(dio_df)} rows")

    # Set output directory
    if output_dir is None:
        output_dir = '../../outputs/obesity_eda'
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, 'a2h_obesity_translation_gap_species.svg')
    plot_translation_gap(dio_df, output_path, figsize=(width, height))


if __name__ == '__main__':
    main()
