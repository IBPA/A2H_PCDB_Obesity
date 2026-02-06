import os

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


SPECIES_LUT_BY_STRAIN_NAME = {
    "c57bl/6": "mice",
    "db/db": "mice",
    "c57bl/6ntac": "mice",
    "c57bl/6nrj": "mice",
    "golden syrian": "hamster",
    "ldlr−/−": "mice",
}


def clean_feature_names(names):
    """Clean feature names for display in plots."""
    names = [n.split("__")[-1].replace("_", " ") for n in names]
    names = [n.lower() for n in names]

    cleaned_names = []
    for name in names:
        if "preclinical animal strain" in name:
            strain_name = " ".join(name.split(" ")[3:])
            animal_name = strain_name
            if strain_name in SPECIES_LUT_BY_STRAIN_NAME:
                animal_name = strain_name + " " + SPECIES_LUT_BY_STRAIN_NAME[strain_name]
            name = "preclinical animal strain: " + animal_name
        elif "preclinical dosage frequency" in name:
            freq_name = " ".join(name.split(" ")[3:])
            if " + " in freq_name:
                val, unit = freq_name.split(" + ")
                # Convert "1 + every 1 day" to "once per day"
                if val == "1" and unit == "every 1 day":
                    name = "preclinical dosage frequency: once per day"
                else:
                    name = "preclinical dosage frequency: " + val + " times " + unit
            else:
                name = "preclinical dosage frequency: " + freq_name
        elif "clinical age groups" in name:
            groups = name.replace("clinical age groups", "").strip()
            groups = groups.replace(",", " & ")
            name = f"clinical age groups:\n({groups}: 1, others: 0)"
        elif "preclinical disease model" in name:
            model = name.replace("preclinical disease model", "").strip()
            name = "preclinical disease model: " + model
        elif "preclinical animal species" in name:
            species = name.replace("preclinical animal species", "").strip()
            name = "preclinical animal species: " + species
        elif "preclinical animal sex" in name:
            sex = name.replace("preclinical animal sex", "").strip()
            name = "preclinical animal sex: " + sex
        elif "preclinical animal weight before treatment" in name:
            name = "preclinical baseline animal\nweight (g)"
        elif "preclinical total dosage amount" in name:
            name = "preclinical total dosage\n(mg/kg)"
        elif "preclinical animal subject size" in name:
            name = "preclinical subject size"
        elif "clinical dosage amount value" in name:
            name = "clinical dosage amount\n(mg)"
        elif "clinical sample size" in name:
            name = "clinical sample size"
        elif "preclinical animal weight before experiment" in name:
            name = "preclinical animal weight\nbefore disease induction(g)"

        cleaned_names.append(name)
    return cleaned_names


def plot_feature_ranks(df, top_k=None, output_path="feature_importance.svg", show_others=0):
    """
    Plot feature rankings from multiple methods.

    Args:
        df: pandas DataFrame with index = feature names, columns = methods, values = ranks
        top_k: optional, number of top features to display (None = show all)
        output_path: path to save the plot
        show_others: whether to include "other features" in the plot (1 = yes, 0 = no)
    """
    n_features = len(df)

    # Max rank should be the total number of features (ranks go from 1 to n_features)
    # max_rank = n_features

    # If top_k is None or >= number of features, show all features
    if top_k is None or top_k >= n_features:
        top_k = n_features
        rest_features_means = []
    elif top_k < n_features:
        rest_features = df.tail(n_features - top_k).copy()
        rest_features_means = rest_features["mean_rank"].to_list()
        df = df.head(top_k)
    else:
        rest_features_means = []

    max_rank = \
        int(df[["mean_rank", "rank_|shap|", "rank_predictive_model", "rank_sequential_feature_selection"]].max().max())
    plt.figure(figsize=(5, 0.5 * (top_k + 1)))

    features = df.index.tolist()
    methods = df.columns.tolist()

    # Colors - one per method
    colors = sns.color_palette("muted", n_colors=len(methods))

    # Reverse vertical order so top feature appears at top
    y_positions = np.arange(len(features), 0, -1)

    # Define drawing order: average rank first (bottom), SHAP, SFS last (top)
    # Sizes decrease: average rank largest, SHAP medium, SFS smallest
    method_draw_order = {
        "mean_rank": {"zorder": 2, "size": 90, "marker": "D", "alpha": 1.0, "edgecolor": "none"},
        "rank_|shap|": {"zorder": 3, "size": 55, "marker": "o", "alpha": 0.9, "edgecolor": "none"},
        "rank_predictive_model": {"zorder": 3, "size": 40, "marker": "o", "alpha": 0.9, "edgecolor": "none"},
        "rank_sequential_feature_selection": {"zorder": 4, "size": 30, "marker": "o", "alpha": 0.9, "edgecolor": "none"},
    }

    # Plot lines connecting methods for each feature
    for i, feat in enumerate(features):
        xs = df.loc[feat].values
        plt.plot(
            xs, [y_positions[i]] * len(xs),
            color="gray", alpha=0.4, linewidth=1.0, zorder=1,
            linestyle="-."
        )

    # Plot points in specified order (average rank first/bottom, SFS last/top)
    for method_name in ["mean_rank", "rank_|shap|", "rank_predictive_model", "rank_sequential_feature_selection"]:
        if method_name not in methods:
            continue
        method_idx = methods.index(method_name)
        style = method_draw_order[method_name]
        c = colors[method_idx]

        for i, feat in enumerate(features):
            x = df.loc[feat, method_name]
            plt.scatter(
                x, y_positions[i], s=style["size"], c=[c], marker=style["marker"],
                alpha=style["alpha"], edgecolor=style["edgecolor"], linewidth=0.5, zorder=style["zorder"]
            )

    # Plot remaining features
    if show_others and rest_features_means:
        y_pos = [0] * len(rest_features_means)
        plt.scatter(
            rest_features_means, y_pos, s=100, c=[colors[-1]], marker="D",
            alpha=1.0, edgecolor="white", linewidth=0.5, zorder=3
        )

    method_name_maps = {
        "rank_sequential_feature_selection": "SFS",
        "rank_|shap|": "SHAP",
        "mean_rank": "Average rank"
    }

    # Legend (one color per method, with corresponding sizes)
    legend_order = ["mean_rank", "rank_|shap|", "rank_sequential_feature_selection"]
    for m in legend_order:
        if m not in methods:
            continue
        c = colors[methods.index(m)]
        label = method_name_maps.get(m, m)
        style = method_draw_order[m]
        plt.scatter(
            [], [], c=[c], label=label, s=style["size"] * 0.7,
            edgecolor=style["edgecolor"], linewidth=0.5, marker=style["marker"], alpha=style["alpha"]
        )
    plt.legend(frameon=True, title="Methods", loc="upper right")

    # Clean feature names for display
    feature_names = clean_feature_names(features)
    feature_names_wrapped = []
    for n in feature_names:
        words = n.split(" ")
        if len(words) > 4:
            feature_names_wrapped.append(" ".join(words[:4]) + "\n" + " ".join(words[4:]))
        else:
            feature_names_wrapped.append(n)

    # Add "other features" if we have rest features
    if show_others and rest_features_means:
        y_positions = np.append(y_positions, 0)
        feature_names_wrapped.append("other features")

    plt.yticks(y_positions, feature_names)
    # Starts at 1, goes up to max_rank, skipping every second number
    plt.xticks(range(1, max_rank + 1, 2))
    # plt.xticks(range(1, max_rank + 1))
    plt.xlim(0.5, max_rank + 0.5)
    plt.xlabel("Feature importance rank (1 = most important)")
    plt.grid(axis="x", linestyle="--", alpha=0.4)
    plt.title("Feature Ranking")

    plt.savefig(output_path, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


@click.command()
@click.argument(
    'path-rgs-dir',
    type=click.Path(exists=True),
)
@click.option(
    '--output-dir',
    type=str,
    default=None,
    help='Directory to save the plot. Defaults to parent visualization folder.'
)
@click.option(
    '--top-k',
    type=int,
    default=7,
    help='Number of top features to display. Default: show all features.'
)
@click.option(
    '--show-others',
    type=int,
    default=0,
    help='Whether to include "other features" in the plot (1 = yes, 0 = no). Default: no.'
)
def main(path_rgs_dir, output_dir, top_k, show_others):
    print(f"RGS dir: {path_rgs_dir}")

    # Set output directory (default: go up to ml_training_{drug}/visualization)
    if output_dir is None:
        # path_rgs_dir is like outputs/ml_training_liraglutide/msap/rgs_rs-42
        # We want outputs/ml_training_liraglutide/visualizations
        parent_dir = os.path.dirname(os.path.dirname(path_rgs_dir))
        output_dir = os.path.join(parent_dir, "visualizations")
    os.makedirs(output_dir, exist_ok=True)

    # Load feature importance
    feature_importance_path = f"{path_rgs_dir}/regressors/feature_importance_rgs.csv"
    print(f"Loading feature importance from: {feature_importance_path}")

    f_importance = pd.read_csv(feature_importance_path)
    f_importance = f_importance.set_index("feature_name")

    # Calculate mean_rank from rank columns only (SHAP and SFS, not model importance)
    rank_columns = [
        'rank_sequential_feature_selection',
        'rank_predictive_model',
        'rank_|shap|'
    ]
    rank_columns_present = [c for c in rank_columns if c in f_importance.columns]
    f_importance["mean_rank"] = f_importance[rank_columns_present].mean(axis=1)
    f_importance = f_importance.sort_values(by="mean_rank")

    # Keep only the columns needed for plotting (exclude model importance)
    plot_columns = rank_columns_present + ["mean_rank"]
    f_importance = f_importance[plot_columns]

    output_path = f"{output_dir}/feature_importance.svg"
    plot_feature_ranks(f_importance, top_k=top_k, output_path=output_path, show_others=show_others)


if __name__ == '__main__':
    main()
