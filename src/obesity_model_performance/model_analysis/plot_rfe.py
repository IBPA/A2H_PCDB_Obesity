import os
import pickle

import click
import pandas as pd
from kneed import KneeLocator
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import seaborn as sns
import matplotlib.pyplot as plt

# Required for unpickling sfs.pkl which contains GroupMAEScorer
from src.obesity_model_performance.ML_pipeline.run_feature_selection import GroupMAEScorer


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
    '--random-state',
    type=int,
    default=42,
)
def main(path_rgs_dir, output_dir, random_state):
    sfs_path = f"{path_rgs_dir}/regressors/sfs.pkl"
    print(f"Loading SFS results from: {sfs_path}")

    with open(sfs_path, 'rb') as f:
        sfs = pickle.load(f)

    k_par, score_par = len(sfs.k_feature_idx_), sfs.k_score_
    print(f"Parsimonious features: {sfs.k_feature_names_}")

    sfs_result = pd.DataFrame(sfs.get_metric_dict()).T
    sfs_result_best = sfs_result.sort_values('avg_score', ascending=False).iloc[0]
    k_best, score_best = len(sfs_result_best['feature_idx']), sfs_result_best['avg_score']

    kneedle = KneeLocator(
        range(1, len(sfs_result) + 1),
        sfs_result['avg_score'],
        S=0.1,
        curve='concave',
        direction='increasing',
    )
    k_elbow, score_elbow = kneedle.elbow, kneedle.elbow_y

    print(f"Parsimonious: k={k_par}, score={score_par}")
    print(f"Best: k={k_best}, score={score_best}")
    print(f"Elbow: k={k_elbow}, score={score_elbow}")

    sns.set_theme(style="whitegrid")

    plot_sfs(
        sfs.get_metric_dict(),
        kind='std_dev',
        figsize=(6.4, 4.8),
        ylabel='Negative Mean Absolute Error (MAE)',
    )

    ax = plt.gca()

    ymin, ymax = plt.gca().get_ylim()
    for k, score, method, color in zip(
        [k_best, k_par, k_elbow],
        [score_best, score_par, score_elbow],
        ['Best', 'Parsimonious', 'Elbow'],
        ['red', 'green', 'blue'],
    ):
        if k is not None:
            plt.vlines(
                k,
                ymin=ymin,
                ymax=ymax,
                ls='--',
                color=color,
                label=f"{method} (k={k}, score={score:.4f})",
            )

    idxs_visible = [1, len(sfs.subsets_)]
    if k_par is not None:
        idxs_visible.append(k_par)
    if k_best is not None:
        idxs_visible.append(k_best)
    if k_elbow is not None:
        idxs_visible.append(k_elbow)

    plt.title('Sequential Forward Selection (w. StdDev)')
    plt.xticks(
        range(1, len(sfs.subsets_) + 1),
        [
            i if i in idxs_visible else ''
            for i in range(1, len(sfs.subsets_) + 1)
        ],
    )
    plt.legend(title="Feature Selection Criteria")

    # Set output directory (default: go up to ml_training_{drug}/visualization)
    if output_dir is None:
        parent_dir = os.path.dirname(os.path.dirname(path_rgs_dir))
        output_dir = os.path.join(parent_dir, "visualizations")
    os.makedirs(output_dir, exist_ok=True)

    output_path = f"{output_dir}/sfs_rgs_rs-{random_state}.svg"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")


if __name__ == '__main__':
    main()
