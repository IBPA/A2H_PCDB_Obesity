import os
import pickle

import click
import pandas as pd
import shap
import numpy as np
import matplotlib.pyplot as plt

from src.obesity_model_performance.ML_pipeline.utils import constants, load_best_rgs_pipeline, get_simple_pipeline



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
    '--output-dir',
    type=str,
    default=None,
    help='Directory to save the plot. Defaults to parent visualization folder.'
)
@click.option(
    '--max-display',
    type=int,
    default=15,
    help='Maximum number of features to display in the plot.'
)
def main(path_rgs_dir, path_data_dir, output_dir, max_display):
    print(f"RGS dir: {path_rgs_dir}")
    print(f"Data dir: {path_data_dir}")

    # Set output directory (default: go up to ml_training_{drug}/visualization)
    if output_dir is None:
        parent_dir = os.path.dirname(os.path.dirname(path_rgs_dir))
        output_dir = os.path.join(parent_dir, "visualizations")
    os.makedirs(output_dir, exist_ok=True)

    save_shap_result_path = f"{output_dir}/shap_values_rgs.csv"
    save_shap_plot_path = f"{output_dir}/shap_rgs.svg"


    model, model_name = load_best_rgs_pipeline(path_rgs_dir)
    print(f"Best model type: {model_name}")

    # Load data
    data_train = pd.read_csv(f"{path_data_dir}/data_train.csv")

    inputs_train = data_train.drop('result_translation', axis=1)
    labels_train = data_train['result_translation']

    # Preprocess
    prep_pipeline = get_simple_pipeline(inputs_train, constants.FEATURES_CAT)
    prep_pipeline.fit(inputs_train)
    inputs_train = prep_pipeline.transform(inputs_train)

    # Fit model and compute SHAP values
    model.fit(inputs_train, labels_train)
    model_rgs = model.named_steps['rgs']

    explainer = shap.Explainer(model_rgs)
    shap_values = explainer(inputs_train)

    # Save SHAP values to CSV
    save_shap_df = inputs_train.copy(deep=True)
    for i, fn in enumerate(shap_values.feature_names):
        save_shap_df[f"shap_{fn}"] = shap_values.values[:, i]
    save_shap_df['shap_base_value'] = shap_values.base_values
    save_shap_df["model_prediction"] = model_rgs.predict(inputs_train)
    save_shap_df.to_csv(save_shap_result_path, index=False)
    print(f"SHAP values saved to: {save_shap_result_path}")

    # Load feature importance for ordering
    feature_importance_path = f"{path_rgs_dir}/regressors/feature_importance_rgs.csv"
    if os.path.exists(feature_importance_path):
        feature_importance = pd.read_csv(feature_importance_path)
        feature_importance = feature_importance.set_index("feature_name")
        feature_importance["mean_rank"] = feature_importance.mean(axis=1)
        feature_importance = feature_importance.sort_values(by="mean_rank")
        order_features = feature_importance.index.to_list()

        col2num = {col: i for i, col in enumerate(model_rgs.feature_names_in_)}
        order = list(map(col2num.get, order_features))
    else:
        order = None
        print("Feature importance file not found, using default order")

    # Clean up feature names for display
    shap_values.feature_names = [
        fn.split('__')[-1].replace("_", " ")
        for fn in shap_values.feature_names
    ]

    # Plot
    k = min(max_display, len(shap_values.feature_names))

    shap.plots.beeswarm(
        shap_values,
        show=False,
        order=order,
        max_display=k + 1,
        s=30
    )

    fig = plt.gcf()
    fig.set_size_inches(8, 0.6 * (k + 1))

    plt.savefig(save_shap_plot_path, bbox_inches='tight')
    print(f"SHAP plot saved to: {save_shap_plot_path}")
    plt.close()


if __name__ == '__main__':
    main()
