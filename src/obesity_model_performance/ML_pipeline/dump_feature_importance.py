import pickle

import numpy as np
import pandas as pd
import shap
import click

from .utils import constants, load_best_rgs_pipeline, get_simple_pipeline
# Required for unpickling sfs.pkl which contains GroupMAEScorer
from .run_feature_selection import GroupMAEScorer


def show_features_with_highest_score(sfs):
    all_subsets = sfs.subsets_
    best_subset = max(all_subsets.items(), key=lambda x: x[1]['avg_score'])
    best_feature_names = best_subset[1]['feature_names']
    return list(best_feature_names)


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
    '--random-state',
    default=42,
    help='Random state for the regressor.',
)
def main(path_rgs_dir, path_data_dir, random_state):
    print(f"RGS dir: {path_rgs_dir}")
    print(f"Data dir: {path_data_dir}")

    data_train = pd.read_csv(f"{path_data_dir}/data_train.csv")
    data_test = pd.read_csv(f"{path_data_dir}/data_test.csv")

    model, model_name = load_best_rgs_pipeline(path_rgs_dir)
    print(f"Best model type: {model_name}")

    sfs_path = f"{path_rgs_dir}/regressors/sfs.pkl"
    with open(sfs_path, 'rb') as f:
        sfs = pickle.load(f)

    # save directories for outputs
    shap_values_path = f"{path_rgs_dir}/regressors/shap_values_rgs.pkl"
    feature_importance_path = f"{path_rgs_dir}/regressors/feature_importance_rgs.csv"

    inputs_train = data_train.drop('result_translation', axis=1)
    labels_train = data_train['result_translation']
    inputs_test = data_test.drop('result_translation', axis=1)

    prep_pipeline = get_simple_pipeline(inputs_train, constants.FEATURES_CAT)
    prep_pipeline.fit(inputs_train)
    inputs_train = prep_pipeline.transform(inputs_train)
    inputs_test = prep_pipeline.transform(inputs_test)

    # Get all feature names after preprocessing
    all_features = list(inputs_train.columns)

    # Initialize result DataFrame with ALL features
    result = pd.DataFrame(index=all_features)
    result.index.name = 'feature_name'

    # Get SFS rankings for ALL features (sfs.subsets_ contains all steps)
    total_features_in_sfs = len(sfs.subsets_)

    sfs_result = pd.DataFrame(sfs.get_metric_dict()).T

    sfs_ranks = {}
    for i in range(1, total_features_in_sfs + 1):
        feature_names = sfs_result.loc[i, 'feature_names']
        if i == 1:
            feature_names_prev = ()
        else:
            feature_names_prev = sfs_result.loc[i - 1, 'feature_names']
        feature_name = list(set(feature_names) - set(feature_names_prev))

        assert len(feature_name) == 1
        sfs_ranks[feature_name[0]] = i

    # Add SFS ranks to result (all features should have a rank now)
    result['rank_sequential_feature_selection'] = result.index.map(sfs_ranks)

    # Get model feature importance rankings using ALL features
    model.fit(inputs_train, labels_train)
    model_rgs = model.named_steps['rgs']

    if hasattr(model_rgs, 'feature_importances_'):
        result_model = pd.DataFrame(
            model_rgs.feature_importances_,
            index=model_rgs.feature_names_in_,
            columns=['importance']
        )
        # Use method='first' to break ties and ensure unique rankings
        result_model['rank'] = result_model['importance'].rank(ascending=False, method='first')
        result['importance_predictive_model'] = result_model['importance']
        result['rank_predictive_model'] = result_model['rank'].astype(int)
    else:
        print(f"Model {model_name} does not have feature_importances_ attribute")

    # Get SHAP rankings using ALL features
    try:
        explainer = shap.Explainer(model_rgs)
        shap_values = explainer(inputs_train)

        with open(shap_values_path, 'wb') as f:
            pickle.dump(shap_values, f)

        mean_abs_shap = np.absolute(shap_values.values).mean(axis=0)
        result_shap = pd.DataFrame(
            mean_abs_shap,
            index=model_rgs.feature_names_in_,
            columns=['mean_|shap|']
        )
        # Use method='first' to break ties and ensure unique rankings
        result_shap['rank'] = result_shap['mean_|shap|'].rank(ascending=False, method='first')
        result['mean_|shap|'] = result_shap['mean_|shap|']
        result['rank_|shap|'] = result_shap['rank'].astype(int)
        print(f"SHAP values saved to: {shap_values_path}")
    except Exception as e:
        print(f"Could not compute SHAP values: {e}")

    result.to_csv(feature_importance_path)
    print(f"Feature importance saved to: {feature_importance_path}")


if __name__ == '__main__':
    main()
