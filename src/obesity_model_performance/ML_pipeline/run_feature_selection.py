import os
import sys
import warnings
import pickle
import numpy as np
import pandas as pd
import click
from sklearn.metrics import make_scorer

from .utils import constants
from .utils import load_best_rgs_pipeline, get_simple_pipeline

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


class GroupMAEScorer:
    """A picklable group MAE scorer class."""

    def __init__(self, groups):
        self.groups = groups
        self.__name__ = 'group_mae_score'

    def __call__(self, y_true, y_pred):
        group_idxs = self.groups.loc[y_true.index].to_numpy()
        tmp = pd.DataFrame({'group': group_idxs,
                            'ae': np.abs(y_true - y_pred)})
        mae_per_group = tmp.groupby('group')['ae'].mean()
        return -float(mae_per_group.mean())


def make_group_mae_scorer(groups):
    """Create a group MAE scorer that is picklable."""
    scorer_func = GroupMAEScorer(groups)
    return make_scorer(scorer_func)

@click.command()
@click.argument(
    'path-rgs-dir',
    type=click.Path(exists=True),
)
@click.argument(
    'path-data-dir',
    type=click.Path(exists=True),
)
@click.argument(
    'path-grid-search-splits',
    type=click.Path(exists=True)
)
@click.option(
    '--path-group-info',
    type=str,
    required=True,
    help='Path to CSV file containing group information (NCT Number column).'
)
def main(
    path_rgs_dir,
    path_data_dir,
    path_grid_search_splits,
    path_group_info
):
    print(f"RGS dir: {path_rgs_dir}")
    print(f"Data dir: {path_data_dir}")
    print(f"Grid search splits: {path_grid_search_splits}")
    print(f"Group info: {path_group_info}")
    rgs_pipeline, _ = load_best_rgs_pipeline(path_rgs_dir)

    with open(path_grid_search_splits, 'rb') as f:
            cv_splits = pickle.load(f)
    data_train = pd.read_csv(f"{path_data_dir}/data_train.csv")

    X_train = data_train.drop(columns=['result_translation'])
    y_train = data_train['result_translation']

    # Load group info for group_mae scoring
    group_info = pd.read_csv(path_group_info)
    groups = group_info['NCT Number'].copy()
    group_mae_scorer = make_group_mae_scorer(groups)

    cat_features = constants.FEATURES_CAT

    prep_pipeline = get_simple_pipeline(
        X_train, cat_features
    )
    prep_pipeline.fit(X_train)
    X_train_transformed = prep_pipeline.transform(X_train)

    from mlxtend.feature_selection import SequentialFeatureSelector

    sfs = SequentialFeatureSelector(
        rgs_pipeline,
        k_features='parsimonious',
        forward=True,
        floating=False,
        verbose=2,
        scoring=group_mae_scorer,
        cv=cv_splits,
        n_jobs=-1,
    )
    sfs.fit(X_train_transformed, y_train)

    with open(f"{path_rgs_dir}/regressors/sfs.pkl", 'wb') as f:
        pickle.dump(sfs, f)


if __name__ == '__main__':
    main()