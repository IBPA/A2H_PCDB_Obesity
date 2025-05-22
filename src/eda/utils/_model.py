import pickle

import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler, MinMaxScaler, FunctionTransformer
)

set_config(transform_output='pandas')


def load_best_pipeline(path_cls_dir):
    gs_result = pd.read_csv(f"{path_cls_dir}/classifiers/grid_search_results.csv")
    best_model_config = gs_result.iloc[0]

    best_clf = best_model_config['cls_method']
    best_od = best_model_config['od_method']
    best_mvi = best_model_config['mvi_method']
    best_fs = best_model_config['fs_method']
    best_os = best_model_config['os_method']

    with open(
        f"{path_cls_dir}/classifiers/cls_{best_clf}_{best_od}_{best_mvi}_{best_fs}"
        f"_{best_os}.pkl",
        'rb'
    ) as f:
        best_model = pickle.load(f)

    return best_model


def _get_simple_pipeline(
        data,
        categorical_features=[]):
    """Get the pipeline for mean/mode imputation.

    Args:
        data (pd.DataFrame): Data to be imputed.
        categorical_features (list): List of categorical features.

    Returns:
        (Pipeline): Pipeline for mean/mode imputation.

    """
    return Pipeline([
        ('simple', ColumnTransformer(
            transformers=[
                (
                    f"mode_{i}",
                    SimpleImputer(strategy='most_frequent'),
                    [col],
                ) if col in categorical_features else
                (
                    f"mean_{i}",
                    SimpleImputer(strategy='mean'),
                    [col],
                ) for i, col in enumerate(data.columns)
            ],
            verbose_feature_names_out=False,
        )),
    ])


def get_mvi_pipeline(method_name, params={}):
    """Get the pipeline for missing value imputation.

    Args:
        method_name (str): Name of the missing value imputation method.
        params (dict): Parameters for the missing value imputation method.

    Returns:
        (Pipeline): Pipeline for missing value imputation.

    Raises:
        ValueError: If the method name is not supported.

    """
    if method_name == 'simple':
        return _get_simple_pipeline(**params)
    else:
        raise ValueError(f'Invalid method name: {method_name}')


def _get_fs(method_name):
    """Get the scaling pipeline.

    Args:
        method_name (str): Name of the scaling method.

    Returns:
        TODO

    Raises:
        ValueError: If the method name is not recognized.

    """
    if method_name == 'standard':
        return StandardScaler()
    elif method_name == 'minmax':
        return MinMaxScaler()
    elif method_name == 'none':
        return FunctionTransformer(feature_names_out='one-to-one')
    else:
        raise ValueError(f"Unknown scaling method: {method_name}")


def _get_ohe(categories):
    """Get the pipeline for one-hot encoding.

    Args:
        categories (list): List of categories.

    Returns:
        TODO

    """
    return OneHotEncoder(
        categories=categories,
        drop='if_binary',
        handle_unknown='ignore',
        sparse_output=False,
    )


def get_ft_pipeline(
        data,
        scaling_method_name,
        categorical_features=[],
        ):
    """Get the pipeline for feature transformation. The feature transformation
    includes scaling for numerical and one-hot encoding for categorical
    features.

    Args:
        data (pd.DataFrame): Data to be transformed.
        scaling_method (str): Name of the scaling method.
        categorical_features (list): List of categorical features.

    Returns:
        (Pipeline): Pipeline for feature transformation.

    """
    return ColumnTransformer(
        transformers=[
            (f'ohe_{i}', _get_ohe([np.sort(data[col].unique())]), [col])
            if col in categorical_features else
            (f'scaler_{i}', _get_fs(scaling_method_name), [col])
            for i, col in enumerate(data.columns)
        ],
    )


def get_simple_pipeline(inputs, categorical_features, scaling_method_name='none'):
    """Do simple feature transformation.

    """
    pipeline = Pipeline(
        [
            ('imputer', get_mvi_pipeline('simple', {
                'data': inputs,
                'categorical_features': categorical_features,
            })),
            ('ft', get_ft_pipeline(inputs, scaling_method_name, categorical_features)),
        ])

    return pipeline