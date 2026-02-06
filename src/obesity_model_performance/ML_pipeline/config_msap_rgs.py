# -*- coding: utf-8 -*-
"""The configuration for using MSAP.

Attributes:
    COLUMNS_CATEGORICAL (list): List of categorical columns.
    PARAMS_OD (dict): Parameters for outlier detection methods.
    PARAMS_MVI (dict): Parameters for missing value imputation methods.

Authors:
    Fangzhou Li - fzli@ucdavis.edu

"""
from .utils import constants

# Experiment parameters.
COLUMNS_CATEGORICAL = constants.FEATURES_CAT

# Hyperparameters for the ourlier detection methods.
PARAMS_OD = {
    'none': {},
}

# Hyperparameters for the missing value imputation methods.
PARAMS_MVI = {
    'simple': {},
}

# Hyperparameters for the grid search.
PARAMS_GRID = {
    'rf': { #random forest
        'n_estimators': [100, 200, 400],
        'criterion': ['squared_error', 'absolute_error'],
        'max_depth': [None, 2, 4, 6, 8],
        'max_features': ['sqrt', 'log2', 0.3],
        "min_samples_leaf": [2, 5, 10, 20],
        'random_state': [None],
    },
    'ada': { #adaboost
        'n_estimators': [100, 200, 400],
        'learning_rate': [0.01, 0.02, 0.04, 0.08, 0.1],
        'loss': ['linear', 'square', 'exponential'],
        'random_state': [None],
    },
    'mlp': { #multi-layer perceptron
        'hidden_layer_sizes': [(100,), (100, 100), (100, 100, 100)],
        'learning_rate': ['adaptive', 'constant'],
        'learning_rate_init': [1e-5, 1e-4, 1e-3],
        'max_iter': [1000],
        'n_iter_no_change': [5, 10],
        'random_state': [None],
    },
    'gb': { #gradient boosting
        'n_estimators': [100, 200, 400],
        'learning_rate': [0.01, 0.02, 0.04, 0.08, 0.1],
        'max_depth': [2, 3, 4, 6, 8],
        'min_samples_leaf': [2, 5, 10, 20],
        'random_state': [None],
    },
    'xg': { #xgboost
        'n_estimators': [100, 200, 400],
        'learning_rate': [0.01, 0.02, 0.04, 0.08, 0.1],
        'max_depth': [2, 3, 4, 6, 8],
        'min_child_weight': [1, 3, 5, 8, 10],
        'subsample': [0.5, 0.8, 1.0],
        'reg_lambda': [0.1, 1, 2, 5],
        'random_state': [None],
    }
}