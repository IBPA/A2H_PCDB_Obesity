"""The three candidate regressors: construction, defaults, and Optuna search spaces."""

import os
import warnings

from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb

from .config import RANDOM_STATE

warnings.filterwarnings("ignore")

# One thread per fit. These models are tiny (~1.4k rows), so the fit is dominated
# by thread setup, not arithmetic: on a busy shared host a LightGBM fit measured
# 59s with every core grabbed versus 0.12s pinned to one. Export RANKING_N_JOBS to
# override when the machine is idle.
N_JOBS = int(os.environ.get("RANKING_N_JOBS", "1"))


def _make_model(family, params):
    params = {**params, "n_jobs": N_JOBS}
    if family == "XGBoost":
        return xgb.XGBRegressor(**params)
    if family == "RandomForest":
        return RandomForestRegressor(**params)
    return lgb.LGBMRegressor(**params)


_DEFAULT_PARAMS = {
    "LightGBM": {
        "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "verbose": -1, "random_state": RANDOM_STATE,
    },
    "XGBoost": {
        "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "verbosity": 0, "random_state": RANDOM_STATE,
    },
    "RandomForest": {
        "n_estimators": 300, "max_depth": 8,
        "random_state": RANDOM_STATE,
    },
}


def _suggest_params(trial, family):
    """Optuna parameter suggestions per family."""
    if family == "LightGBM":
        return {
            "n_estimators":    trial.suggest_int("n_estimators", 100, 800),
            "max_depth":       trial.suggest_int("max_depth", 3, 10),
            "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample":       trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":       trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":      trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "verbose": -1, "random_state": RANDOM_STATE,
        }
    elif family == "XGBoost":
        return {
            "n_estimators":    trial.suggest_int("n_estimators", 100, 800),
            "max_depth":       trial.suggest_int("max_depth", 3, 10),
            "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample":       trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":       trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":      trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "verbosity": 0, "random_state": RANDOM_STATE,
        }
    else:  # RandomForest
        return {
            "n_estimators":trial.suggest_int("n_estimators", 100, 600),
            "max_depth":   trial.suggest_int("max_depth", 3, 15),
            "max_features":trial.suggest_float("max_features", 0.3, 1.0),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "random_state": RANDOM_STATE,
        }
