"""The two modelling choices, both made strictly inside an outer training fold.

select_family_inner picks the family; lono_tune_mae tunes its hyperparameters.
Each takes the outer held-out drug and asserts it is absent from the frame it was
handed — the nested-CV boundary is enforced at the point the choice is made, not
documented and hoped for.
"""

import warnings

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
import optuna

from .config import ABS_TGT, EVAL_GROUP, GROUP, NCT, RANDOM_STATE
from .evaluation import normalized_ranking_score
from .features import _ohe, ohe_dummy_cols
from .models import _DEFAULT_PARAMS, _make_model, _suggest_params

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")


def select_family_inner(train_cells, feat_cols, seeds=range(10), holdout_group=None):
    """
    Choose the model family using ONLY the outer training drugs.

    This is the nested counterpart of model_selection.py. That script compares
    families across all drugs at once, so every held-out drug influences the
    choice — the selection bias raised in review. Here the comparison is run
    inside a single outer training fold, so the held-out drug has no say.

    Inner scheme is leave-one-drug-out over the training drugs, mirroring the
    outer task (generalise to an unseen drug). Families are compared at default
    hyperparameters; the winner is then tuned by lono_tune_mae as before.

    holdout_group: the outer held-out drug. If given, asserts it is absent from
    train_cells — the nested boundary, enforced where the choice is made.

    Returns (family, votes) with votes mapping family -> number of seeds won.
    """
    if holdout_group is not None:
        leaked = int((train_cells[GROUP] == holdout_group).sum())
        assert leaked == 0, (
            f"NESTED-CV LEAKAGE: {leaked} row(s) from the outer held-out drug "
            f"'{holdout_group}' are present in the family-selection frame.")

    families = list(_DEFAULT_PARAMS)
    votes = {f: 0 for f in families}
    for seed in seeds:
        scored = _family_pct_ranks(train_cells, feat_cols, seed)
        votes[max(families, key=lambda f: scored[f])] += 1
    return max(votes, key=votes.get), votes


def _family_pct_ranks(cells, feat_cols, seed):
    """Drug-averaged chosen-arm percentile rank per family, one seed, inner LODO."""
    y = cells[ABS_TGT].values
    groups = cells[GROUP].values
    cell_ids = cells[EVAL_GROUP].values
    ohe_cols = ohe_dummy_cols(feat_cols)
    per_drug = {f: [] for f in _DEFAULT_PARAMS}

    for drug in sorted(np.unique(groups)):
        tr, te = groups != drug, groups == drug
        Xtr_df, Xte_df = _ohe(cells.loc[tr, feat_cols], cells.loc[te, feat_cols], ohe_cols)
        imp = SimpleImputer(strategy="median")        # fold-safe, shared across families
        Xtr, Xte = imp.fit_transform(Xtr_df.values), imp.transform(Xte_df.values)

        for family, base_params in _DEFAULT_PARAMS.items():
            sc = StandardScaler()
            model = _make_model(family, {**base_params, "random_state": seed})
            model.fit(sc.fit_transform(Xtr), y[tr])
            pred = model.predict(sc.transform(Xte))
            cells_te = cell_ids[te]
            prs = [normalized_ranking_score(y[te][cells_te == c],
                                            y[te][cells_te == c][np.argmin(pred[cells_te == c])])
                   for c in np.unique(cells_te)]
            per_drug[family].append(float(np.mean(prs)))

    return {f: float(np.mean(v)) for f, v in per_drug.items()}


def lono_tune_mae(df, feat_cols, family, n_trials=60, holdout_group=None):
    """
    Tune the winning model family using Leave-One-NCT-Out inner CV.
    Objective: minimize mean MAE on abs_gap across all NCT folds.
    No same-drug leakage: LONO holds out one trial at a time.

    holdout_group: the LODO outer held-out drug. If given, this asserts the tuning
    frame `df` contains ZERO rows from that drug. Because every Optuna trial's inner
    LONO fold is a subset of `df`, that single guard proves no held-out-drug row can
    enter ANY trial — the nested-CV boundary is enforced here, at the innermost point.
    Returns (best_params, best_value).
    """
    if holdout_group is not None:
        leaked = int((df[GROUP] == holdout_group).sum())
        assert leaked == 0, (
            f"NESTED-CV LEAKAGE: {leaked} row(s) from the LODO held-out drug "
            f"'{holdout_group}' are present in the Optuna tuning frame — the inner LONO "
            f"tuning must run strictly inside the outer 8-drug training fold.")

    ncts = sorted(df[NCT].unique())
    X = df[feat_cols].values
    y = df[ABS_TGT].values
    nct_arr = df[NCT].values

    def objective(trial):
        params = _suggest_params(trial, family)
        maes = []
        for nct_id in ncts:
            tr = nct_arr != nct_id
            te = nct_arr == nct_id
            # inner LONO is a proper partition of the training frame only
            assert not (tr & te).any() and (tr | te).all()
            sc = StandardScaler()
            m  = _make_model(family, params)
            m.fit(sc.fit_transform(X[tr]), y[tr])
            pred = m.predict(sc.transform(X[te]))
            maes.append(mean_absolute_error(y[te], pred))
        return float(np.mean(maes))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    print(f"  LONO-tuned {family}: MAE={study.best_value:.3f}  params={best}")
    return best, study.best_value
