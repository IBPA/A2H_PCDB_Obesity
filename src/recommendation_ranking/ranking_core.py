"""
ranking_core.py — the pipeline's shared vocabulary, re-exported from one place.
==============================================================================

The implementation lives in the `core/` package, split by concern; this stays the
single import every script reaches for. Import from here unless you are editing
the internals.

  core.config     : ABS_TGT, GROUP, NCT, EVAL_GROUP, RANDOM_STATE, RAW_PATH, WEIGHT
  core.features   : build_base, build_eval_cells, feat_cols_of, ohe_dummy_cols, _ohe
  core.evaluation : normalized_ranking_score, aggregate, per_drug_summary
  core.models     : _make_model, _DEFAULT_PARAMS, _suggest_params, N_JOBS
  core.selection  : select_family_inner, lono_tune_mae
  core.labels     : make_design_label
"""

from core.config import (
    ABS_TGT, GROUP, NCT, EVAL_GROUP, RANDOM_STATE,
    SCRIPT_DIR, REPO_ROOT, RAW_PATH, WEIGHT, CAT_COLS,
)
from core.features import build_base, build_eval_cells, feat_cols_of, ohe_dummy_cols, _ohe
from core.evaluation import normalized_ranking_score, aggregate, per_drug_summary
from core.models import N_JOBS, _make_model, _DEFAULT_PARAMS, _suggest_params
from core.selection import select_family_inner, lono_tune_mae
from core.labels import make_design_label

__all__ = [
    "ABS_TGT", "GROUP", "NCT", "EVAL_GROUP", "RANDOM_STATE",
    "SCRIPT_DIR", "REPO_ROOT", "RAW_PATH", "WEIGHT", "CAT_COLS",
    "build_base", "build_eval_cells", "feat_cols_of", "ohe_dummy_cols", "_ohe",
    "normalized_ranking_score", "aggregate", "per_drug_summary",
    "N_JOBS", "_make_model", "_DEFAULT_PARAMS", "_suggest_params",
    "select_family_inner", "lono_tune_mae",
    "make_design_label",
]
