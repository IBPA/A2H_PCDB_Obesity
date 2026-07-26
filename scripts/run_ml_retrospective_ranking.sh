#!/usr/bin/env bash
# run_ml_retrospective_ranking.sh
# Reproduce the fold-safe ML retrospective ranking pipeline (src/recommendation_ranking/),
# running the steps in the exact order documented in src/recommendation_ranking/README.md.
#
# Usage:
#   bash scripts/run_ml_retrospective_ranking.sh      # run from anywhere
#   PYTHON=python bash scripts/run_ml_retrospective_ranking.sh   # override interpreter
#
# All outputs land in outputs/.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO_ROOT/src/recommendation_ranking"
OUT="$REPO_ROOT/outputs"
PYTHON="${PYTHON:-python3}"

# cd into the package so `import ranking_core` resolves; every script writes to
# absolute output paths, so the working directory does not affect the outputs.
cd "$SRC"

echo "================================================================"
echo "  ML retrospective ranking pipeline"
echo "  Repo root: $REPO_ROOT"
echo "  Python:    $("$PYTHON" --version 2>&1)"
echo "================================================================"
echo ""

echo "── Step 1/5: model_selection.py  (family selection, majority vote / 10 seeds) ──"
"$PYTHON" model_selection.py
echo ""

echo "── Step 2/5: lodo_ranking.py  (nested CV)  —  WARNING: ~1 hour ──"
"$PYTHON" lodo_ranking.py
"$PYTHON" aggregate_performance.py
echo ""

echo "── Step 3/5: train_production_model.py  (production model, all drugs) ──"
"$PYTHON" train_production_model.py
echo ""

echo "── Step 4/5: heuristic_rules_comparison.py  (rule-based baselines) ──"
"$PYTHON" heuristic_rules_comparison.py --write
echo ""

echo "── Step 5/5: preclinical_selection.py  (per-drug selected design; needs lodo_per_fold_params.json from step 2) ──"
"$PYTHON" preclinical_selection.py
echo ""

echo "================================================================"
echo "  PART 1 complete — data files written to:"
echo "  $OUT/"
echo "================================================================"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — FIGURES / PLOTS   (TODO: fill in when the "plots" half of
#                            src/recommendation_ranking/README.md is written)
# ══════════════════════════════════════════════════════════════════════════════
# The steps above regenerate the DATA FILES only. Add the figure-generation steps
# here, in README order, once that section exists. Uncomment and complete — the plot
# scripts already present in src/recommendation_ranking/ are listed below as candidates:
#
echo "── Figure: generate_per_drug_gap_ci.py  (CI data for the by-drug plot) ──"
"$PYTHON" generate_per_drug_gap_ci.py

echo "── Figure: visualizations/plot_lodo_by_drug.py ──"
"$PYTHON" visualizations/plot_lodo_by_drug.py

echo "── Figure: visualizations/plot_shap_beeswarm.py  (needs recommender_model.pkl from Part 1 step 3) ──"
"$PYTHON" visualizations/plot_shap_beeswarm.py
#
#   # ... add the remaining figures (rank stability, imputation sensitivity, …) here ...
#
echo "================================================================"
echo "  PART 2 complete — figures written."
echo "================================================================"

# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — Additional Sensitivity Analyses
# ══════════════════════════════════════════════════════════════════════════════
#
echo "── Analyze the missingness of the model input dataset ──"
"$PYTHON" imputation_analysis/imputation_missingness.py

echo "── Evaluate the sensitivity of the model to missing value imputation methods ──"
"$PYTHON" imputation_analysis/imputation_sensitivity.py

echo "── Measure the SHAP value stability of alternative missing value imputation methods ──"
"$PYTHON" imputation_analysis/shap_imputation_stability.py

echo "── Measure the SHAP value stability of LODO models ──"
"$PYTHON" shap_fold_stability.py

echo "================================================================"
echo "  PART 3 complete — sensitivity analyses written."
echo "================================================================"




