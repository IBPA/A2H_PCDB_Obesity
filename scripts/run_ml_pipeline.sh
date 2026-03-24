#!/usr/bin/env bash
# run_ml_pipeline.sh
# Runs the full ML recommendation pipeline and regenerates all figures,
# including the primary output: results/ranking_evaluation/R0_lodo_by_drug.svg
#
# Usage:
#   bash app/scripts/run_ml_pipeline.sh             # run from repo root
#   bash app/scripts/run_ml_pipeline.sh --retune    # force model re-selection + tuning (slow)
#
# Pipeline steps:
#   1. ranking_evaluation.py          → lodo/lono_ranking_results.csv, R0–R4 PNGs
#   2. generate_per_drug_gap_ci.py    → per_drug_gap_ci.csv
#   3. bootstrap_uncertainty.py       → bootstrap_results_*.csv, R5 PNGs
#   4. bootstrap_rank_stability.py    → bootstrap_rank_stability_*.csv, R6 PNG
#   5. plot_lodo_by_drug.py           → R0_lodo_by_drug.svg + PNG  ← primary target
#   6. plot_pool_size_performance.py  → R7 PNG
#   7. plot_rank_stability.py         → R6 PNG (refined version)
#   8. plot_shap_beeswarm.py          → R8 PNG

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="$REPO_ROOT/src/recommendtion"
PARAMS="$REPO_ROOT/outputs/recommender_model_params.json"

cd "$REPO_ROOT"
echo "================================================================"
echo "  ML Recommendation Pipeline"
echo "  Repo root: $REPO_ROOT"
echo "================================================================"

# ── Option: --retune forces model re-selection even if params are cached ──
if [[ "${1:-}" == "--retune" ]]; then
    echo "  --retune flag set: removing cached model params"
    rm -f "$PARAMS"
fi

if [[ -f "$PARAMS" ]]; then
    echo "  Cached model params found — skipping model selection/tuning"
    echo "  (run with --retune to force re-selection)"
else
    echo "  No cached params — model selection + LONO-Optuna tuning will run (~60 trials)"
fi
echo ""

# ── Step 1: Ranking evaluation (LODO + LONO) ──────────────────────────────────
echo "── Step 1: ranking_evaluation.py ──"
python3 "$APP_DIR/ranking_evaluation.py"
echo ""

# ── Step 2: Generate per_drug_gap_ci.csv ──────────────────────────────────────
# Bootstraps eval cells within each drug to produce 95% CI on mean status_quo
# and chosen_gap. Required by plot_lodo_by_drug.py.
echo "── Step 2: generate_per_drug_gap_ci.py ──"
python3 "$APP_DIR/generate_per_drug_gap_ci.py"
echo ""

# ── Step 3: Bootstrap uncertainty (pct_rank / improvement CIs) ────────────────
echo "── Step 3: bootstrap_uncertainty.py ──"
python3 "$APP_DIR/bootstrap_uncertainty.py"
echo ""

# ── Step 4: Bootstrap rank stability ──────────────────────────────────────────
echo "── Step 4: bootstrap_rank_stability.py ──"
python3 "$APP_DIR/bootstrap_rank_stability.py"
echo ""

# ── Step 5: Primary figure — R0_lodo_by_drug.svg ──────────────────────────────
echo "── Step 5: plot_lodo_by_drug.py  (primary target) ──"
python3 "$APP_DIR/plot_lodo_by_drug.py"
echo ""

# ── Step 6: Pool-size performance plot ────────────────────────────────────────
echo "── Step 6: plot_pool_size_performance.py ──"
python3 "$APP_DIR/plot_pool_size_performance.py"
echo ""

# ── Step 7: Rank stability plot ───────────────────────────────────────────────
echo "── Step 7: plot_rank_stability.py ──"
python3 "$APP_DIR/plot_rank_stability.py"
echo ""

# ── Step 8: SHAP beeswarm plot ────────────────────────────────────────────────
echo "── Step 8: plot_shap_beeswarm.py ──"
python3 "$APP_DIR/plot_shap_beeswarm.py"
echo ""

echo "================================================================"
echo "  Pipeline complete."
echo "  Primary output: outputs/ranking_evaluation/R0_lodo_by_drug.svg"
echo "================================================================"
