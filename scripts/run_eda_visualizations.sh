#!/usr/bin/env bash
# run_eda_visualizations.sh
# Regenerates all obesity exploratory-data-analysis (EDA) figures for the
# animal-to-human (A2H) case study — the panels of manuscript Figure 3.
#
# Usage:
#   bash scripts/run_eda_visualizations.sh      # run from anywhere
#
# Data source (v2): data/obesity/obesity_a2h_dataset.csv
#
# Steps / outputs (written to outputs/obesity_eda/):
#   1. plot_species_coverage.py              → a2h_obesity_species_coverage.svg          (Fig 3c)
#   2. plot_translation_gap_dio_mice_rats.py → a2h_obesity_translation_gap_species.svg   (Fig 3a)
#   3. plot_dio_mice_treatment_duration.py   → combined_mice_plot.svg + per-drug plots   (Fig 3b)
#   4. plot_a2h_distributions.py             → a2h_preclinical_clinical_delta_dist.svg,
#                                              drug/species distribution plots           (Fig 3d,e)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EDA_DIR="$REPO_ROOT/src/obesity_eda"

# The translation-gap script resolves its default data path relative to the
# working directory, so run every script from the EDA source directory.
cd "$EDA_DIR"

echo "================================================================"
echo "  Obesity EDA Visualizations (manuscript Figure 3)"
echo "  Repo root:   $REPO_ROOT"
echo "  Output dir:  $REPO_ROOT/outputs/obesity_eda"
echo "================================================================"
echo ""

echo "── Step 1: plot_species_coverage.py  (Fig 3c) ──"
python3 plot_species_coverage.py
echo ""

echo "── Step 2: plot_translation_gap_dio_mice_rats.py  (Fig 3a) ──"
python3 plot_translation_gap_dio_mice_rats.py
echo ""

echo "── Step 3: plot_dio_mice_treatment_duration.py  (Fig 3b) ──"
python3 plot_dio_mice_treatment_duration.py
echo ""

echo "── Step 4: plot_a2h_distributions.py  (Fig 3d,e) ──"
python3 plot_a2h_distributions.py
echo ""

echo "================================================================"
echo "  All EDA figures regenerated:"
ls -1 "$REPO_ROOT/outputs/obesity_eda"
echo "================================================================"
