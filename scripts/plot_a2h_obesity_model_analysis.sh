#!/bin/bash

RANDOM_STATE=42
TRAINING_DRUG="liraglutide"
PATH_RGS_DIR=outputs/ml_training_${TRAINING_DRUG}/msap/rgs_rs-${RANDOM_STATE}
PATH_DATA_DIR=outputs/ml_training_${TRAINING_DRUG}/data_processing/splits/rgs_rs-${RANDOM_STATE}

echo "Plot grid search results for visualization..."
python -m src.obesity_model_performance.model_analysis.plot_grid_search_results \
    $PATH_RGS_DIR \
    $PATH_DATA_DIR \
    --training-drug $TRAINING_DRUG

echo "Generating test set predictions with intervention labels..."
python -m src.obesity_model_performance.ML_pipeline.generate_predictions \
    --random-state $RANDOM_STATE \
    --training-drug $TRAINING_DRUG

echo "Generating feature importance ranking data..."
python -m src.obesity_model_performance.ML_pipeline.dump_feature_importance \
    $PATH_RGS_DIR \
    $PATH_DATA_DIR \
    --random-state $RANDOM_STATE

echo "Plot feature importance..."
python -m src.obesity_model_performance.model_analysis.plot_feature_importance \
    $PATH_RGS_DIR

echo "Plot Recursive Feature Elimination results..."
python -m src.obesity_model_performance.model_analysis.plot_rfe \
    $PATH_RGS_DIR \
    --random-state $RANDOM_STATE

echo "Plot SHAP values..."
python -m src.obesity_model_performance.model_analysis.plot_shap \
    $PATH_RGS_DIR\
    $PATH_DATA_DIR