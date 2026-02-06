#!/bin/bash

RANDOM_STATE=42
TRAINING_SET_DRUGS="liraglutide"

DATA_FILE_PATH=data/obesity/obesity_a2h.csv
PATH_DATA_DIR="outputs/ml_training_${TRAINING_SET_DRUGS}/data_processing/splits"
PATH_OUTPUTS_DIR="outputs/ml_training_${TRAINING_SET_DRUGS}/msap"
PATH_CONFIG=src.obesity_model_performance.ML_pipeline.config_msap_rgs
COLUMN_TARGET="result_translation" 

MODELS=ada,xg,rf,mlp,gb
OD_METHODS=none
MVI_METHODS=simple
FS_METHODS=minmax,standard,none
OS_METHODS=none
SCORING=group_mae

python -m src.obesity_model_performance.ML_pipeline.data_processing.run_data_splitting \
    $DATA_FILE_PATH \
    "outputs/ml_training_${TRAINING_SET_DRUGS}/data_processing/splits" \
    --n-folds 3 \
    --random-state $RANDOM_STATE \
    --training-set-drugs $TRAINING_SET_DRUGS

python -m src.MSAP.msap.run_preprocess \
    $PATH_DATA_DIR/rgs_rs-${RANDOM_STATE}/data_train.csv \
    $PATH_OUTPUTS_DIR/rgs_rs-${RANDOM_STATE} \
    --path-config $PATH_CONFIG \
    --column-target "$COLUMN_TARGET" \
    --od-methods $OD_METHODS \
    --mvi-methods $MVI_METHODS \
    --fs-methods $FS_METHODS \
    --random-state $RANDOM_STATE \
    --data-type "regression"

# echo "Run grid search - validation split method: ${SPLIT_METHOD}"
# python -m src.MSAP.msap.run_grid_search \
#     $PATH_OUTPUTS_DIR/rgs_rs-${RANDOM_STATE} \
#     --path-config $PATH_CONFIG \
#     --column-target "$COLUMN_TARGET" \
#     --models $MODELS \
#     --od-methods $OD_METHODS \
#     --mvi-methods $MVI_METHODS \
#     --fs-methods $FS_METHODS \
#     --os-methods $OS_METHODS \
#     --path-grid-search-splits $PATH_DATA_DIR/rgs_rs-${RANDOM_STATE}/cv_splits.pkl \
#     --path-group-info $PATH_DATA_DIR/rgs_rs-${RANDOM_STATE}/data_train_with_labels.csv \
#     --grid-search-scoring $SCORING \
#     --random-state $RANDOM_STATE

# echo "Running feature selection"
# python -m src.ML_pipeline.run_feature_selection \
#     $PATH_OUTPUTS_DIR/rgs_rs-${RANDOM_STATE} \
#     $PATH_DATA_DIR/rgs_rs-${RANDOM_STATE} \
#     $PATH_DATA_DIR/rgs_rs-${RANDOM_STATE}/cv_splits.pkl \
#     --path-group-info $PATH_DATA_DIR/rgs_rs-${RANDOM_STATE}/data_train_with_labels.csv