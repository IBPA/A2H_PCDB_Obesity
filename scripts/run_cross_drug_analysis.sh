#!/bin/bash

RANDOM_STATE=42
N_BOOTSTRAP=10000
TRAINING_DRUG="liraglutide"

PREDICTIONS=outputs/ml_training_${TRAINING_DRUG}/testset_predictions/test_set_predictions_with_intervention.tsv
TEST_DATA=outputs/ml_training_${TRAINING_DRUG}/data_processing/splits/rgs_rs-${RANDOM_STATE}/data_test_with_labels.csv
TRAIN_DATA=outputs/ml_training_${TRAINING_DRUG}/data_processing/splits/rgs_rs-${RANDOM_STATE}/data_train_with_labels.csv
PKPD_PATH=data/obesity/pkpd_profiles.csv
OUTPUT_DIR=outputs/cross_drug_analysis_${TRAINING_DRUG}

mkdir -p $OUTPUT_DIR

echo "Running cross-drug performance analysis for ${TRAINING_DRUG}..."
python -m src.obesity_model_performance.model_performance.run_analysis \
    --training-drug $TRAINING_DRUG \
    --predictions $PREDICTIONS \
    --test-data $TEST_DATA \
    --train-data $TRAIN_DATA \
    --pkpd-path $PKPD_PATH \
    --output-dir $OUTPUT_DIR \
    --n-bootstrap $N_BOOTSTRAP \
    --random-state $RANDOM_STATE
