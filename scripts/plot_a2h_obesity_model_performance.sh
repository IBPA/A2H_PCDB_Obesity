#!/bin/bash

RANDOM_STATE=42
N_BOOTSTRAP=10000
TRAINING_DRUG="liraglutide"
INPUT_DIR=outputs/cross_drug_analysis_${TRAINING_DRUG}
OUTPUT_DIR=$INPUT_DIR/visualizations

mkdir -p $OUTPUT_DIR

echo "Generating distance vs MAE plot for ${TRAINING_DRUG}-trained model..."
python -m src.obesity_model_performance.model_performance.plot_distance_vs_MAE \
    --nct-table $INPUT_DIR/nct_level_metrics.csv \
    --output-path $OUTPUT_DIR/${TRAINING_DRUG}_distance_vs_performance \
    --training-drug $TRAINING_DRUG \
    --inference-results $INPUT_DIR/inference_results_EC50_and_half-life.csv \
    --distance-mode "both"


PREDICTIONS=outputs/ml_training_${TRAINING_DRUG}/testset_predictions/test_set_predictions_with_intervention.tsv
TEST_DATA=outputs/ml_training_${TRAINING_DRUG}/data_processing/splits/rgs_rs-${RANDOM_STATE}/data_test_with_labels.csv
TRAIN_DATA=outputs/ml_training_${TRAINING_DRUG}/data_processing/splits/rgs_rs-${RANDOM_STATE}/data_train_with_labels.csv
TEST_DRUGS="semaglutide,tirzepatide,survodutide,medi0382,exenatide"

echo "Generating bar plot..."
python -m src.obesity_model_performance.model_performance.plot_MAE_bar_plot \
    --predictions $PREDICTIONS \
    --test-data $TEST_DATA \
    --train-data $TRAIN_DATA \
    --output-path $OUTPUT_DIR/mae_bar_plot_${TRAINING_DRUG} \
    --drugs $TEST_DRUGS \
    --n-bootstrap $N_BOOTSTRAP \
    --random-state $RANDOM_STATE

# Generate plot
echo "Generating R² plot for GLP-1 test drugs..."
python -m src.obesity_model_performance.model_performance.plot_r2_glp1_drugs \
    --predictions $PREDICTIONS \
    --test-data $TEST_DATA \
    --output-path $OUTPUT_DIR/r2_glp1_drugs \
    --left-drug semaglutide \
    --right-drugs exenatide,tirzepatide,survodutide,medi0382 \
    --left-title "Semaglutide" \
    --right-title "Other GLP-1 Drugs"
