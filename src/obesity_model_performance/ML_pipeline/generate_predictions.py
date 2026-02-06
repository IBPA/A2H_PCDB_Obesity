import os
import pickle

import click
import pandas as pd
from src.obesity_model_performance.ML_pipeline.utils import constants, load_best_rgs_pipeline, get_simple_pipeline
from .run_feature_selection import GroupMAEScorer

TRAINING_DRUG = "liraglutide"

def get_model_predictions(
    model,
    data_train,
    data_test,
    sfs=None,
):
    X_train = data_train.drop(columns=['result_translation'])
    y_train = data_train['result_translation']
    X_test = data_test.drop(columns=['result_translation'])
    y_test = data_test['result_translation']
    
    prep_pipeline = get_simple_pipeline(X_train, constants.FEATURES_CAT)
    prep_pipeline.fit(X_train)
    X_train_transformed = prep_pipeline.transform(X_train)
    X_test_transformed = prep_pipeline.transform(X_test)
    
    if sfs is not None:
        X_train_transformed = X_train_transformed[list(sfs.k_feature_names_)]
        X_test_transformed = X_test_transformed[list(sfs.k_feature_names_)]

    model.fit(X_train_transformed, y_train)
    y_pred = model.predict(X_test_transformed)

    return y_test, y_pred

@click.command()
@click.option(
    '--training-drug',
    default=TRAINING_DRUG,
    help='Drug used for training the model. Default: liraglutide.',
)
@click.option(
    '--random-state',
    default=42,
)

def main(training_drug, random_state):
    path_data_dir \
            = f"outputs/ml_training_{training_drug}/data_processing/splits/rgs_rs-{random_state}"
    
    data_train = pd.read_csv(f"{path_data_dir}/data_train.csv")
    data_test = pd.read_csv(f"{path_data_dir}/data_test.csv")
    data_test_with_labels = pd.read_csv(f"{path_data_dir}/data_test_with_labels.csv")

    X_train = data_train.drop(columns=['result_translation'])
    y_train = data_train['result_translation']
    X_test = data_test.drop(columns=['result_translation'])
    y_test = data_test['result_translation']

    assert set(list(X_train.columns)) == set(list(X_test.columns))

    test_result_df = data_test.copy()

    path_rgs_dir = f"outputs/ml_training_{training_drug}/msap/rgs_rs-{random_state}"
    print(f"RGS dir: {path_rgs_dir}")
    model, model_name = load_best_rgs_pipeline(path_rgs_dir)
    print(f"Loaded model {model_name} : {model}")

    y_test, y_pred \
        = get_model_predictions(model, data_train, data_test)

    save_dir = f"outputs/ml_training_{training_drug}/testset_predictions"
    os.makedirs(save_dir, exist_ok=True)
    
    test_result_df["prediction"] = y_pred
    test_result_df["model_name"] = model_name
    
    test_result_df.to_csv(f"{save_dir}/test_set_predictions.tsv",
                        sep='\t', index=False)
        
    test_result_df['intervention'] = data_test_with_labels['intervention']

    test_result_df.to_csv(f"{save_dir}/test_set_predictions_with_intervention.tsv",
                        sep='\t', index=False)

if __name__ == '__main__':
    main()
