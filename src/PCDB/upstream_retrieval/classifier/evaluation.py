"""Score grid-search checkpoints on their validation folds, and the final model on the test set.

Archived code. Paths now follow the repository layout; the interactive confirmation
prompt and the doubled ".tsv.tsv" file suffix were removed.
"""
import argparse
import os
from glob import glob

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm

from .grid_search import get_param_set_str_list, SAVE_DATA_FOLDER, OUTPUT_WEIGHT_FOLDER
from .run_model import run_ft_model

SAVE_GS_RESULTS_FOLDER = "outputs/pcdb/upstream_retrieval/classifier/predictions/grid_search"
TEST_SET_DATA_PATH = os.path.join(SAVE_DATA_FOLDER, "test_data.tsv")


def parse_argument():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--action', type=str, required=True,
                        help="evaluate | run_grid_search_results | evaluate_grid_search_results | run_and_evaluate_finetune_result")
    parser.add_argument('--data_file_path', type=str, default=TEST_SET_DATA_PATH)
    parser.add_argument('--trained_model_weight_path', type=str)
    parser.add_argument('--save_file_name', type=str, default="ft_test_set")
    parser.add_argument('--save_results_folder', type=str, default="outputs/pcdb/upstream_retrieval/classifier/predictions")
    return parser.parse_args()


def evaluate(data_file_path):
    df = pd.read_csv(data_file_path, sep="\t")
    df['answer'] = df['answer'].apply(lambda x: str(x).lower().strip(". "))
    y_true = [1 if x == 'yes' else 0 for x in df["label"]]
    y_pred = [1 if x == 'yes' else 0 for x in df["answer"]]

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Precision: {precision}  Recall: {recall}  F1: {f1}  Accuracy: {accuracy}")
    return precision, recall, f1, accuracy


def run_grid_search_results():
    """Predict each validation fold with the last-epoch checkpoint of each hyperparameter set."""
    for param_set_str in get_param_set_str_list():
        epoch = int(param_set_str[-1]) - 1  # torchtune checkpoints are 0-indexed
        for fold in range(5):
            trained_weight_path = os.path.join(OUTPUT_WEIGHT_FOLDER, param_set_str, f"fold_{fold}", f"epoch_{epoch}")
            validation_data = pd.read_csv(os.path.join(SAVE_DATA_FOLDER, f"validation_data_fold_{fold}.tsv"), sep="\t")
            run_ft_model(data=validation_data,
                         trained_model_path=trained_weight_path,
                         batch_size=1,
                         save_file_name=f"{param_set_str}_fold_{fold}",
                         save_results_folder=SAVE_GS_RESULTS_FOLDER)


def evaluate_grid_search_results():
    rows = []
    for gs_result_file_path in tqdm(sorted(glob(os.path.join(SAVE_GS_RESULTS_FOLDER, "lr_*.tsv")))):
        _, lr, _, batch_size, _, epochs, _, fold = os.path.basename(gs_result_file_path)[:-4].split("_")
        precision, recall, f1, accuracy = evaluate(gs_result_file_path)
        rows.append({
            "learning_rate": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "fold": fold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        })

    save_path = os.path.join(SAVE_GS_RESULTS_FOLDER, "all_gs_results.tsv")
    pd.DataFrame.from_records(rows).to_csv(save_path, sep="\t", index=False)
    print("Saved grid-search summary to ", save_path)


def run_and_evaluate_finetune_result(fine_tune_model_weight_path, save_file_name,
                                     save_results_folder, data_file_path=TEST_SET_DATA_PATH):
    test_df = pd.read_csv(data_file_path, sep="\t")
    saved_prediction_path = run_ft_model(data=test_df,
                                         trained_model_path=fine_tune_model_weight_path,
                                         batch_size=1,
                                         save_results_folder=save_results_folder,
                                         save_file_name=save_file_name)
    evaluate(saved_prediction_path)


if __name__ == '__main__':
    args = parse_argument()
    if args.action == 'evaluate':
        evaluate(args.data_file_path)
    elif args.action == 'run_grid_search_results':
        run_grid_search_results()
    elif args.action == 'evaluate_grid_search_results':
        evaluate_grid_search_results()
    elif args.action == 'run_and_evaluate_finetune_result':
        run_and_evaluate_finetune_result(args.trained_model_weight_path,
                                         save_file_name=args.save_file_name,
                                         save_results_folder=args.save_results_folder,
                                         data_file_path=args.data_file_path)
    else:
        raise ValueError(f"Unknown action: {args.action}")
