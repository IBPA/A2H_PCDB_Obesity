"""Split the 525 labeled articles and write one torchtune recipe per (fold, hyperparameter set).

Archived code. With random_state=1 this reproduces data/PCDB/upstream_retrieval/classifier/split_assignments.tsv
exactly: 132 test articles, and 5 cross-validation folds over the remaining 393.
Also writes the combined 393-article training file used for the final model
(previously produced by a notebook).
"""
import argparse
import os
import shutil
from itertools import product

import pandas as pd
from ruamel.yaml import YAML
from sklearn.model_selection import KFold, train_test_split

from .create_dataset_json import create_json_file

PARAMS = {
    "lr": [2e-5, 5e-5],
    "epochs": [3, 5],
    "batch_size": [2, 4],
}

DATA_FILE_PATH = "data/PCDB/upstream_retrieval/classifier/labeled_dataset.tsv"
SAVE_DATA_FOLDER = "outputs/pcdb/upstream_retrieval/classifier/splits"
RECIPE_FOLDER = "outputs/pcdb/upstream_retrieval/classifier/recipes"
DEFAULT_RECIPE = "src/PCDB/upstream_retrieval/classifier/recipes/qlora_base.yaml"
OUTPUT_WEIGHT_FOLDER = "outputs/pcdb/upstream_retrieval/classifier/model_weights"


def parse_argument():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--data_file_path', type=str, default=DATA_FILE_PATH)
    parser.add_argument('--output_weight_folder', type=str, default=OUTPUT_WEIGHT_FOLDER)
    return parser.parse_args()


def modify_yaml_file(save_path, updates, input_file=DEFAULT_RECIPE):
    yaml = YAML()
    yaml.preserve_quotes = True
    with open(input_file, 'r') as f:
        data = yaml.load(f)

    if 'batch_size' in updates:
        data['batch_size'] = updates['batch_size']
    if 'epochs' in updates:
        data['epochs'] = updates['epochs']
    if 'optimizer' in data and 'lr' in updates:
        data['optimizer']['lr'] = updates['lr']
    if 'output_dir' in updates:
        data['output_dir'] = updates['output_dir']
    if 'dataset' in data and 'data_files' in updates:
        data['dataset']['data_files'] = updates['data_files']

    with open(save_path, 'w') as f:
        yaml.dump(data, f)

    print(f"Updated YAML file saved to {save_path}")


def get_param_sets():
    return [dict(zip(PARAMS.keys(), combination)) for combination in product(*PARAMS.values())]


def get_param_set_str_list():
    return [f"lr_{p['lr']}_bs_{p['batch_size']}_epochs_{p['epochs']}" for p in get_param_sets()]


def create_recipes(fold_n, data_file_path, output_weight_folder):
    recipe_folder = os.path.join(RECIPE_FOLDER, f"fold_{fold_n}")
    if os.path.exists(recipe_folder):
        shutil.rmtree(recipe_folder)
    os.makedirs(recipe_folder)

    for param_set, param_set_str in zip(get_param_sets(), get_param_set_str_list()):
        output_dir = os.path.join(output_weight_folder, param_set_str, f"fold_{fold_n}")
        updates = {**param_set, "output_dir": output_dir, "data_files": data_file_path}
        modify_yaml_file(os.path.join(recipe_folder, f"fold_{fold_n}_{param_set_str}.yaml"), updates)


def save_split(df, name):
    df = df.reset_index(drop=True)
    df.to_csv(os.path.join(SAVE_DATA_FOLDER, f"{name}.tsv"), sep="\t", index=False)
    json_path = os.path.join(SAVE_DATA_FOLDER, f"{name}.json")
    create_json_file(df, json_path)
    return json_path


def split_data_for_cv(dataframe, output_weight_folder, n_splits=5, shuffle=True, random_state=1):
    os.makedirs(SAVE_DATA_FOLDER, exist_ok=True)
    train_val_data, test_data = train_test_split(dataframe, test_size=0.25, random_state=random_state)
    save_split(test_data, "test_data")
    save_split(train_val_data, "training_and_validation_data")

    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    for fold_number, (train_index, val_index) in enumerate(kf.split(train_val_data)):
        train_json_path = save_split(train_val_data.iloc[train_index], f"training_data_fold_{fold_number}")
        save_split(train_val_data.iloc[val_index], f"validation_data_fold_{fold_number}")
        create_recipes(fold_number, train_json_path, output_weight_folder)


if __name__ == '__main__':
    args = parse_argument()
    data = pd.read_csv(args.data_file_path, sep="\t")
    split_data_for_cv(data, args.output_weight_folder)
