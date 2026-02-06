import os
import pickle

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, GroupKFold
import click


def generate_train_test_splits(
    inputs,
    targets,
    test_size,
    random_state,
    verbose=True,
):
    """
    """
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    idxs_train, idxs_test = list(splitter.split(
        X=inputs,
        y=targets,
        groups=inputs.index,
    ))[0]

    assert set(targets.iloc[idxs_train].index.tolist()) \
        & set(targets.iloc[idxs_test].index.tolist()) \
        == set()

    if verbose:
        print("Training set distribution:")
        print(targets.iloc[idxs_train].value_counts())
        print()
        print("Testing set distribution:")
        print(targets.iloc[idxs_test].value_counts())

    return idxs_train, idxs_test


def generate_k_fold_splits(
    data_inputs,
    data_targets,
    n_folds,
    random_state,
    verbose=True,
    index_col=None
):
    if index_col is None:
        raise ValueError("Please specify an index column to group by.")
    inputs = data_inputs.copy()
    inputs.index = inputs[index_col]
    targets = data_targets.copy()

    splitter = GroupKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=random_state
        )

    splits = list(splitter.split(
        X=inputs,
        y=targets,
        groups=inputs.index,
    ))

    # Make sure grouped samples are unique to each split.
    for i, (idxs_train, idxs_val) in enumerate(splits):
        targets_train, targets_val = targets.iloc[idxs_train], targets.iloc[idxs_val]
        assert set(targets_train.index.tolist()) & set(targets_val.index.tolist()) \
            == set()

        if verbose:
            print(f"Validation Fold {i}:")
            print(targets_val.value_counts())
            print()

    return splits

def do_exist_drugs(df, drugs):
    for d in drugs:
        if d not in df.intervention.to_list():
            return False
    return True

def run_data_splitting(
    data_path,
    path_save_dir,
    n_folds,
    random_state,
    training_set_drugs=None
):
    if training_set_drugs is None:
        raise ValueError("Please specify at least one drug for the training set using --training-set-drug option.")

    df = pd.read_csv(data_path)
    labels = df['translation_outcome']
    inputs = df.drop(columns=['translation_outcome'])
    targets = pd.Series(labels, index=df.index, name='result_translation')
    path_save_dir \
        = f"{path_save_dir}/rgs_rs-{random_state}"
    
    training_set_drugs = list(set(training_set_drugs.split("_")))
    if not do_exist_drugs(inputs, training_set_drugs):
        raise ValueError(f"Drugs not exist: {training_set_drugs} "
                        f" in interventions: \n{set(inputs.intervention.to_list())}")
    test_set_drugs = []
    for drug in inputs.intervention.unique():
        if drug not in training_set_drugs:
            test_set_drugs.append(drug)

    print("All available drugs : ", inputs.intervention.unique())
    print(f"Training set uses drugs: {training_set_drugs}")
    print(f"Test set uses drug: {test_set_drugs}")

    inputs_train = inputs[inputs.intervention.isin(training_set_drugs)]
    targets_train = targets.iloc[inputs_train.index]
    
    inputs_test = inputs[inputs.intervention.isin(test_set_drugs)]
    targets_test = targets.iloc[inputs_test.index]

    assert len(set(inputs_test.intervention.to_list())) == len(test_set_drugs)
    assert len(set(inputs_train.intervention.to_list())) == len(training_set_drugs)
    for td in inputs_test.intervention.to_list():
        assert td not in inputs_train.intervention.to_list()
        
    os.makedirs(path_save_dir, exist_ok=True)

    pd.concat([inputs_train, targets_train], axis=1).to_csv(
        f"{path_save_dir}/data_train_with_labels.csv",
        index=False,
    )
    pd.concat([inputs_test, targets_test], axis=1).to_csv(
        f"{path_save_dir}/data_test_with_labels.csv",
        index=False,
    )
    
    const_cols = \
        [c for c in inputs_train.columns if (inputs_train[c].nunique() <= 1) and (c!='intervention')]
    print(f"Single or zero value columns : {const_cols}, drop them in both training and test sets" )
    
    inputs_train = inputs_train.drop(columns=const_cols)
    inputs_test = inputs_test.drop(columns=const_cols)
    
    
    print("Split by clinical studies: \n")
    k_fold_splits_by_clinical = generate_k_fold_splits(
        inputs_train,
        targets_train,
        n_folds=n_folds,
        random_state=random_state,
        index_col='NCT Number'
    )

    drop_columns = ['intervention', 'pmcid', 'NCT Number',
                    "preclinical_arm_id", "clinical_arm_id",
                    'preclinical_outcome', 'clinical_outcome']

    inputs_train = inputs_train.drop(columns=drop_columns)
    inputs_test = inputs_test.drop(columns=drop_columns)

    pd.concat([inputs_train, targets_train], axis=1).to_csv(
        f"{path_save_dir}/data_train.csv",
        index=False,
    )
    pd.concat([inputs_test, targets_test], axis=1).to_csv(
        f"{path_save_dir}/data_test.csv",
        index=False,
    )
    
    with open(f"{path_save_dir}/cv_splits.pkl", 'wb') as f:
            pickle.dump(k_fold_splits_by_clinical, f)
    

@click.command()
@click.argument(
    'data-path',
)
@click.argument(
    'path-save-dir',
)

@click.option(
    '--n-folds',
    default=3,
    help='Number of folds for cross-validation.',
)
@click.option(
    '--random-state',
    default=42,
    help='Random state for reproducibility.',
)
@click.option(
    '--training-set-drugs',
    default='liraglutide',
    help='Drugs to include in the training set.',
)
def main(
    data_path,
    path_save_dir,
    n_folds,
    random_state,
    training_set_drugs
):
    run_data_splitting(
        data_path,
        path_save_dir,
        n_folds,
        random_state,
        training_set_drugs
    )
    


if __name__ == '__main__':
    main()