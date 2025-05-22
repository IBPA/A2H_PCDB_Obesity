import pickle

import pandas as pd


def load_splits(path_data_dir):
    """
    """
    data_train = pd.read_csv(f"{path_data_dir}/data_train.csv")
    data_test = pd.read_csv(f"{path_data_dir}/data_test.csv")

    with open(f"{path_data_dir}/cv_splits.pkl", 'rb') as f:
        cv_splits = pickle.load(f)

    return data_train, data_test, cv_splits


def load_regression_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the dataset for regression. In this case, both result preclinical and result
    clinical remain intact as continuous variables.

    Returns:
        inputs (pd.DataFrame): The input features.
        labels (pd.Series): The labels.

    """
    data = pd.read_csv(
        "../../data/cleaned_t2dm_a2h_05_21.tsv", sep='\t'
    ).set_index('pmcid')

    inputs = data.drop(columns=['clinical_change_of_hba1c_value'])
    labels = data['clinical_change_of_hba1c_value']

    return inputs, labels


def _load_classification_data_delta(
    delta: float,
    is_sigma: bool
) -> pd.DataFrame:
    """Helper function for load_classification_data().

    """
    data = pd.read_csv(
        "../../data/cleaned_t2dm_a2h_05_21.tsv", sep='\t'
    ).set_index('pmcid')

    data['result delta'] = (data['clinical_change_of_hba1c_value'] - data['preclinical_change_of_hba1c_value'])
    if is_sigma:
        sigma = data['result delta'].std()
        delta = delta * sigma
    data['result translation'] \
        = data['result delta'].abs().apply(lambda x: 1 if x < delta else 0)
    data = data.drop(columns=['clinical_change_of_hba1c_value', 'preclinical_change_of_hba1c_value', 'result delta'])

    return data


def _load_classification_data_binned(n_bins):
    data = pd.read_csv(
        "../../data/cleaned_t2dm_a2h_05_21.tsv", sep='\t'
    ).set_index('pmcid')

    data['result preclinical binned'] = pd.cut(
        data['preclinical_change_of_hba1c_value'],
        bins=[i / n_bins for i in range(n_bins)] + [1], labels=range(n_bins)
    )
    data['result clinical binned'] = pd.cut(
        data['clinical_change_of_hba1c_value'],
        bins=[i / n_bins for i in range(n_bins)] + [1], labels=range(n_bins)
    )
    data['result translation'] \
        = (
            data['result preclinical binned'] == data['result clinical binned']
        ).astype(int)

    data = data.drop(
        columns=[
            'clinical_change_of_hba1c_value',
            'preclinical_change_of_hba1c_value',
            'result preclinical binned',
            'result clinical binned',
        ]
    )

    return data


def load_classification_data(delta=None, is_sigma=False, n_bins=None, subset='all'):
    if (delta is None and n_bins is None) or (delta is not None and n_bins is not None):
        raise ValueError("Either delta or n_bins must be specified, but not both.")

    if delta is not None:
        data = _load_classification_data_delta(delta, is_sigma)
    else:
        raise NotImplementedError()
        # data = _load_classification_data_binned(n_bins)

    if subset == 'all':
        pass
    else:
        raise NotImplementedError()
    # elif subset == 'acute':
    #     data = data.query(
    #         "`acute/sustained preclinical` == 'acute' "
    #         "& `acute/sustained clinical` == 'acute'")
    #     data = data.drop(
    #         columns=['acute/sustained preclinical', 'acute/sustained clinical']
    #     )
    # elif subset == 'sustained':
    #     data = data.query(
    #         "`acute/sustained preclinical` == 'sustained' "
    #         "& `acute/sustained clinical` == 'sustained'")
    #     data = data.drop(
    #         columns=['acute/sustained preclinical', 'acute/sustained clinical']
    #     )

    inputs = data.drop(columns=['result translation'])
    labels = data['result translation']

    return inputs, labels
