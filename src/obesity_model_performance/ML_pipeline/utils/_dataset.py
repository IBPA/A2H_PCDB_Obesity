import pandas as pd
import numpy as np
import pickle

def load_splits(path_data_dir):
    """
    """
    data_train = pd.read_csv(f"{path_data_dir}/data_train.csv")
    data_test = pd.read_csv(f"{path_data_dir}/data_test.csv")

    with open(f"{path_data_dir}/cv_splits.pkl", 'rb') as f:
        cv_splits = pickle.load(f)

    return data_train, data_test, cv_splits

def load_regression_data(data_path,
                         subset='all'):
    
    print(f'Load regression data from {data_path} ... ')
    df = pd.read_csv(data_path, sep='\t')
    df = df.set_index("group_id")
    labels = df['result']
    labels = np.array(labels)
    
    if subset == 'all':
        pass
    else:
        raise NotImplementedError()
    
    inputs = df.drop(columns=['result'])
    labels = pd.Series(labels, index=df.index, name='result_translation')
    return inputs, labels