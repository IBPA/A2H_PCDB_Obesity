"""PK/PD distance computation for cross-drug analysis."""

import pandas as pd
import numpy as np
from typing import Dict, Literal


# Valid distance modes
DistanceMode = ['both', 'ec50_only', 'halflife_only']


def compute_pkpd_distances(
    pkpd_path: str,
    training_drugs: Dict[str, str] = None,
) -> pd.DataFrame:
    """Compute PK/PD distances from training drugs.

    The distance is computed in feature space using z-scored log-transformed values:
    - 'both': 2D Euclidean distance using log(EC50_ratio) and log(half_life_hr)
    - 'ec50_only': 1D absolute distance using log(EC50_ratio) only
    - 'halflife_only': 1D absolute distance using log(half_life_hr) only

    Features are z-scored across all drugs, then distance is computed to each training drug.

    Args:
        pkpd_path: Path to PK/PD profiles CSV.
        training_drugs: Dict mapping model name to training drug name.
            Default: {'lira_trained': 'Liraglutide', 'sema_trained': 'Semaglutide'}
        distance_mode: Which features to use for distance calculation.
            'both' (default), 'ec50_only', or 'halflife_only'.

    Returns:
        DataFrame with columns: Drug, distance_lira_trained, distance_sema_trained,
        plus the raw and z-scored features.
    """
    if training_drugs is None:
        training_drugs = {
            'lira_trained': 'Liraglutide',
            'sema_trained': 'Semaglutide'
        }

    # Load PK/PD data
    pkpd = pd.read_csv(pkpd_path)

    # Rename columns for consistency
    pkpd = pkpd.rename(columns={
        'half_life(hrs)': 'half_life_hr',
        'EC50_ratio': 'EC50_ratio'
    })

    # Compute log features
    pkpd['log_EC50_ratio'] = np.log(pkpd['EC50_ratio'])
    pkpd['log_half_life_hr'] = np.log(pkpd['half_life_hr'])

    # Z-score features
    f1_mean = pkpd['log_EC50_ratio'].mean()
    f1_std = pkpd['log_EC50_ratio'].std()
    f2_mean = pkpd['log_half_life_hr'].mean()
    f2_std = pkpd['log_half_life_hr'].std()

    pkpd['z_log_EC50_ratio'] = (pkpd['log_EC50_ratio'] - f1_mean) / f1_std
    pkpd['z_log_half_life_hr'] = (pkpd['log_half_life_hr'] - f2_mean) / f2_std

    # Compute distances to each training drug
    for distance_mode in DistanceMode:
        for model_name, train_drug in training_drugs.items():
            train_row = pkpd[pkpd['Drug'] == train_drug].iloc[0]
            z1_train = train_row['z_log_EC50_ratio']
            z2_train = train_row['z_log_half_life_hr']

            if distance_mode == 'both':
                distances = np.sqrt(
                    (pkpd['z_log_EC50_ratio'] - z1_train) ** 2 +
                    (pkpd['z_log_half_life_hr'] - z2_train) ** 2
                )
            elif distance_mode == 'ec50_only':
                distances = np.abs(pkpd['z_log_EC50_ratio'] - z1_train)
            elif distance_mode == 'halflife_only':
                distances = np.abs(pkpd['z_log_half_life_hr'] - z2_train)
            else:
                raise ValueError(f"Invalid distance_mode: {distance_mode}. "
                            "Must be 'both', 'ec50_only', or 'halflife_only'.")

            pkpd[f'distance_{model_name}_{distance_mode}'] = distances

    return pkpd


def attach_distances_to_evaluation_table(
    eval_table: pd.DataFrame,
    pkpd_distances: pd.DataFrame
) -> pd.DataFrame:
    """Attach PK/PD distances to the evaluation table.

    Args:
        eval_table: Evaluation table with columns: model, test_drug, nct_id, y_true, y_pred.
        pkpd_distances: DataFrame with distance columns per model.

    Returns:
        Evaluation table with added 'distance' column.
    """
    # Create a lookup for distances
    distance_lookup = {}
    for distance_mode in DistanceMode:
        for _, row in pkpd_distances.iterrows():
            drug = row['Drug']
            distance_lookup[('lira_trained', drug, distance_mode)] = \
                row[f'distance_lira_trained_{distance_mode}']
            distance_lookup[('sema_trained', drug, distance_mode)] = \
                row[f'distance_sema_trained_{distance_mode}']

    # Attach distances
    def get_distance(row, distance_mode):
        key = (row['model'], row['test_drug'], distance_mode)
        return distance_lookup.get(key, np.nan)

    eval_table = eval_table.copy()
    for distance_mode in DistanceMode:
        eval_table[f'distance_{distance_mode}'] = \
            eval_table.apply(get_distance, args=(distance_mode,), axis=1)

    return eval_table
