"""Data preparation for cross-drug performance analysis."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple


def load_prediction_data(
    predictions_path: str,
    test_data_path: str,
    train_data_path: str
) -> Tuple[pd.DataFrame, float]:
    """Load prediction data and compute baseline mean.

    Args:
        predictions_path: Path to predictions TSV file.
        test_data_path: Path to test data CSV file.
        train_data_path: Path to training data CSV file.

    Returns:
        Tuple of (predictions DataFrame with intervention/NCT, baseline mean).
    """
    predictions = pd.read_csv(predictions_path, sep='\t')
    test_data = pd.read_csv(test_data_path)
    train_data = pd.read_csv(train_data_path)

    # Add intervention and NCT Number if not present
    if 'intervention' not in predictions.columns:
        predictions['intervention'] = test_data['intervention'].values
    if 'NCT Number' not in predictions.columns:
        predictions['NCT Number'] = test_data['NCT Number'].values

    baseline_mean = train_data['result_translation'].mean()

    return predictions, baseline_mean


def create_single_drug_evaluation_table(
    training_drug: str,
    predictions_path: str,
    test_data_path: str,
    train_data_path: str,
    exclude_training_drug: bool = True
) -> pd.DataFrame:
    """Create evaluation table for a single training drug model.

    Args:
        training_drug: Name of the training drug (e.g., 'liraglutide' or 'semaglutide').
        predictions_path: Path to model predictions TSV.
        test_data_path: Path to model test data CSV.
        train_data_path: Path to model training data CSV.
        exclude_training_drug: If True, exclude predictions on the training drug itself.

    Returns:
        DataFrame with columns: model, test_drug, nct_id, y_true, y_pred, y_baseline.
    """
    rows = []

    # Normalize training drug name
    training_drug_lower = training_drug.lower()
    model_name = f"{training_drug_lower[:4]}_trained"

    # Load model predictions
    pred, baseline = load_prediction_data(
        predictions_path, test_data_path, train_data_path
    )

    for _, row in pred.iterrows():
        drug = row['intervention']
        if exclude_training_drug and drug.lower() == training_drug_lower:
            continue
        rows.append({
            'model': model_name,
            'test_drug': drug,
            'nct_id': row['NCT Number'],
            'y_true': row['result_translation'],
            'y_pred': row['prediction'],
            'y_baseline': baseline
        })

    df = pd.DataFrame(rows)

    # Standardize drug names to title case for matching with PK/PD table
    drug_name_map = {
        'semaglutide': 'Semaglutide',
        'liraglutide': 'Liraglutide',
        'tirzepatide': 'Tirzepatide',
        'survodutide': 'Survodutide',
        'medi0382': 'Medi0382',
        'exenatide': 'Exenatide'
    }
    df['test_drug'] = df['test_drug'].map(lambda x: drug_name_map.get(x, x))

    return df
