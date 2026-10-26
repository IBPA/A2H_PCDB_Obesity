"""Human-readable labels for a preclinical design  (migrated verbatim from
bootstrap_rank_stability.py)."""

import pandas as pd

ROUTE_ABBREV = {
    "subcutaneous": "SC",
    "intraperitoneal": "IP",
    "oral": "oral",
    "unknown": "?route",
}
MODEL_ABBREV = {
    "diet-induced": "DIO",
    "genetic": "genetic",
    "genetic+diet": "genetic+DIO",
}
STRAIN_ABBREV = {
    "C57BL/6": "C57BL6",
    "Sprague-Dawley": "SD",
    "DIO-prone Sprague–Dawley": "DIO-SD",
    "Unknown": "?strain",
}


def make_design_label(row: pd.Series) -> str:
    """
    Build a human-readable label from one row of preclinical features.
    All seven biological dimensions are encoded so the label is always unique.
    Format: {species}/{strain}/{model}/{sex}/{route}/{duration}d/n={N}/age={A}d/w={W}g
    Example: mice/C57BL6/DIO/male/SC/28d/n=10/age=161d/w=45g
    """
    species = _decode_onehot(row, "preclinical_animal_species_")
    strain  = _decode_onehot(row, "preclinical_animal_strain_", STRAIN_ABBREV)
    model   = _decode_onehot(row, "preclinical_disease_model_", MODEL_ABBREV)
    sex     = _decode_onehot(row, "preclinical_animal_sex_")
    route   = _decode_onehot(row, "preclinical_administration_route_", ROUTE_ABBREV)
    dur     = int(row["preclinical_dosage_duration(days)"])
    n       = int(row["preclinical_animal_subject_size"]) if not pd.isna(row["preclinical_animal_subject_size"]) else "?"
    age_val = row["preclinical_animal_age_before_treatment(days)"]
    wt_val  = row["preclinical_animal_weight_before_treatment(grams)"]
    age     = int(age_val) if not pd.isna(age_val) else "?"
    wt      = int(wt_val)  if not pd.isna(wt_val)  else "?"
    return f"{species}/{strain}/{model}/{sex}/{route}/{dur}d/n={n}/age={age}d/w={wt}g"


def _decode_onehot(row: pd.Series, prefix: str, abbrevs: dict = None) -> str:
    """Return the category name for a one-hot encoded group, or '?' if none set."""
    cols = {c: c[len(prefix):] for c in row.index if c.startswith(prefix)}
    for col, label in cols.items():
        if row[col] == 1:
            return abbrevs.get(label, label) if abbrevs else label
    return "?"
