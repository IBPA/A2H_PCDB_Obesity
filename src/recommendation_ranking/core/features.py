"""Build the modelling frame from the raw A2H dataset, and the fold-safe encoder.

build_base turns one raw row (preclinical arm × clinical arm) into features:
category normalisation, the animal→human bridge ratios, then one-hot encoding.
build_eval_cells attaches the target and the evaluation-cell id on top.
"""

import re

import numpy as np
import pandas as pd

from .config import (
    ABS_TGT, EVAL_GROUP, NCT, RAW_PATH, WEIGHT,
    CAT_COLS, DROP_COLS, HUMAN_WEIGHT_KG, KM_FACTORS, ORAL_VARIANTS,
)


def build_base(drop_weight_missing=False):
    df = pd.read_csv(RAW_PATH)
    if drop_weight_missing:
        df = df[df[WEIGHT].notna()].copy()
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    for col in ["preclinical_administration_route", "clinical_administration_route"]:
        df[col] = df[col].map(_norm_route)
    df["preclinical_disease_model"] = df["preclinical_disease_model"].map(_consolidate_disease)
    df["preclinical_animal_strain"] = df["preclinical_animal_strain"].map(_consolidate_strain)

    km = df["preclinical_animal_species"].str.lower().map(KM_FACTORS)
    df["dose_translation_ratio"] = (
        df["clinical_dosage_amount_value(mg)"]
        / (df["preclinical_dosage_amount_value(mg/kg)"] * km * HUMAN_WEIGHT_KG))
    df["cumulative_dose_translation_ratio"] = (
        df["clinical_total_dosage_amount(mg)"]
        / (df["preclinical_total_dosage_amount(mg/kg)"] * km * HUMAN_WEIGHT_KG))
    df.drop(columns=[c for c in [
        "preclinical_dosage_amount_value(mg/kg)", "clinical_dosage_amount_value(mg)",
        "preclinical_total_dosage_amount(mg/kg)", "clinical_total_dosage_amount(mg)"]
        if c in df.columns], inplace=True)

    df["duration_ratio"] = df["clinical_dosage_duration(days)"] / df["preclinical_dosage_duration(days)"]
    df.drop(columns=["clinical_dosage_duration(days)"], inplace=True)
    # frequency_ratio: clinical dosing frequency relative to preclinical, from the
    # dosing intervals (1 = matched; <1 = dosed less often in the clinic).
    pre_i = df["preclinical_dosage_frequency"].map(_parse_interval)
    clin_i = df["clinical_dosage_frequency"].map(_parse_interval)
    df["frequency_ratio"] = (7.0 / clin_i) / (7.0 / pre_i)
    df.drop(columns=["preclinical_dosage_frequency", "clinical_dosage_frequency"], inplace=True)
    df["is_route_match"] = (
        df["preclinical_administration_route"] == df["clinical_administration_route"]).astype(int)

    ohe = [c for c in CAT_COLS if c in df.columns]
    keep = {k: df[k].copy() for k in ["intervention", "translation_outcome"]}
    df.drop(columns=["intervention", "translation_outcome"], inplace=True)
    df = pd.get_dummies(df, columns=ohe, drop_first=False, dtype=int)
    for k, v in keep.items():
        df[k] = v.values

    return df


def build_eval_cells(df, rankable_only=True):
    """Attach the target (ABS_TGT) and the (NCT x clinical arm) cell id (EVAL_GROUP).
    rankable_only=True (default) keeps only cells with >=2 candidate preclinical arms —
    the ranking-evaluation set (drops single-candidate cells, i.e. the 9-drug set).
    rankable_only=False keeps EVERY arm — the complete dataset (all 11 drugs, including
    single-candidate clinical arms), as used to train the production recommender."""
    df = df.copy()
    df[ABS_TGT] = df["translation_outcome"].abs()
    df[EVAL_GROUP] = df.groupby([NCT, "clinical_arm_id"]).ngroup().astype(str)
    if not rankable_only:
        return df                                      # full data: no >=2-candidate filter
    sizes = df.groupby(EVAL_GROUP).size()
    rankable = sizes[sizes >= 2].index                 # >=2 preclinical options -> a real choice
    return df[df[EVAL_GROUP].isin(rankable)].copy()


def feat_cols_of(df):
    bridge = ["dose_translation_ratio", "cumulative_dose_translation_ratio",
              "duration_ratio", "frequency_ratio", "is_route_match"]
    clinical = [c for c in df.columns if c.startswith("clinical_") and c != "clinical_arm_id"]
    preclinical = [c for c in df.columns if c.startswith("preclinical_")]
    return preclinical + bridge + clinical


def ohe_dummy_cols(feat_cols):
    """The one-hot dummy columns among feat_cols (produced by build_base's get_dummies
    over CAT_COLS), named '<categorical>_<value>'."""
    return [c for c in feat_cols if any(c.startswith(p + "_") for p in CAT_COLS)]


def _ohe(Xtr, Xte, ohe_cols):
    """Fold-safe one-hot encoding on the fixed codebook: zero out, in the held-out
    matrix Xte, every one-hot column whose category is ABSENT from the training fold
    (all-zero in Xtr). The model therefore only ever sees categories present in its
    own training data — the same guarantee as OneHotEncoder(handle_unknown='ignore')
    fit on the training fold — while the column codebook stays fixed across folds, so
    feature-subsampling models (LightGBM/XGBoost colsample_bytree) are not perturbed by
    a changing column count. Missing categoricals are already all-zero from get_dummies.
    Xtr/Xte are DataFrames over the same feature columns; returns (Xtr, Xte) with Xte's
    held-out-only categories zeroed (Xtr unchanged — its "unseen" columns are already
    all-zero by definition)."""
    unseen = [c for c in ohe_cols if (Xtr[c] == 0).all()]
    if unseen:
        Xte = Xte.copy()
        Xte.loc[:, unseen] = 0
    return Xtr, Xte


def _norm_route(v):
    if pd.isna(v):
        return v
    return "oral" if str(v).strip().lower() in ORAL_VARIANTS else str(v).strip().lower()


def _consolidate_disease(v):
    if pd.isna(v):
        return v
    s = str(v).strip().lower()
    if s.startswith("genetic + diet") or s == "genetic+diet":
        return "genetic+diet"
    if s.startswith("genetic"):
        return "genetic"
    return "diet-induced"


def _consolidate_strain(v):
    if pd.isna(v):
        return v
    s = str(v).strip()
    for pat, out in [(r"(?i)C57BL/6", "C57BL/6"), (r"(?i)Wistar", "Wistar"),
                     (r"(?i)Sprague", "Sprague-Dawley")]:
        if re.match(pat, s):
            return out
    return s


def _parse_interval(fs):
    """Dosing-frequency string "N + every M <unit>" -> the dosing INTERVAL M, in days.
    """
    if pd.isna(fs):
        return np.nan
    m = re.search(r"every\s+(\d+(?:\.\d+)?)\s+day", str(fs), re.IGNORECASE)
    return float(m.group(1)) if m else np.nan
