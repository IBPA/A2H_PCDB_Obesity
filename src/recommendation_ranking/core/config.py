"""Column names, paths, and the fixed vocabularies the feature build depends on."""

from pathlib import Path

# ── Column constants ──────────────────────────────────────────────────────────
ABS_TGT      = "abs_gap"
GROUP        = "intervention"
NCT          = "NCT Number"
EVAL_GROUP   = "eval_group"          # (NCT × clinical arm) cell identifier
RANDOM_STATE = 42

PACKAGE_DIR = Path(__file__).resolve().parent         # src/recommendation_ranking/core
SCRIPT_DIR  = PACKAGE_DIR.parent                      # src/recommendation_ranking
REPO_ROOT   = SCRIPT_DIR.parent.parent                # repo root
RAW_PATH    = REPO_ROOT / "data" / "obesity" / "obesity_a2h_dataset.csv"

# Imputation targets no pre-specified column list: the median imputer is fit on the
# whole training-fold feature matrix and fills whatever is NaN. WEIGHT is named only
# because build_base can drop its missing rows for the complete-case sensitivity run.
WEIGHT = "preclinical_animal_weight_before_treatment(grams)"   # top SHAP feature

DROP_COLS = [
    "preclinical_arm_id", "pmcid", "clinical_outcome", "preclinical_outcome",
    "preclinical_animal_weight_before_experiment(grams)", "clinical_total_doses",
    "preclinical_total_doses", "clinical_dose_escalation",
    "preclinical_animal_age_before_experiment(days)",
]
ORAL_VARIANTS = {"gastric gavage", "gavage", "intragastric", "oral", "oral gavage"}
KM_FACTORS = {"mice": 0.081, "hamsters": 0.135, "rats": 0.162, "cynomolgus monkeys": 0.324}
HUMAN_WEIGHT_KG = 100

# The 8 categorical variables one-hot encoded by build_base. Named so the fold-safe
# encoder (_ohe) can identify the resulting dummy columns.
CAT_COLS = [
    "preclinical_administration_route", "preclinical_animal_strain",
    "preclinical_animal_species", "preclinical_animal_sex", "preclinical_disease_model",
    "clinical_age_groups", "clinical_phases", "clinical_administration_route",
]
