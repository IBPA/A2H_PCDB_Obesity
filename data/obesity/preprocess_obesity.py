"""
Preprocessing pipeline for obesity_a2h_v2.csv.
Produces ml_ready_obesity_dataset.csv suitable for Leave-One-Drug-Out
cross-validation with 'intervention' as the grouping key.
"""

import re
import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────────
INPUT_FILE  = "obesity_a2h_v2.csv"
OUTPUT_FILE = "ml_ready_obesity_dataset.csv"

# ──────────────────────────────────────────────────────────────────────────────
# Step 1 – Initial Cleanup & Dropping Leakage
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Step 1: Initial Cleanup & Dropping Leakage")

df = pd.read_csv(INPUT_FILE)
print(f"  Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

drop_cols = [
    "preclinical_arm_id",
    "pmcid",
    # clinical_arm_id: kept as metadata for eval_group identity (not a model feature)
    # NCT Number: kept as metadata for clinical-study-level aggregation at eval time
    # leakage
    "clinical_outcome",
    "preclinical_outcome",
    # highly missing
    "preclinical_animal_weight_before_experiment(grams)",
    # redundant raw clinical variables — model forced to use bridge ratios
    # Note: clinical_dosage_duration(days) is dropped in Step 4 after duration_ratio is built
    "clinical_total_doses",
    # clinical_sample_size: kept — included to test whether trial size is informative
    # collinear with preclinical_dosage_duration (r=0.89) — causes SHAP dilution
    "preclinical_total_doses",
    # zero variance — every row is "no"
    "clinical_dose_escalation",
    # collinear with age_before_treatment (r=0.99) — age at treatment is the true biological baseline
    "preclinical_animal_age_before_experiment(days)",
]
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
print(f"  After drops: {df.shape[0]} rows × {df.shape[1]} columns")

# ──────────────────────────────────────────────────────────────────────────────
# Step 1b – String Normalisation (routes & strains)
# ──────────────────────────────────────────────────────────────────────────────
print("\nStep 1b: String Normalisation")

# ── Route consolidation ────────────────────────────────────────────────────────
# All enteral / oral variants → 'oral'
ORAL_VARIANTS = {
    "gastric gavage", "gavage", "intragastric", "oral", "oral gavage",
}

def normalise_route(val):
    if pd.isna(val):
        return val
    return "oral" if str(val).strip().lower() in ORAL_VARIANTS else str(val).strip().lower()

for col in ["preclinical_administration_route", "clinical_administration_route"]:
    if col in df.columns:
        df[col] = df[col].map(normalise_route)

print("  Routes: gastric gavage / gavage / intragastric / oral / oral gavage → 'oral'")
print("  Unique preclinical routes:", sorted(df["preclinical_administration_route"].dropna().unique()))
print("  Unique clinical routes   :", sorted(df["clinical_administration_route"].dropna().unique()))

# ── Disease model consolidation ───────────────────────────────────────────────
# Collapse 10 sparse micro-categories into 3 interpretable master categories
def consolidate_disease_model(val):
    if pd.isna(val):
        return val
    s = str(val).strip().lower()
    if s.startswith("genetic + diet") or s == "genetic+diet":
        return "genetic+diet"
    if s.startswith("genetic"):
        return "genetic"
    return "diet-induced"  # all diet:* variants

df["preclinical_disease_model"] = df["preclinical_disease_model"].map(consolidate_disease_model)
print("  Disease models → 3 categories:", sorted(df["preclinical_disease_model"].dropna().unique()))
print("  Value counts:\n", df["preclinical_disease_model"].value_counts().to_string())

# ── Strain consolidation ───────────────────────────────────────────────────────
def consolidate_strain(val):
    if pd.isna(val):
        return val
    s = str(val).strip()
    if re.match(r"(?i)C57BL/6", s):
        return "C57BL/6"
    if re.match(r"(?i)Wistar", s):
        return "Wistar"
    if re.match(r"(?i)Sprague", s):
        return "Sprague-Dawley"
    return s

df["preclinical_animal_strain"] = df["preclinical_animal_strain"].map(consolidate_strain)
print("  Strains: C57BL/6* → C57BL/6 | Wistar* → Wistar | Sprague–Dawley* → Sprague-Dawley")
print("  Unique strains:", sorted(df["preclinical_animal_strain"].dropna().unique()))

# ──────────────────────────────────────────────────────────────────────────────
# Step 2 – Missing Value Imputation
# ──────────────────────────────────────────────────────────────────────────────
print("\nStep 2: Missing Value Imputation")

grouped_cols = [
    "preclinical_animal_age_before_treatment(days)",
    "preclinical_animal_weight_before_treatment(grams)",
    "preclinical_animal_subject_size",
]
group_keys = ["preclinical_animal_species", "preclinical_animal_sex"]

for col in grouped_cols:
    if col not in df.columns:
        continue
    group_median = df.groupby(group_keys)[col].transform("median")
    overall_median = df[col].median()
    df[col] = df[col].fillna(group_median).fillna(overall_median)
    print(f"  Grouped-imputed: {col}")

standard_impute_cols = [
    "preclinical_total_doses",
    "preclinical_total_dosage_amount(mg/kg)",
]
for col in standard_impute_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())
        print(f"  Median-imputed : {col}")

# ──────────────────────────────────────────────────────────────────────────────
# Step 3 – Feature Engineering: Human Equivalent Dose (HED) Ratios
# ──────────────────────────────────────────────────────────────────────────────
print("\nStep 3: HED Ratio Features")

KM_FACTORS = {
    "mice":               0.081,
    "hamsters":           0.135,
    "rats":               0.162,
    "cynomolgus monkeys": 0.324,
}
HUMAN_WEIGHT_KG = 100

km = df["preclinical_animal_species"].str.lower().map(KM_FACTORS)

# single-dose HED
HED_mg_per_kg   = df["preclinical_dosage_amount_value(mg/kg)"] * km
HED_absolute_mg = HED_mg_per_kg * HUMAN_WEIGHT_KG

df["dose_translation_ratio"] = (
    df["clinical_dosage_amount_value(mg)"] / HED_absolute_mg
)
print("  Created: dose_translation_ratio")

# cumulative-dose HED
cum_HED_absolute_mg = (
    df["preclinical_total_dosage_amount(mg/kg)"] * km * HUMAN_WEIGHT_KG
)
df["cumulative_dose_translation_ratio"] = (
    df["clinical_total_dosage_amount(mg)"] / cum_HED_absolute_mg
)
print("  Created: cumulative_dose_translation_ratio")

# drop raw dosage columns
raw_dosage_cols = [
    "preclinical_dosage_amount_value(mg/kg)",
    "clinical_dosage_amount_value(mg)",
    "preclinical_total_dosage_amount(mg/kg)",
    "clinical_total_dosage_amount(mg)",
]
df.drop(columns=[c for c in raw_dosage_cols if c in df.columns], inplace=True)
print(f"  Dropped raw dosage columns. Shape: {df.shape}")

# ──────────────────────────────────────────────────────────────────────────────
# Step 4 – Feature Engineering: Frequency, Duration & Route Match
# ──────────────────────────────────────────────────────────────────────────────
print("\nStep 4: Frequency, Duration & Route Match Features")

# duration ratio (preclinical raw duration kept; clinical duration already dropped)
df["duration_ratio"] = (
    df["clinical_dosage_duration(days)"] / df["preclinical_dosage_duration(days)"]
)
df.drop(columns=["clinical_dosage_duration(days)"], inplace=True)
print("  Created: duration_ratio  |  Dropped: clinical_dosage_duration(days)")


def parse_interval(freq_str):
    """Extract interval (days) from strings like '1 + every 7 days'."""
    if pd.isna(freq_str):
        return np.nan
    match = re.search(r"every\s+(\d+(?:\.\d+)?)\s+day", str(freq_str), re.IGNORECASE)
    return float(match.group(1)) if match else np.nan


preclin_interval = df["preclinical_dosage_frequency"].map(parse_interval)
clinical_interval = df["clinical_dosage_frequency"].map(parse_interval)

preclinical_doses_per_week = 7.0 / preclin_interval
clinical_doses_per_week    = 7.0 / clinical_interval

# frequency ratio: how much more/less often is the human dosed vs the animal?
df["frequency_ratio"] = clinical_doses_per_week / preclinical_doses_per_week
print("  Created: frequency_ratio (clinical_doses_per_week / preclinical_doses_per_week)")

df.drop(
    columns=["preclinical_dosage_frequency", "clinical_dosage_frequency"],
    inplace=True,
)
print("  Dropped: raw frequency string columns")

# route match: 1 if preclinical and clinical routes are the same (post-normalisation)
df["is_route_match"] = (
    df["preclinical_administration_route"] == df["clinical_administration_route"]
).astype(int)
print("  Created: is_route_match")
print(f"  Shape: {df.shape}")

# ──────────────────────────────────────────────────────────────────────────────
# Step 5 – Categorical Encoding (One-Hot, drop_first=True)
# ──────────────────────────────────────────────────────────────────────────────
print("\nStep 5: Categorical Encoding")

ohe_cols = [
    "preclinical_administration_route",
    "preclinical_animal_strain",
    "preclinical_animal_species",
    "preclinical_animal_sex",
    "preclinical_disease_model",
    "clinical_age_groups",
    "clinical_phases",
    "clinical_administration_route",
]
ohe_cols = [c for c in ohe_cols if c in df.columns]

# preserve intervention & target before encoding
intervention_col = df["intervention"].copy()
target_col       = df["translation_outcome"].copy()

df.drop(columns=["intervention", "translation_outcome"], inplace=True)
df = pd.get_dummies(df, columns=ohe_cols, drop_first=False, dtype=int)

# restore
df["intervention"]        = intervention_col.values
df["translation_outcome"] = target_col.values

print(f"  One-hot encoded {len(ohe_cols)} categorical columns.")
print(f"  Shape after encoding: {df.shape}")

# ──────────────────────────────────────────────────────────────────────────────
# Step 6 – Final Output
# ──────────────────────────────────────────────────────────────────────────────
print("\nStep 6: Final Output")

# Impute any remaining NaNs in numeric columns with their medians
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
remaining_na = df[numeric_cols].isna().sum()
cols_with_na = remaining_na[remaining_na > 0]
if not cols_with_na.empty:
    print(f"  Filling residual NaNs in: {cols_with_na.index.tolist()}")
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Metadata columns that are intentionally non-numeric
METADATA_COLS = ["intervention", "NCT Number", "clinical_arm_id"]

# Verify no missing values remain (excluding metadata)
assert df.drop(columns=METADATA_COLS).isna().sum().sum() == 0, \
    "ERROR: Missing values remain in non-metadata columns!"

# Verify all non-metadata columns are numeric
non_numeric = (
    df.drop(columns=METADATA_COLS)
    .select_dtypes(exclude=[np.number])
    .columns.tolist()
)
assert len(non_numeric) == 0, f"ERROR: Non-numeric columns found: {non_numeric}"

df.to_csv(OUTPUT_FILE, index=False)

print(f"\nFinal shape : {df.shape}")
print(f"Saved to    : {OUTPUT_FILE}")
print("\nFinal columns:")
for col in df.columns:
    print(f"  {col}")
