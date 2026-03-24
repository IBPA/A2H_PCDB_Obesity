"""
SHAP Dependence Plot — dose_translation_ratio (top bridge feature).

Trains LightGBM on the full dataset, computes SHAP values, and plots the
dependence of SHAP(dose_translation_ratio) on the feature value, coloured
by the SHAP-detected strongest interaction variable.

Output: outputs/ranking_evaluation/R9_shap_dependence.png / .svg
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR  = Path(__file__).parent
REPO_ROOT   = SCRIPT_DIR.parent.parent
DATA_PATH   = REPO_ROOT / "data" / "ml_ready_obesity_dataset.csv"
PARAMS_PATH = REPO_ROOT / "outputs" / "recommender_model_params.json"
RESULTS_DIR = REPO_ROOT / "outputs" / "ranking_evaluation"
OUT_FIG     = RESULTS_DIR / "R9_shap_dependence.png"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET  = "translation_outcome"
ABS_TGT = "abs_gap"
FOCUS   = "dose_translation_ratio"

FEATURE_LABELS = {
    "dose_translation_ratio":             "Single-dose ratio\n(preclinical/clinical)",
    "cumulative_dose_translation_ratio":  "Cumulative dose ratio\n(preclinical/clinical)",
    "duration_ratio":                     "Duration ratio\n(preclinical/clinical)",
    "frequency_ratio":                    "Dosing frequency ratio\n(preclinical/clinical)",
    "is_route_match":                     "Route match\n(pre=clin: 1, Others: 0)",
    "preclinical_animal_weight_before_treatment(grams)": "Preclinical: animal\nbaseline weight (g)",
    "preclinical_dosage_duration(days)":  "Preclinical: animal\ntreatment duration (days)",
    "preclinical_animal_age_before_treatment(days)":     "Preclinical: animal\nage at start (days)",
    "preclinical_animal_subject_size":    "Preclinical: animal\ngroup size (n)",
    "clinical_sample_size":               "Clinical: trial\nsample size (n)",
}

def clean_name(col):
    return FEATURE_LABELS.get(col, col.replace("preclinical_", "Pre: ")
                                       .replace("clinical_", "Clin: ")
                                       .replace("_", " "))

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df[ABS_TGT] = df[TARGET].abs()

bridge_cols      = ["dose_translation_ratio", "cumulative_dose_translation_ratio",
                    "duration_ratio", "frequency_ratio", "is_route_match"]
clinical_cols    = [c for c in df.columns if c.startswith("clinical_")
                    and c != "clinical_arm_id"]
preclinical_cols = [c for c in df.columns if c.startswith("preclinical_")]
feat_cols        = preclinical_cols + bridge_cols + clinical_cols

X_raw = df[feat_cols].values
y     = df[ABS_TGT].values

focus_idx = feat_cols.index(FOCUS)

# ── Train model ────────────────────────────────────────────────────────────────
with open(PARAMS_PATH) as f:
    params = json.load(f)["params"]

sc = StandardScaler()
X_scaled = sc.fit_transform(X_raw)

print(f"Training LightGBM on all {len(df)} rows...")
model = lgb.LGBMRegressor(**params, verbose=-1)
model.fit(X_scaled, y)

# ── SHAP values ────────────────────────────────────────────────────────────────
print("Computing SHAP values...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_scaled)

mean_abs = np.abs(shap_values).mean(axis=0)
top_idx  = np.argsort(mean_abs)[::-1]
print(f"  {FOCUS} rank: {list(top_idx).index(focus_idx)+1} of {len(feat_cols)}")

# ── Derive species label per row ───────────────────────────────────────────────
focus_shap = shap_values[:, focus_idx]   # SHAP values for our feature
focus_vals = X_raw[:, focus_idx]         # original (unscaled) feature values

SPECIES_COLS = {
    "preclinical_animal_species_mice":               "Mice",
    "preclinical_animal_species_rats":               "Rats",
    "preclinical_animal_species_hamsters":           "Hamsters",
    "preclinical_animal_species_cynomolgus monkeys": "Cynomolgus monkeys",
}
SPECIES_COLORS = {
    "Mice":                "#1f77b4",
    "Rats":                "#ff7f0e",
    "Hamsters":            "#2ca02c",
    "Cynomolgus monkeys":  "#9467bd",
}

species_label = pd.Series(["Unknown"] * len(df))
for col, label in SPECIES_COLS.items():
    if col in df.columns:
        species_label[df[col].astype(float) == 1] = label

# ── LOWESS smoothing ───────────────────────────────────────────────────────────
from statsmodels.nonparametric.smoothers_lowess import lowess
order     = np.argsort(focus_vals)
smooth_xy = lowess(focus_shap[order], focus_vals[order], frac=0.35, it=3)

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5))

# Clip extreme x for readability
X_LO, X_HI = np.percentile(focus_vals, 1), np.percentile(focus_vals, 99)
mask_plot   = (focus_vals >= X_LO) & (focus_vals <= X_HI)

# Scatter by species
for sp, color in SPECIES_COLORS.items():
    sp_mask = (species_label == sp).values & mask_plot
    if sp_mask.sum() == 0:
        continue
    ax.scatter(
        focus_vals[sp_mask], focus_shap[sp_mask],
        c=color, s=28, alpha=0.6, linewidths=0, zorder=2, label=sp,
    )

# Smoothing line (all species combined)
smooth_mask = (smooth_xy[:, 0] >= X_LO) & (smooth_xy[:, 0] <= X_HI)
ax.plot(smooth_xy[smooth_mask, 0], smooth_xy[smooth_mask, 1],
        color="black", lw=2.2, zorder=4, label="LOWESS trend (all species)")

# Reference line at ratio = 1.0
ax.axvline(1.0, color="#555555", lw=1.5, linestyle="--",
           zorder=3, label="Ratio = 1.0 (no scaling)")

# Zero SHAP line
ax.axhline(0, color="#aaaaaa", lw=0.8, linestyle=":", zorder=1)

# Labels
ax.set_xlabel(f"{clean_name(FOCUS)}", fontsize=11)
ax.set_ylabel("SHAP value\n(impact on predicted |translation gap|, %BW)", fontsize=10)
ax.set_title(
    f"SHAP Dependence — {clean_name(FOCUS)}\n"
    f"Final LightGBM Model (n={len(df)} preclinical–clinical pairs)\n"
    "(positive SHAP → larger predicted gap; negative → smaller gap)",
    fontweight="bold", fontsize=11, pad=10,
)
ax.legend(fontsize=9, loc="upper right")

n_clipped = (~mask_plot).sum()
if n_clipped:
    ax.annotate(f"{n_clipped} points outside plotted range",
                xy=(0.01, 0.01), xycoords="axes fraction",
                fontsize=8, color="#888888")

plt.tight_layout()
fig.savefig(OUT_FIG, dpi=180, bbox_inches="tight")
fig.savefig(OUT_FIG.with_suffix(".svg"), bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_FIG} + .svg")
