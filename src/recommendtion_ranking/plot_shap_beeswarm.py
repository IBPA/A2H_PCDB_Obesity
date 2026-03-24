"""
SHAP beeswarm plot from the final model trained on all data.

Trains LightGBM (saved tuned params) on the full dataset, computes SHAP values
for all predictions, and generates a summary beeswarm plot showing the top-20
most important features and their effect direction on predicted |translation gap|.
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
OUT_FIG     = RESULTS_DIR / "R8_shap_beeswarm.png"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "translation_outcome"
ABS_TGT = "abs_gap"

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df[ABS_TGT] = df[TARGET].abs()

bridge_cols      = ["dose_translation_ratio", "cumulative_dose_translation_ratio",
                    "duration_ratio", "frequency_ratio", "is_route_match"]
clinical_cols    = [c for c in df.columns if c.startswith("clinical_")
                    and c != "clinical_arm_id"]
preclinical_cols = [c for c in df.columns if c.startswith("preclinical_")]
feat_cols        = preclinical_cols + bridge_cols + clinical_cols

X = df[feat_cols].values
y = df[ABS_TGT].values

# ── Load tuned params & train on all data ─────────────────────────────────────
with open(PARAMS_PATH) as f:
    saved = json.load(f)
params = saved["params"]
print(f"Training LightGBM on all {len(df)} rows...")

sc = StandardScaler()
X_scaled = sc.fit_transform(X)

model = lgb.LGBMRegressor(**params)
model.fit(X_scaled, y)
print("  Done.")

# ── SHAP values ───────────────────────────────────────────────────────────────
print("Computing SHAP values...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_scaled)   # shape (n_samples, n_features)
print(f"  SHAP matrix: {shap_values.shape}")

# ── Readable feature names ────────────────────────────────────────────────────
FEATURE_LABELS = {
    "preclinical_animal_weight_before_treatment(grams)":  "Preclinical: animal\nbaseline weight (g)",
    "preclinical_dosage_duration(days)":                   "Preclinical: animal\ntreatment duration (days)",
    "preclinical_animal_age_before_treatment(days)":       "Preclinical: animal\nage at start (days)",
    "preclinical_animal_subject_size":                     "Preclinical: animal\ngroup size (n)",
    "dose_translation_ratio":                              "Single-dose ratio\n(preclinical/clinical)",
    "cumulative_dose_translation_ratio":                   "Cumulative dose ratio\n(preclinical/clinical)",
    "preclinical_animal_strain_Wistar":                    "Preclinical: animal strain\n(Wistar rat: 1, Others: 0)",
    "clinical_sample_size":                                "Clinical: trial\nsample size (n)",
    "preclinical_administration_route_intraperitoneal":    "Preclinical: i.p. route\n(i.p.: 1, Others: 0)",
    "clinical_age_groups_CHILD,ADULT":                     "Clinical age group\n(Child&Adult: 1, Others: 0)",
    "duration_ratio":                                      "Duration ratio\n(preclinical/clinical)",
    "frequency_ratio":                                     "Dosing frequency ratio\n(preclinical/clinical)",
    "is_route_match":                                      "Route match\n(pre=clin: 1, Others: 0)",
}

def clean_name(col: str) -> str:
    return FEATURE_LABELS.get(col, col.replace("preclinical_", "Pre: ")
                                       .replace("clinical_", "Clin: ")
                                       .replace("_", " "))

clean_cols = [clean_name(c) for c in feat_cols]

# ── Beeswarm plot — top 20 features ──────────────────────────────────────────
TOP_N = 10
mean_abs_shap = np.abs(shap_values).mean(axis=0)
top_idx = np.argsort(mean_abs_shap)[::-1][:TOP_N]

shap_top  = shap_values[:, top_idx]
X_top     = X_scaled[:, top_idx]       # for color encoding
names_top = [clean_cols[i] for i in top_idx]

fig, ax = plt.subplots(figsize=(10, 8))

# Draw beeswarm manually so we control axes
shap.summary_plot(
    shap_top, X_top,
    feature_names=names_top,
    plot_type="dot",
    max_display=TOP_N,
    show=False,
    plot_size=None,
)

fig = plt.gcf()
fig.set_size_inches(12, 8)

ax = plt.gca()

# Clip x-axis; note how many points fall outside
X_CLIP = 12
n_clipped = int((np.abs(shap_values[:, top_idx]).max(axis=1) > X_CLIP).sum())
ax.set_xlim(-7, X_CLIP)
ax.annotate(f"▶ {n_clipped} outliers clipped (up to +34)\nlight (~20g) mice treated with semaglutide",
            xy=(X_CLIP, ax.get_ylim()[1] * 0.93),
            fontsize=7.5, color="#555555", ha="right", va="top")

ax.set_xlabel(
    "SHAP value\n(impact on predicted |translation gap|, %BW)",
    fontsize=10)
ax.set_title("")

plt.tight_layout()
fig.text(0.5, -0.01,
         "(positive SHAP → larger predicted |gap|; negative → smaller predicted |gap|)",
         ha="center", va="top", fontsize=9, fontstyle="italic", color="#444444")
fig.savefig(OUT_FIG, dpi=180, bbox_inches="tight")
fig.savefig(OUT_FIG.with_suffix(".svg"), bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_FIG} + .svg")
