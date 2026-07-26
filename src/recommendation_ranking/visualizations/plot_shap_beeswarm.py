"""
SHAP beeswarm plot for the fold-safe production recommender.
============================================================
Loads the production model trained on ALL drugs (recommender_model.pkl, from
train_production_model.py), computes SHAP values over the full (rankable) dataset,
and draws a beeswarm of the top features and their effect on predicted |translation
gap|.

Outputs (outputs/ranking/visualization/, each as .png + .svg):
  shap_beeswarm.png               top-10 features — the Figure 4 panel
  shap_beeswarm_all_features.png  all features — supplementary
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

# ranking_core lives one directory up (src/recommendation/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ranking_core import build_base, feat_cols_of, build_eval_cells

warnings.filterwarnings("ignore")

FONT_SIZE = 11
plt.rcParams.update({
    "font.size":       FONT_SIZE,
    "axes.labelsize":  FONT_SIZE,
    "axes.titlesize":  FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
})

PANEL_HEIGHT_IN = 8.5
PANEL_DPI       = 300


SCRIPT_DIR  = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parent.parent.parent
OUT_DIR     = REPO_ROOT / "outputs" / "ranking"
MODEL_PATH  = OUT_DIR / "recommender_model.pkl"
FIG_DIR     = OUT_DIR / "visualization"
OUT_FIG     = FIG_DIR / "shap_beeswarm.png"              # top features, Figure 4 panel
OUT_FIG_TOP_20 = FIG_DIR / "shap_beeswarm_top_20.png"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BEESWARM_JITTER_SEED = 42          # matches ranking_core.RANDOM_STATE
TOP_N = 10                         # features in the Figure 4 panel
# The main panel's height is pinned to the LODO panel, so its row spacing follows
# from TOP_N. The all-feature figure has no panel to match, so it sizes per row —
# every label is two lines, so each row needs ~half an inch to breathe.
FULL_ROW_HEIGHT_IN = 0.55

# Readable feature names (same mapping as the repro figure).
FEATURE_LABELS = {
    "preclinical_animal_weight_before_treatment(grams)":  "Preclinical: animal\nbaseline weight (g)",
    "preclinical_dosage_duration(days)":                   "Preclinical: animal\ntreatment duration (days)",
    "preclinical_animal_age_before_treatment(days)":       "Preclinical: animal\nage at start (days)",
    "preclinical_animal_subject_size":                     "Preclinical: animal\nsubject size (n)",
    "dose_translation_ratio":                              "Single-dose ratio\n(clinical / preclinical-HED)",
    "cumulative_dose_translation_ratio":                   "Cumulative dose ratio\n(clinical / preclinical-HED)",
    "preclinical_animal_strain_Wistar":                    "Preclinical: animal strain\n(Wistar rat: 1, Others: 0)",
    "clinical_sample_size":                                "Clinical: trial\nsample size (n)",
    "preclinical_administration_route_intraperitoneal":    "Preclinical: i.p. route\n(i.p.: 1, Others: 0)",
    "clinical_age_groups_CHILD,ADULT":                     "Clinical age group\n(Child&Adult: 1, Others: 0)",
    "duration_ratio":                                      "Duration ratio\n(clinical/preclinical)",
    "frequency_ratio":                                     "Dosing frequency ratio\n(clinical/preclinical)",
    "is_route_match":                                      "Route match\n(pre=clin: 1, Others: 0)",
    "preclinical_animal_species_rats":                     "Preclinical: animal species\n(rat: 1, Others: 0)",
    "preclinical_disease_model_diet-induced":              "Preclinical: disease model\n(diet-induced: 1, Others: 0)",
    "preclinical_animal_strain_Goto-Kakizaki":             "Preclinical: animal strain\n(Goto-Kakizaki rat: 1, Others: 0)",
    "preclinical_animal_sex_male":                         "Preclinical: animal sex\n(male: 1, Others: 0)",
    "preclinical_animal_sex_female":                       "Preclinical: animal sex\n(female: 1, Others: 0)",
    "preclinical_animal_strain_db/db":                     "Preclinical: animal strain\n(db/db mouse: 1, Others: 0)",
    "clinical_administration_route_oral":                  "Clinical: oral route\n(oral: 1, Others: 0)",
    "clinical_phases_PHASE2":                              "Clinical: trial phase\n(Phase 2: 1, Others: 0)",
    "clinical_phases_PHASE4":                              "Clinical: trial phase\n(Phase 4: 1, Others: 0)",
}


def clean_name(col: str) -> str:
    return FEATURE_LABELS.get(col, col.replace("preclinical_", "Pre: ")
                                       .replace("clinical_", "Clin: ")
                                       .replace("_", " "))


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found — run train_production_model.py first.")
    bundle = joblib.load(MODEL_PATH)
    imp, sc, model, feat_cols = (bundle["imputer"], bundle["scaler"],
                                 bundle["model"], bundle["feat_cols"])

    # Explain the deployed model on the data it was trained on — the complete
    # dataset (all 11 drugs, incl. single-candidate arms), matching
    # train_production_model.py (rankable_only=False).
    cells = build_eval_cells(build_base(), rankable_only=False)
    X_scaled = sc.transform(imp.transform(cells[feat_cols].values))
    print(f"Computing SHAP over {X_scaled.shape[0]} rows × {X_scaled.shape[1]} features…")

    shap_values = shap.TreeExplainer(model).shap_values(X_scaled)
    print(f"  SHAP matrix: {shap_values.shape}")

    clean_cols = [clean_name(c) for c in feat_cols]

    # Main figure: the top features, at the height that aligns with the LODO panel.
    # Clipped: this panel sits in Figure 4 and can't afford the width the outlier tails cost.
    draw_beeswarm(shap_values, X_scaled, clean_cols, TOP_N, PANEL_HEIGHT_IN, OUT_FIG)
    # Supplementary: top 20 features, tall enough that each row keeps its own space.
    # Unclipped: as a standalone supplement it can spend the width to show the full tails.
    draw_beeswarm(shap_values, X_scaled, clean_cols, 20, FULL_ROW_HEIGHT_IN * 20, OUT_FIG_TOP_20,
                  clip=False)


def draw_beeswarm(shap_values, X_scaled, clean_cols: list[str],
                  n_features: int, fig_height: float, out_fig, clip: bool = True) -> None:
    """Beeswarm of the n_features most important features, saved as .png and .svg.
    clip=True pins the x-axis to [-7, 8] and annotates the outliers left off-panel;
    clip=False lets the axis autoscale to the full SHAP range (tails visible, bulk
    compressed near zero)."""
    top_idx = np.argsort(np.abs(shap_values).mean(axis=0))[::-1][:n_features]
    shap_top  = shap_values[:, top_idx]
    X_top     = X_scaled[:, top_idx]
    names_top = [clean_cols[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    # summary_plot jitters the dots vertically from the global RNG; seed it so the
    # figure is byte-reproducible. Affects dot placement only, never the SHAP values.
    np.random.seed(BEESWARM_JITTER_SEED)
    shap.summary_plot(shap_top, X_top, feature_names=names_top,
                      plot_type="dot", max_display=n_features, show=False, plot_size=None)
    fig = plt.gcf()
    fig.set_size_inches(10, fig_height)
    ax = plt.gca()

    # summary_plot hard-codes ~13pt for the feature names and for its "Feature value"
    # colorbar, ignoring rcParams, so reset them here — otherwise this panel's text
    # is visibly larger than the LODO panel's beside it in Figure 4.
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    for extra_ax in fig.axes[1:]:                       # the SHAP colorbar
        extra_ax.tick_params(labelsize=FONT_SIZE)
        extra_ax.yaxis.label.set_size(FONT_SIZE)

    # Clip the x-axis to where the points actually are. The top-10 SHAP values run
    # -7.0 to +30.9, but nothing at all falls between +8 and +31 — so extending the
    # axis past +8 buys no data and costs a quarter of the panel width.
    if clip:
        X_LO, X_HI = -7, 8
        n_clipped = int(((shap_top < X_LO) | (shap_top > X_HI)).sum())
        max_shap  = float(np.abs(shap_top).max())
        ax.set_xlim(X_LO, X_HI)
        # Vertically centred on the top feature's row — that row is where the clipped
        # points belong, and its right half is empty. Located by matching the tick label
        # rather than assuming a y position, since summary_plot inverts the axis.
        tick_labels = [t.get_text() for t in ax.get_yticklabels()]
        top_row_y = ax.get_yticks()[tick_labels.index(names_top[0])]
        ax.text(X_HI, top_row_y, f"▶ {n_clipped} outliers clipped\n(max |SHAP| ≈ {max_shap:.0f})",
                color="#555555", ha="right", va="center", linespacing=1.3)

    # |δ| is the paper's notation for the absolute translation gap — the model's
    # prediction target — so the figure uses it throughout rather than mixing
    # the symbol in the caption with the words on the axis.
    ax.set_xlabel("SHAP value\n(impact on predicted |δ|, %BW)")
    ax.set_title("")
    plt.tight_layout()
    fig.text(0.5, -0.01,
             "(positive SHAP → larger predicted |δ|; negative → smaller predicted |δ|)",
             ha="center", va="top", fontstyle="italic", color="#444444")
    fig.savefig(out_fig, dpi=PANEL_DPI, bbox_inches="tight")
    fig.savefig(out_fig.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_fig.name} + .svg  ({n_features} features, {fig_height:.1f} in tall)")


if __name__ == "__main__":
    main()
