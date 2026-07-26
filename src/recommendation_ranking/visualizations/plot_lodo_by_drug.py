"""
Regenerate lodo_by_drug.png with 95% CI error bars.
Reads the lodo_ranking_results.csv and per_drug_gap_ci.csv produced by
lodo_ranking.py and generate_per_drug_gap_ci.py.

Output: outputs/ranking/visualization/lodo_by_drugs.png (+ .pdf, .svg)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap

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


RESULTS_DIR = Path(__file__).parent.parent.parent.parent / "outputs" / "ranking"
LODO_CSV    = RESULTS_DIR / "lodo_ranking_results.csv"
CI_CSV      = RESULTS_DIR / "per_drug_gap_ci.csv"
FIG_DIR     = RESULTS_DIR / "visualization"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG     = FIG_DIR / "lodo_by_drugs.png"


def per_drug_summary(lodo_df):
    rows = []
    for drug, grp in lodo_df.groupby("drug"):
        evaluable = grp[grp["n_preclinical"] >= 3]
        n_preclinical_total = int(grp["n_preclinical"].mode().iloc[0])
        rows.append({
            "drug":              drug,
            "n_cells":           grp["eval_cell"].nunique(),
            "n_ncts":            grp["NCT"].nunique(),
            "n_preclinical_tot": n_preclinical_total,
            "n_cells_eval":      len(evaluable),
            "status_quo":        round(grp["status_quo"].mean(), 2),
            "chosen_gap":        round(grp["chosen_gap"].mean(), 2),
            "improvement":       round(grp["improvement"].mean(), 2),
            "pct_improvement":   round(100 * grp["improvement"].mean()
                                       / grp["status_quo"].mean(), 1),
            "mean_pct_rank":     round(grp["chosen_pct_rank"].mean(), 1),
            "mean_rho":          round(evaluable["spearman_rho"].mean(), 3)
                                 if len(evaluable) > 0 else float("nan"),
        })
    return (pd.DataFrame(rows)
            .sort_values("pct_improvement", ascending=False)
            .reset_index(drop=True))


lodo_df = pd.read_csv(LODO_CSV)
ci_df   = pd.read_csv(CI_CSV)

drug_df = per_drug_summary(lodo_df)
drug_df = drug_df.merge(ci_df[["drug","sq_lo","sq_hi","cg_lo","cg_hi"]],
                        on="drug", how="left")

drugs  = drug_df["drug"].tolist()
x      = np.arange(len(drugs))
width  = 0.32

xlabels = [
    f"{d}\n(prec.={r.n_preclinical_tot}, clin.arms={r.n_cells},\n clin.studies={r.n_ncts})"
    for d, r in zip(drugs, drug_df.itertuples())
]

# Stacked so the two panels share one x-axis: the drugs are in the same order in
# both, so the tick labels are drawn once, under the bottom panel.
fig, axes = plt.subplots(2, 1, figsize=(10, PANEL_HEIGHT_IN), sharex=True)

# ── Panel 1: Status quo vs recommended gap, with CI error bars ────────────────
ax = axes[0]

# Status quo bars + CI
sq_yerr_lo = (drug_df["status_quo"] - drug_df["sq_lo"]).clip(lower=0)
sq_yerr_hi = (drug_df["sq_hi"] - drug_df["status_quo"]).clip(lower=0)
sq_yerr = np.array([
    sq_yerr_lo.where(drug_df["sq_lo"].notna(), other=0).values,
    sq_yerr_hi.where(drug_df["sq_hi"].notna(), other=0).values,
])

ax.bar(x - width/2, drug_df["status_quo"], width,
       label="Baseline (avg. all arms)", color="#bdc3c7", edgecolor="white")
ax.errorbar(x - width/2, drug_df["status_quo"],
            yerr=sq_yerr,
            fmt="none", color="#7f8c8d", capsize=4, linewidth=1.4, capthick=1.4)

# Recommended bars + CI
cg_yerr_lo = (drug_df["chosen_gap"] - drug_df["cg_lo"]).clip(lower=0)
cg_yerr_hi = (drug_df["cg_hi"] - drug_df["chosen_gap"]).clip(lower=0)
cg_yerr = np.array([
    cg_yerr_lo.where(drug_df["cg_lo"].notna(), other=0).values,
    cg_yerr_hi.where(drug_df["cg_hi"].notna(), other=0).values,
])

rec_colors = ["#27ae60" if imp >= 0 else "#f5a623"
              for imp in drug_df["improvement"]]
b2 = ax.bar(x + width/2, drug_df["chosen_gap"], width,
            label="Model recommendation (LODO)", color=rec_colors, edgecolor="white")
ax.errorbar(x + width/2, drug_df["chosen_gap"],
            yerr=cg_yerr,
            fmt="none", color="#2c3e50", capsize=4, linewidth=1.4, capthick=1.4)

# Annotations above recommended bars
for bar, row in zip(b2, drug_df.itertuples()):
    h = bar.get_height()
    # Position annotation above the CI upper bound (or bar top if no CI)
    ci_top = row.cg_hi if not np.isnan(row.cg_hi) else h
    sign  = "▼" if row.improvement >= 0 else "▲"
    color = "#1a7a45" if row.improvement >= 0 else "#c87000"
    ax.text(bar.get_x() + bar.get_width() / 2, ci_top + 0.5,
            f"{h:.1f}\n{sign}{abs(row.pct_improvement):.0f}%",
            ha="center", va="bottom", color=color)

# |δ| = absolute translation gap, the paper's notation for the prediction target.
ax.set_ylabel("Mean actual |δ| (%BW)")                 # x tick labels: bottom panel only

legend_elements = [
    Patch(facecolor="#bdc3c7", label="Baseline (mean of all arms)"),
    Patch(facecolor="#27ae60", label="Recommendation — reduces gap (▼)"),
    Patch(facecolor="#f5a623", label="Recommendation — increases gap (▲)"),
]
ax.set_ylim(top=27)
ax.set_yticks([0, 5, 10, 15, 20, 25])
ax.legend(handles=legend_elements, loc="upper left")

# ── Panel 2: Chosen-arm percentile rank per drug ──────────────────────────────
ax2 = axes[1]

n_cells_arr = drug_df["n_cells"].values.astype(float)
# Scale over the OBSERVED range: every drug has >=1 clinical arm, so starting the
# scale at 0 would show a value no drug can take. Blues is truncated at the pale end
# because vmin now maps to the bottom of the colormap, which would be pure white.
blues = LinearSegmentedColormap.from_list(
    "blues_observed", plt.cm.Blues(np.linspace(0.15, 1.0, 256)))
norm  = plt.Normalize(vmin=n_cells_arr.min(), vmax=n_cells_arr.max())
dot_colors = blues(norm(n_cells_arr))

rank_vals = drug_df["mean_pct_rank"] / 100

ax2.scatter(x, rank_vals,
            s=180, c=dot_colors,
            edgecolors="#333333", linewidths=0.8, zorder=3)

ax2.axhline(0.5, color="#e74c3c", linestyle="--", lw=1.5)
# Label the line in place rather than in a legend. Blended transform: x in axes
# fraction (pin to the left edge), y in data units (sit just under the line).
ax2.text(0.01, 0.5 - 0.025, "random selection baseline",
         transform=ax2.get_yaxis_transform(),
         ha="left", va="top", color="#e74c3c")

for xi, val in zip(x, rank_vals):
    ax2.annotate(f"{val:.2f}",
                 (xi, val),
                 textcoords="offset points", xytext=(0, 10),
                 ha="center",
                )

ax2.set_xticks(x)
ax2.set_xticklabels(xlabels, rotation=40, ha="right")
# "0.5 = random selection" would repeat the in-panel label on the dashed line.
ax2.set_ylabel("Mean normalized ranking score\n(higher = better)")
ax2.set_ylim(-0.05, 1.12)
ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
# Shade key: a slim colorbar inset in the panel's empty lower-left corner, so it
# needs no width from the axes and the two stacked panels stay the same size.
sm = plt.cm.ScalarMappable(cmap=blues, norm=norm)
sm.set_array([])
cax = ax2.inset_axes([0.035, 0.10, 0.011, 0.17])
cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
# Label on the LEFT, rotated like a y-axis title; ticks stay on the right, so the
# two never collide.
# Text kept short so it fits within the bar height — a longer label is centred
# on the bar and overflows past the bottom of the panel.
cbar.set_label("# clinical arms", rotation=90, labelpad=5)
cax.yaxis.set_label_position("left")
# Endpoints + midpoint, over the observed range (min arm count, not 0).
cbar.set_ticks(np.linspace(n_cells_arr.min(), n_cells_arr.max(), 3).round().astype(int))
cbar.ax.tick_params(length=2, pad=1.5)
cbar.outline.set_linewidth(0.5)

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=PANEL_DPI, bbox_inches="tight")
plt.savefig(OUT_FIG.with_suffix(".pdf"), bbox_inches="tight")
plt.savefig(OUT_FIG.with_suffix(".svg"), bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_FIG} + .pdf + .svg")
