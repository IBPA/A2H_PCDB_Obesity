"""
Regenerate R0_lodo_by_drug.png with 95% CI error bars on Panel 1.
Reads lodo_ranking_results.csv and per_drug_gap_ci.csv.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

RESULTS_DIR = Path(__file__).parent.parent.parent / "outputs" / "ranking_evaluation"
LODO_CSV    = RESULTS_DIR / "lodo_ranking_results.csv"
CI_CSV      = RESULTS_DIR / "per_drug_gap_ci.csv"
OUT_FIG     = RESULTS_DIR / "R0_lodo_by_drug.png"


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

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

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
            ha="center", va="bottom", fontsize=9, color=color)

ax.set_xticks(x)
ax.set_xticklabels(xlabels, rotation=40, ha="right", fontsize=8)
ax.set_ylabel("Mean actual |translation gap| (%BW)")

legend_elements = [
    Patch(facecolor="#bdc3c7", label="Baseline (mean of all arms)"),
    Patch(facecolor="#27ae60", label="Recommendation — reduces gap (▼)"),
    Patch(facecolor="#f5a623", label="Recommendation — increases gap (▲)"),
]
ax.set_ylim(top=35)
ax.set_yticks([0, 10, 20, 30])
ax.legend(handles=legend_elements, fontsize=8, loc="upper left")

# ── Panel 2: Chosen-arm percentile rank per drug ──────────────────────────────
ax2 = axes[1]

n_cells_arr = drug_df["n_cells"].values.astype(float)
blues = plt.cm.Blues
norm  = plt.Normalize(vmin=0, vmax=n_cells_arr.max())
dot_colors = blues(norm(n_cells_arr))

rank_vals = drug_df["mean_pct_rank"] / 100

ax2.scatter(x, rank_vals,
            s=180, c=dot_colors,
            edgecolors="#333333", linewidths=0.8, zorder=3)

ax2.axhline(0.5, color="#e74c3c", linestyle="--", lw=1.5,
            label="0.5 — random pick baseline")

for xi, val in zip(x, rank_vals):
    ax2.annotate(f"{val:.2f}",
                 (xi, val),
                 textcoords="offset points", xytext=(0, 10),
                 ha="center",
                 fontsize=9,
                )

ax2.set_xticks(x)
ax2.set_xticklabels(xlabels, rotation=40, ha="right", fontsize=8)
ax2.set_ylabel("Mean normalized ranking score\n"
              "(higher = better; 0.5 = random selection)", fontsize=10)
ax2.set_ylim(-0.05, 1.12)
ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
sm = plt.cm.ScalarMappable(cmap=blues, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax2, shrink=0.6, pad=0.02)
cbar.set_label("# clinical trial arms", fontsize=8)
cbar.locator = MaxNLocator(integer=True)
cbar.update_ticks()
ax2.legend(fontsize=8, loc="upper right")

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
plt.savefig(OUT_FIG.with_suffix(".svg"), bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_FIG} + .svg")
