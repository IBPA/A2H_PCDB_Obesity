"""
Candidate-pool-size performance plot.
Mean chosen-arm percentile rank (LODO) by preclinical candidate pool size.

Stratum definitions (arm_stratum column):
  small   : 1–5 candidate preclinical arms
  medium  : 6–20
  large   : 21+
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent.parent.parent / "outputs" / "ranking_evaluation"
LODO_CSV    = RESULTS_DIR / "lodo_ranking_results.csv"
OUT_FIG     = RESULTS_DIR / "R7_pool_size_performance.png"

df  = pd.read_csv(LODO_CSV)
rng = np.random.default_rng(42)
B   = 10_000

ORDER  = ["small (1-5)", "medium (6-20)", "large (21+)"]
XLABELS = ["Small\n(1–5 preclinical designs)", "Medium\n(6–20 preclinical designs)", "Large\n(≥21 preclinical designs)"]
COLORS  = ["#e8927c", "#5aafe0", "#7bbf70"]

# ── Bootstrap CI per stratum ────────────────────────────────────────────────────
means, lo, hi, ns = [], [], [], []
for s in ORDER:
    vals = df[df["arm_stratum"] == s]["chosen_pct_rank"].values
    boot = np.array([vals[rng.integers(0, len(vals), len(vals))].mean()
                     for _ in range(B)])
    means.append(vals.mean() / 100)
    lo.append(np.percentile(boot, 2.5) / 100)
    hi.append(np.percentile(boot, 97.5) / 100)
    ns.append(len(vals))

# ── Per-drug means within each stratum (for dot overlay) ─────────────────────
drug_stratum = (df.groupby(["drug", "arm_stratum"])["chosen_pct_rank"]
                  .agg(["mean", "count"])
                  .reset_index()
                  .rename(columns={"mean": "pct_rank", "count": "n_cells"}))

# ── Plot ────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5))

x = np.arange(len(ORDER))
bars = ax.bar(x, means, color=COLORS, edgecolor="white",
              linewidth=0.8, width=0.35, zorder=2)

# Error bars
yerr = np.array([[m - l for m, l in zip(means, lo)],
                 [h - m for m, h in zip(means, hi)]])
ax.errorbar(x, means, yerr=yerr, fmt="none",
            color="#333333", capsize=6, linewidth=1.8, capthick=1.8, zorder=3)

# Random baseline
ax.axhline(0.5, color="#e74c3c", linestyle="--", linewidth=1.6,
           label="0.5 (random selection)", zorder=1)

# Per-drug dots
DRUG_MARKERS = {
    "tirzepatide":    ("^", "#c0392b", "tirzepatide"),
    "canagliflozin":  ("o", "#2c3e50", "canagliflozin"),
    "medi0382":       ("o", "#8e44ad", "medi0382"),
    "survodutide":    ("o", "#16a085", "survodutide"),
    "exenatide":      ("o", "#2980b9", "exenatide"),
    "semaglutide":    ("s", "#27ae60", "semaglutide"),
    "liraglutide":    ("D", "#d35400", "liraglutide"),
    "metformin":      ("o", "#7f8c8d", "metformin"),
    "orlistat":       ("o", "#95a5a6", "orlistat"),
}

jitter_rng = np.random.default_rng(7)
plotted_labels = set()
for _, row in drug_stratum.iterrows():
    s_idx = ORDER.index(row["arm_stratum"])
    marker, color, label = DRUG_MARKERS.get(
        row["drug"], ("o", "#aaaaaa", row["drug"]))
    jitter = jitter_rng.uniform(-0.14, 0.14)
    legend_label = label if label not in plotted_labels else "_nolegend_"
    plotted_labels.add(label)
    ax.scatter(s_idx + jitter, row["pct_rank"] / 100,
               marker=marker, color=color, s=70, zorder=4,
               edgecolors="white", linewidths=0.6,
               label=legend_label)
    # Drug name annotation for outliers
    # if row["drug"] == "tirzepatide":
    #     ax.annotate("tirzepatide\n(dose-arm only)",
    #                 (s_idx + jitter, row["pct_rank"]),
    #                 xytext=(14, 4), textcoords="offset points",
    #                 fontsize=7, color="#c0392b",
    #                 arrowprops=dict(arrowstyle="-", color="#c0392b",
    #                                lw=0.8))

# Value + CI annotation above bars
for xi, m, l, h, n in zip(x, means, lo, hi, ns):
    ax.text(xi, h + 0.015, f"{m:.2f}\n[{l:.2f}, {h:.2f}]",
            ha="center", va="bottom", fontsize=10, color="#333333")

ax.set_xticks(x)
ax.set_xticklabels(XLABELS, fontsize=11)
ax.set_ylabel("Mean normalized ranking score\n"
              "(higher = better; 0.5 = random selection)", fontsize=10)
ax.set_ylim(-0.02, 1.15)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
# ax.set_title("Recommender Performance by Candidate Pool Size\n"
#              "(LODO evaluation, 95% bootstrap CI)",
#              fontweight="bold", fontsize=12, pad=8)

ax.spines[["top", "right"]].set_visible(False)

# Legend — drug dots
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, fontsize=7.5, 
        #   loc="upper right",
          framealpha=1.0, edgecolor="#cccccc", ncol=2,
        #   title="Individual drugs",
          title_fontsize=7.5)

plt.tight_layout()
fig.savefig(OUT_FIG, dpi=180, bbox_inches="tight")
fig.savefig(OUT_FIG.with_suffix(".svg"), bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_FIG} + .svg")
