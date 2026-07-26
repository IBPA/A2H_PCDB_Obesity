"""
Candidate-pool-size performance plot.
Mean normalized ranking score (LODO) by preclinical candidate pool size.

Aggregation is DRUG-FIRST: within each stratum we average the score within
each drug (one vote per drug), then average across drugs.
CI = cluster bootstrap resampling drugs within the stratum.

Stratum definitions (derived here from n_preclinical, since the fold-safe
lodo_ranking_results.csv does not carry a precomputed arm_stratum column):
  small   : 1–5 candidate preclinical arms
  medium  : 6–20
  large   : 21+
n_preclinical is constant within a drug, so every drug falls in exactly one
stratum and the drug-first average is well defined per stratum.

Reads the lodo_ranking_results.csv produced by lodo_ranking.py.
Output: outputs/ranking/visualization/pool_size_performance.png (+ .pdf, .svg)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent.parent.parent.parent / "outputs" / "ranking"
LODO_CSV    = RESULTS_DIR / "lodo_ranking_results.csv"
FIG_DIR     = RESULTS_DIR / "visualization"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG     = FIG_DIR / "pool_size_performance.png"


def arm_stratum(n):                       # matches ranking_evaluation._arm_count_stratum
    if n <= 5:   return "small (1-5)"
    if n <= 20:  return "medium (6-20)"
    return "large (21+)"


df  = pd.read_csv(LODO_CSV)
df["arm_stratum"] = df["n_preclinical"].map(arm_stratum)
rng = np.random.default_rng(42)
B   = 10_000

ORDER  = ["small (1-5)", "medium (6-20)", "large (21+)"]
XLABELS = ["Small\n(1–5 preclinical designs)", "Medium\n(6–20 preclinical designs)", "Large\n(≥21 preclinical designs)"]
COLORS  = ["#e8927c", "#5aafe0", "#7bbf70"]

# ── Per-drug means within each stratum (drug-first; used for bars AND dots) ──
drug_stratum = (df.groupby(["drug", "arm_stratum"])["chosen_pct_rank"]
                  .agg(["mean", "count"])
                  .reset_index()
                  .rename(columns={"mean": "pct_rank", "count": "n_cells"}))

# ── Drug-first mean + cluster-bootstrap CI per stratum ───────────────────────
# One vote per drug (avg within drug, then across drugs); CI resamples the
# drugs in the stratum. Matches aggregate_performance.py's drug-level framing.
means, lo, hi, ns = [], [], [], []
for s in ORDER:
    drug_means = drug_stratum[drug_stratum["arm_stratum"] == s]["pct_rank"].values / 100
    k = len(drug_means)
    boot = drug_means[rng.integers(0, k, size=(B, k))].mean(axis=1)   # cluster resample of drugs
    means.append(drug_means.mean())
    lo.append(np.percentile(boot, 2.5))
    hi.append(np.percentile(boot, 97.5))
    ns.append(k)                                                     # number of drugs (one vote each)

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

ax.spines[["top", "right"]].set_visible(False)

# Legend — drug dots
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, fontsize=7.5,
          framealpha=1.0, edgecolor="#cccccc", ncol=2,
          title_fontsize=7.5)

plt.tight_layout()
fig.savefig(OUT_FIG, dpi=180, bbox_inches="tight")
fig.savefig(OUT_FIG.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(OUT_FIG.with_suffix(".svg"), bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_FIG} + .pdf + .svg")
