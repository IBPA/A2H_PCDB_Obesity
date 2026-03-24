"""
Generate Figure: Bootstrap rank-1 frequency for top candidate designs.
Two-panel figure: Panel A = semaglutide, Panel B = liraglutide.
x-axis: top-5 candidate preclinical designs (by top-1 bootstrap frequency)
y-axis: proportion of bootstrap resamples in which that design ranked first
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent.parent.parent / "outputs" / "ranking_evaluation"
IN_CSV  = RESULTS_DIR / "bootstrap_rank_stability_candidate.csv"
OUT_FIG = RESULTS_DIR / "R6_rank_stability.png"

df = pd.read_csv(IN_CSV)


def short_label(design_id: str) -> str:
    """Compress a full design label to the key distinguishing fields."""
    parts = design_id.split("/")
    # parts order: species, strain, model, sex, route, duration, n, age, weight
    species  = parts[0]
    strain   = parts[1]
    model    = parts[2]
    route    = parts[4]
    duration = parts[5]          # e.g. "14d"
    n_val    = parts[6]          # e.g. "n=8"
    # Shorten species
    sp = "mice" if species == "mice" else "rats" if species == "rats" else species
    return f"{strain}/{model}\n{route}/{duration}/{n_val}"


fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)
fig.suptitle(
    "Bootstrap Top-1 Frequency by Candidate Preclinical Design\n"
    "(proportion of resamples in which each design ranked first)",
    fontweight="bold", fontsize=12, y=1.01)

DRUG_META = {
    "semaglutide": {
        "ax_label": "A",
        "color_top": "#2ca02c",
        "color_rest": "#a8d5a2",
        "n_eval_cells": 17,
    },
    "liraglutide": {
        "ax_label": "B",
        "color_top": "#d62728",
        "color_rest": "#f4a7a8",
        "n_eval_cells": 19,
    },
}

for ax, drug in zip(axes, ["semaglutide", "liraglutide"]):
    meta = DRUG_META[drug]
    sub = (df[df["held_out_drug"] == drug]
           .sort_values("top1_frequency", ascending=False)
           .head(5)
           .reset_index(drop=True))

    labels  = [short_label(d) for d in sub["design_id"]]
    freqs   = sub["top1_frequency"].values
    orig_rk = sub["original_rank"].values

    # Color top-ranked design differently from competitors
    colors = [meta["color_top"] if i == 0 else meta["color_rest"]
              for i in range(len(sub))]

    bars = ax.bar(np.arange(len(sub)), freqs, color=colors,
                  edgecolor="white", linewidth=0.8, width=0.6)

    # Frequency annotation above each bar
    for i, (bar, freq, rk) in enumerate(zip(bars, freqs, orig_rk)):
        h = bar.get_height()
        if freq >= 0.01:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                    f"{freq:.1%}\n(rank #{rk})",
                    ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold" if i == 0 else "normal",
                    color="#222222")

    ax.set_xticks(np.arange(len(sub)))
    ax.set_xticklabels(labels, fontsize=8, ha="center")
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_ylabel("Top-1 bootstrap frequency", fontsize=10)
    ax.set_xlabel("Candidate preclinical design\n(species/strain/model · route/duration/n)",
                  fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    n_designs = df[df["held_out_drug"] == drug]["design_id"].nunique()
    ax.set_title(
        f"({'A' if drug == 'semaglutide' else 'B'})  {drug.capitalize()}\n"
        f"K={n_designs} candidate designs,  C={meta['n_eval_cells']} evaluation cells",
        fontweight="bold", fontsize=11, loc="left")

    # Note: top-5 shown; remaining designs have ~0% frequency
    ax.text(0.98, 0.97,
            f"Top 5 shown; remaining\n{n_designs - 5} designs: ~0%",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5, color="#666666", style="italic")

plt.tight_layout()
fig.savefig(OUT_FIG, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_FIG}")
