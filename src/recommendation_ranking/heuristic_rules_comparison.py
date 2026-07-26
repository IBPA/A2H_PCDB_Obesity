#!/usr/bin/env python3
"""
Heuristic-rule ranking comparison.
=====================================================================
Builds every row of the ranking-evaluation summary table and (with ``--write``)
writes it to outputs/ranking_summary.csv.

Rows of the summary
-------------------
  * Rule-based heuristics
        recomputed from the raw data via ``run_all_rule_baselines`` -> ``aggregate``
        (deterministic).
  * LODO model row
        re-aggregated from the per-cell (per clinical arm) predictions
        (outputs/lodo_ranking_results.csv,
        written by lodo_ranking.py) via ``aggregate``.
  * Random selection (baseline)
        the analytic expectation of a uniform-random pick = the mean of all
        candidate arms = status_quo, taken PER aggregation level so each
        %-improvement column closes exactly against it. Deterministic.

Usage:
    python src/recommendation/heuristic_rules_comparison.py            # print
    python src/recommendation/heuristic_rules_comparison.py --write    # also write the CSV
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the corrected (fold-safe) core utilities — same directory.
from ranking_core import (
    build_base,
    build_eval_cells,
    aggregate,
    normalized_ranking_score,
    ABS_TGT,
    GROUP,
    NCT,
    EVAL_GROUP,
)

SCRIPT_DIR   = Path(__file__).parent
REPO_ROOT    = SCRIPT_DIR.parent.parent
RESULTS_DIR  = REPO_ROOT / "outputs" / "ranking"
SUMMARY_PATH = RESULTS_DIR / "ranking_summary.csv"
LODO_PRED_PATH = RESULTS_DIR / "lodo_ranking_results.csv"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Human-readable labels for each picker (display order is by gap, below).
DISPLAY_NAME = {
    "LODO_model":            "LightGBM recommendation (LODO)",
    "random_pick":           "Random selection (baseline)",
    "rule_largest_n":        "Largest sample size",
    "rule_dose_ratio→1":     "Single-dose ratio → 1",
    "rule_cumdose_ratio→1":  "Cumulative dose ratio → 1",
    "rule_route_match":      "Route match (mean of same-route arms)",
    "rule_duration_ratio→1": "Duration ratio → 1",
}


# ── Arm-count stratum (matches ranking_evaluation._arm_count_stratum) ─────────
def _arm_count_stratum(n):
    if n <= 5:   return "small (1-5)"
    if n <= 20:  return "medium (6-20)"
    return "large (21+)"


# ── Rule baselines ───────────────────────────────────────────────────────────

def _eval_picker(df, pick_fn, name):
    """
    For each (NCT × clinical arm) eval cell, apply pick_fn(cell_rows), which is
    self-contained and returns (chosen_gap, chosen_pct_rank) — either a single-arm
    pick (via _score_pick) or a mean-of-matched pool (via _score_pool). This
    function only assembles the per-cell record around that score.
    Ranking is over preclinical designs only; clinical context is fixed within each cell.
    chosen_pct_rank: 100=best, 50=random expectation, 0=worst.
    """
    records = []
    for cell_id in sorted(df[EVAL_GROUP].unique()):
        rows = df[df[EVAL_GROUP] == cell_id].copy()
        if len(rows) == 0:
            continue
        chosen_gap, pct_rank = pick_fn(rows)
        status_quo = float(rows[ABS_TGT].mean())
        records.append({
            "eval_cell":       cell_id,
            "NCT":             rows[NCT].iloc[0],
            "drug":            rows[GROUP].iloc[0],
            "n_preclinical":   len(rows),
            "arm_stratum":     _arm_count_stratum(len(rows)),
            "status_quo":      status_quo,
            "chosen_gap":      chosen_gap,
            "improvement":     status_quo - chosen_gap,
            "chosen_pct_rank": pct_rank,
            "picker":          name,
        })
    return pd.DataFrame(records)


# Each rule encodes a domain-motivated hypothesis about which arm is most
# translationally predictive:
#
# Rule                   | Hypothesis                                  | Picker
# -----------------------|---------------------------------------------|---------------
# duration_ratio→1       | Closest preclinical/clinical duration match | idxmin
# dose_ratio→1           | Closest dose (mg/kg) match                  | idxmin
# cumulative_dose→1      | Closest total exposure match                | idxmin
# largest_n              | Largest animal group → most stable estimate | idxmax
# route_match            | Same administration route as human trial    | mean-of-matched

# Each rule is self-contained: it returns (chosen_gap, chosen_pct_rank) for the
# cell, either via _score_pick (one chosen arm) or _score_pool (mean over a matched
# subset). _eval_picker just records the result.

def _score_pick(rows, idx):
    """Single-arm pick → (chosen_gap, chosen_pct_rank)."""
    gaps = rows[ABS_TGT].values
    g = float(rows.loc[idx, ABS_TGT])
    return g, normalized_ranking_score(gaps, g)

def _score_pool(rows, pool_idx):
    """Mean over a matched pool → (mean gap, mean pct_rank): the expected value of a
    uniform random pick within the matched arms ('mean-of-matched')."""
    gaps = rows[ABS_TGT].values
    pool_gaps = rows.loc[pool_idx, ABS_TGT].values
    return (float(np.mean(pool_gaps)),
            float(np.mean([normalized_ranking_score(gaps, g) for g in pool_gaps])))

def _closest_to_one(rows, col):
    # idxmin skips NaN; if the whole cell is missing this feature, no arm can be
    # ranked by it — fall back to the first arm (deterministic).
    diffs = rows[col].sub(1).abs()
    idx = rows.index[0] if diffs.isna().all() else diffs.idxmin()
    return _score_pick(rows, idx)

def _route_match(rows):
    # Membership filter → score the MEAN over the whole same-route pool
    # (mean-of-matched). Falls back to all arms when no arm matches the route.
    matched = rows[rows["is_route_match"] == 1]
    pool = matched if len(matched) > 0 else rows
    return _score_pool(rows, pool.index.values)

def _largest_n(rows):
    col = rows["preclinical_animal_subject_size"]
    idx = rows.index[0] if col.isna().all() else col.idxmax()
    return _score_pick(rows, idx)

RULES = {
    "rule_duration_ratio→1":    lambda rows: _closest_to_one(rows, "duration_ratio"),
    "rule_dose_ratio→1":        lambda rows: _closest_to_one(rows, "dose_translation_ratio"),
    "rule_cumdose_ratio→1":     lambda rows: _closest_to_one(rows, "cumulative_dose_translation_ratio"),
    "rule_route_match":         _route_match,
    "rule_largest_n":           _largest_n,
}


def run_all_rule_baselines(df):
    """Evaluate every heuristic rule in RULES and aggregate its per-cell records."""
    results = {}
    for name, fn in RULES.items():
        records = _eval_picker(df, fn, name)
        results[name] = {**aggregate(records), "records": records}
        print(f"  {name:35s}  chosen_gap={results[name]['cell_mean_chosen_gap']:.3f}  "
              f"SQ={results[name]['cell_mean_status_quo']:.3f}  "
              f"Δ(cell)={results[name]['cell_pct_improvement']:+.1f}%  "
              f"Δ(drug)={results[name]['drug_pct_improvement']:+.1f}%")
    return results


# ── Row builders ─────────────────────────────────────────────────────────────

def _summary_row(picker: str, agg: dict) -> dict:
    return {
        "picker":          picker,
        "cell_chosen_gap": agg["cell_mean_chosen_gap"],
        "drug_chosen_gap": agg["drug_mean_chosen_gap"],
        "cell_pct_impr":   agg["cell_pct_improvement"],
        "drug_pct_impr":   agg["drug_pct_improvement"],
    }


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    rule_results = run_all_rule_baselines(df)

    # Random selection (baseline) — analytic expectation of a uniform pick =
    # status_quo, taken per aggregation level so each %-column closes exactly
    # against it. status_quo is picker-independent, so read it off any rule's
    # aggregate; improvement is 0% by definition.
    any_agg = next(iter(rule_results.values()))
    rows.append({"picker": "random_pick",
                 "cell_chosen_gap": any_agg["cell_mean_status_quo"],
                 "drug_chosen_gap": any_agg["drug_mean_status_quo"],
                 "cell_pct_impr": 0.0, "drug_pct_impr": 0.0})

    # Rule-based heuristics.
    for name, res in rule_results.items():
        rows.append(_summary_row(name, res))

    # LODO model row — aggregate the fold-safe per-cell predictions.
    if LODO_PRED_PATH.exists():
        rows.append(_summary_row("LODO_model", aggregate(pd.read_csv(LODO_PRED_PATH))))
    else:
        print(f"  ! {LODO_PRED_PATH.name} not found — skipping LODO_model. "
              f"Run lodo_ranking.py to regenerate the fold-safe predictions.")

    return (pd.DataFrame(rows)
            .sort_values("drug_chosen_gap")
            .reset_index(drop=True))


# ── Text rendering ───────────────────────────────────────────────────────────

def _pct(v: float) -> str:
    return ("0.0%" if v == 0 else f"{v:+.1f}%").replace("-", "−")


def print_summary(summary: pd.DataFrame) -> None:
    name_w = max(len(DISPLAY_NAME.get(p, p)) for p in summary["picker"])
    header = (f"{'Strategy':<{name_w}}  "
              f"{'Cell |gap|':>10}  {'Cell %impr':>11}  "
              f"{'Drug |gap|':>10}  {'Drug %impr':>11}")
    print()
    print("Ranking-evaluation summary — fold-safe  (drug-level |gap| = value reported in the paper)")
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for _, r in summary.iterrows():
        name = DISPLAY_NAME.get(r["picker"], r["picker"])
        print(f"{name:<{name_w}}  "
              f"{r['cell_chosen_gap']:>10.2f}  {_pct(r['cell_pct_impr']):>11}  "
              f"{r['drug_chosen_gap']:>10.2f}  {_pct(r['drug_pct_impr']):>11}")
    print("-" * len(header))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true",
                    help="overwrite outputs/ranking_summary.csv")
    args = ap.parse_args()

    cells = build_eval_cells(build_base())     # raw features, NO global-median imputation
    summary = build_summary(cells)
    print_summary(summary)

    if args.write:
        summary.to_csv(SUMMARY_PATH, index=False)
        print(f"\nWrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
