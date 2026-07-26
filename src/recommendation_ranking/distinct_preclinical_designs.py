"""
distinct_preclinical_designs.py — distinct preclinical designs per drug.
========================================================================
Show, reproducibly, how many DISTINCT preclinical study designs each drug's candidate
pool contains.

WHAT COUNTS AS A "DESIGN"
-------------------------
A design is the study-design fingerprint of one preclinical arm, built from these
9 fields :species / strain / disease model / sex / route / duration / group size / age / weight
    e.g.  "mice/C57BL6/DIO/male/SC/28d/n=8/age=112d/w=44g"

THE RULE: dose_only = (n_distinct_designs == 1) and (n_candidate_arms >= 2)

i.e. the pool offers >=2 arms to choose between, but they are all the same design.
Those drugs are excluded from the 7-drug "distinct-design" analysis in the Results.

OUTPUTS
-------
  outputs/preclinical_design_summary.csv  — one row per drug (counts + dose_only)
  outputs/preclinical_designs_by_drug.csv — one row per DISTINCT design, so the
                                            design list itself can be inspected
"""
from pathlib import Path

import pandas as pd

from ranking_core import (build_base, build_eval_cells, make_design_label,
                          RAW_PATH, GROUP, NCT, EVAL_GROUP)

# Raw columns (build_base drops these, so we read them from the raw file)
ARM  = "preclinical_arm_id"     # identifies one preclinical study arm
PMC  = "pmcid"                  # identifies the preclinical publication
SPEC = "preclinical_animal_species"
DUR  = "preclinical_dosage_duration(days)"

OUT_DIR      = Path(__file__).parent.parent.parent / "outputs" / "ranking"
OUT_SUMMARY  = OUT_DIR / "preclinical_design_summary.csv"
OUT_DESIGNS  = OUT_DIR / "preclinical_designs_by_drug.csv"
SEL_CSV      = OUT_DIR / "selected_design.csv"     # written by preclinical_selection.py
HIGHLIGHT    = ["liraglutide", "semaglutide"]


def build_arm_table():
    """One row per distinct candidate preclinical ARM, with its design fingerprint.

    Why a dedupe is needed: the evaluation frame has one row per (preclinical arm x
    clinical arm) pair, so each preclinical arm repeats once per clinical arm it was
    compared against. A drug's candidate POOL is the set of distinct preclinical arms.
    """
    raw   = pd.read_csv(RAW_PATH)
    cells = build_eval_cells(build_base())            # the 9-drug ranking set

    # Fingerprint every row, then attach the raw identifiers by index.
    arms = pd.DataFrame({
        "drug":   cells[GROUP].values,
        "arm_id": raw.loc[cells.index, ARM].values,   # same index -> safe alignment
        "pmcid":  raw.loc[cells.index, PMC].values,
        "design": cells.apply(make_design_label, axis=1).values,
    })
    # collapse the pair-level rows down to the distinct arms in each drug's pool
    return arms.drop_duplicates(subset=["drug", "arm_id"]).reset_index(drop=True), raw, cells


def main():
    arms, raw, cells = build_arm_table()

    # ── Per drug: how many candidate arms, and how many DISTINCT designs? ──────
    rows = []
    for drug, g in arms.groupby("drug"):
        n_arms    = g["arm_id"].nunique()      # = the candidate pool size
        n_designs = g["design"].nunique()      # distinct designs among those arms
        rows.append({
            "drug":                 drug,
            "candidate_arms":       n_arms,
            "distinct_designs":     n_designs,
            "dose_only":            bool(n_designs == 1 and n_arms >= 2),
            "preclinical_studies":  g["pmcid"].nunique(),
        })
    summary = pd.DataFrame(rows)

    # clinical-side counts come from the pair-level frame
    clin = (cells.groupby(GROUP)
                 .agg(clinical_arms=(EVAL_GROUP, "nunique"),
                      clinical_studies=(NCT, "nunique"))
                 .reset_index().rename(columns={GROUP: "drug"}))
    summary = summary.merge(clin, on="drug").sort_values("candidate_arms")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_SUMMARY, index=False)
    print(summary.to_string(index=False))
    print(f"\nPools that differ ONLY in dose (distinct_designs == 1 and arms >= 2): "
          f"{list(summary[summary.dose_only]['drug'])}")

    # ── The design list itself, so the counts above can be inspected ───────────
    designs = (arms.groupby(["drug", "design"])
                   .agg(n_arms_sharing_design=("arm_id", "nunique"))
                   .reset_index()
                   .sort_values(["drug", "n_arms_sharing_design"], ascending=[True, False]))
    designs.to_csv(OUT_DESIGNS, index=False)

    # Print the designs for the small pools — this is the evidence for the rule.
    print("\n" + "=" * 78)
    print("  Distinct designs in the small candidate pools (the Results' key contrast)")
    print("=" * 78)
    for drug in ["tirzepatide", "medi0382", "canagliflozin", "survodutide"]:
        sub = designs[designs.drug == drug]
        flag = "DOSE-ONLY" if bool(summary.loc[summary.drug == drug, "dose_only"].iloc[0]) else "distinct designs"
        n_arms = int(summary.loc[summary.drug == drug, "candidate_arms"].iloc[0])
        print(f"\n  {drug}  ({n_arms} candidate arms -> {len(sub)} distinct design(s); {flag})")
        for _, d in sub.iterrows():
            print(f"     [{d['n_arms_sharing_design']} arm(s)] {d['design']}")

    # ── Highlighted drugs: species mix, duration spread, most-selected design ──
    ev = raw.loc[cells.index].copy()
    sel = pd.read_csv(SEL_CSV) if SEL_CSV.exists() else None
    for drug in HIGHLIGHT:
        g = ev[ev[GROUP] == drug].drop_duplicates(ARM)
        d = g[DUR]
        print(f"\n[{drug}] {g[ARM].nunique()} preclinical arms | "
              f"species={g[SPEC].str.lower().value_counts().to_dict()} | "
              f"duration median={d.median():.0f}d range={d.min():.0f}-{d.max():.0f}d")
        if sel is not None:
            top = (sel[sel.drug == drug]
                     .sort_values("n_cells_selected", ascending=False).iloc[0]["design_id"])
            pmc = sorted(arms[(arms.drug == drug) & (arms.design == top)]["pmcid"].unique())
            print(f"   most-selected design: {top}")
            print(f"   source study (pmcid): {pmc}")

    print(f"\nSaved: {OUT_SUMMARY}")
    print(f"Saved: {OUT_DESIGNS}")


if __name__ == "__main__":
    main()
