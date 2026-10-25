"""Statistics of the drug-mapping audit.

    python src/PCDB/audit/summarize_audit_outcomes.py
    python src/PCDB/audit/summarize_audit_outcomes.py \
        --annotations /path/to/annotations.csv --output-dir /path/to/out

Input is data/PCDB/drug_mapping_audit/annotations.csv and nothing else: every
figure below is computed from it, so the file is the whole provenance of the
reported shares.

This audit asks a different question from the relation-extraction validation
in data/PCDB/pcdb_relations/. Here a row is one sampled publication and the
question is which pipeline stage lost its drug; there a row is one asserted
relation and the question is whether an experiment supports it. The two share
no columns beyond pmcid and are not interchangeable.

Each row is one sampled publication. The columns used here are the final
annotation and two lists aligned with it, compound by compound:

  author       compounds given to the animals to prevent or treat disease,
               separated by ";", or one of: no drug | not in vivo: <reason> |
               uncertain | abstract only
  pcdb_match   for each compound, the drug name PCDB recorded for it, or "none"
  drug_types   for each compound, its type (small_molecule, antibody_biologic, ...)

Each publication is assigned the stage that lost its drug:

  extracted_not_normalised   every compound has a PCDB match (extracted, not normalised)
  partly_extracted           some compounds have a match, at least one does not
  not_extracted              no compound has a match
  out_of_scope               not an original in vivo animal study
  nothing_missed             no qualifying compound was administered

Publications marked `abstract only` are excluded. The sample is a simple random
draw, so each share is a binomial proportion with a Wilson 95% interval.
"""

import argparse
from pathlib import Path

import pandas as pd

Z = 1.96
NO_COMPOUND_VALUES = {"no drug", "uncertain", "abstract only"}
OUTCOME_ORDER = ["extracted_not_normalised", "partly_extracted", "not_extracted",
                 "out_of_scope", "nothing_missed", "annotator_uncertain"]
OUTCOME_LABELS = {
    "extracted_not_normalised": "Extracted, not normalised",
    "partly_extracted": "Partly extracted",
    "not_extracted": "Not extracted",
    "out_of_scope": "Not primary in vivo",
    "nothing_missed": "No drug administered",
    "annotator_uncertain": "Uncertain",
    "abstract_only": "Excluded (abstract only)",
}
DRUG_TYPE_LABELS = {
    "antibody_biologic": "antibodies and other biologics",
    "nucleic_acid": "nucleic acids",
    "peptide": "peptides",
    "formulation_nanoparticle": "nanoparticle formulations",
    "natural_product": "natural products",
    "cell_therapy": "cell therapies",
}
ADDITIONAL_FILE_COLUMNS = {
    "pmcid": "PMCID",
    "title": "Title",
    "pcdb_recorded": "Drug names recorded in PCDB",
    "opus5": "Claude Opus 5",
    "fable5.1": "Claude Fable 5.1",
    "author": "Final annotation (authors)",
    "pcdb_match": "PCDB match per compound",
    "drug_types": "Drug type per compound",
    "outcome": "Outcome",
    "author_note": "Note",
}


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    proportion = successes / total
    denominator = 1 + Z**2 / total
    center = (proportion + Z**2 / (2 * total)) / denominator
    margin = Z * ((proportion * (1 - proportion) / total
                   + Z**2 / (4 * total**2)) ** 0.5) / denominator
    return max(0.0, center - margin) * 100, min(1.0, center + margin) * 100


def split_list(cell: str) -> list[str]:
    return [item.strip() for item in cell.split(";") if item.strip()]


def compounds_of(annotation: str) -> list[str]:
    if annotation in NO_COMPOUND_VALUES or annotation.startswith("not in vivo"):
        return []
    return split_list(annotation)


def check_alignment(table: pd.DataFrame):
    """Every compound needs exactly one PCDB match and one drug type."""
    for _, row in table.iterrows():
        if not row["author"].strip():
            raise ValueError(f"{row['pmcid']}: empty final annotation")
        n_compounds = len(compounds_of(row["author"]))
        n_matches, n_types = len(split_list(row["pcdb_match"])), len(split_list(row["drug_types"]))
        if not n_compounds == n_matches == n_types:
            raise ValueError(f"{row['pmcid']}: {n_compounds} compounds, {n_matches} PCDB matches, "
                             f"{n_types} drug types")


def classify(row) -> str:
    annotation = row["author"]
    if annotation == "abstract only":
        return "abstract_only"
    if annotation.startswith("not in vivo"):
        return "out_of_scope"
    if annotation == "uncertain":
        return "annotator_uncertain"
    if annotation == "no drug":
        return "nothing_missed"
    matched = [match != "none" for match in split_list(row["pcdb_match"])]
    if all(matched):
        return "extracted_not_normalised"
    if not any(matched):
        return "not_extracted"
    return "partly_extracted"


def summarize_outcomes(outcomes: pd.Series) -> pd.DataFrame:
    counts = outcomes.value_counts()
    rows = []
    for outcome in [o for o in OUTCOME_ORDER if o in counts]:
        low, high = wilson_interval(int(counts[outcome]), len(outcomes))
        rows.append({"outcome": outcome, "n": int(counts[outcome]),
                     "pct": counts[outcome] / len(outcomes) * 100, "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


def summarize_drug_types(audited: pd.DataFrame) -> pd.DataFrame:
    drug_types = pd.Series([drug_type for cell in audited["drug_types"]
                            for drug_type in split_list(cell)])
    counts = drug_types.value_counts()
    return pd.DataFrame({"drug_type": counts.index, "n": counts.values,
                         "pct": counts.values / len(drug_types) * 100})


def describe_outcomes(outcome_summary: pd.DataFrame, n_audited: int) -> str:
    row = outcome_summary.set_index("outcome")
    pct = row["pct"].reindex(OUTCOME_ORDER).fillna(0.0)
    low, high = row.loc["extracted_not_normalised", ["ci_low", "ci_high"]]
    return (f"Of the {n_audited} publications, {pct['extracted_not_normalised']:.1f}% "
            f"(95% CI {low:.1f}-{high:.1f}) had every administered drug extracted by the entity "
            f"extraction module but not normalised to a curated entity, "
            f"{pct['partly_extracted']:.1f}% were partly extracted (at least one administered drug "
            f"missed), and {pct['not_extracted']:.1f}% had no administered drug extracted. "
            f"A further {pct['out_of_scope']:.1f}% were not primary in vivo studies and "
            f"{pct['nothing_missed']:.1f}% administered no drug.")


def describe_drug_types(drug_type_summary: pd.DataFrame) -> str:
    pct = drug_type_summary.set_index("drug_type")["pct"]
    parts = [f"{label} ({pct.get(drug_type, 0.0):.1f}%)"
             for drug_type, label in DRUG_TYPE_LABELS.items()]
    return (f"Of the {drug_type_summary['n'].sum()} drug compounds identified in the audited "
            f"publications, {100 - pct.get('small_molecule', 0.0):.1f}% were not small molecules: "
            f"{', '.join(parts[:-1])} and {parts[-1]}, and others.")


def build_additional_file(table: pd.DataFrame, outcomes: pd.Series) -> pd.DataFrame:
    additional_file = table.assign(outcome=outcomes.map(OUTCOME_LABELS).values)
    return additional_file[list(ADDITIONAL_FILE_COLUMNS)].rename(columns=ADDITIONAL_FILE_COLUMNS)


def main(annotation_path: Path, output_dir: Path) -> int:
    if not annotation_path.exists():
        raise SystemExit(f"error: {annotation_path} not found; pass "
                         f"--annotations to point at it")
    table = pd.read_csv(annotation_path, dtype=str, keep_default_na=False)
    check_alignment(table)
    outcomes = table.apply(classify, axis=1)
    audited = table[outcomes != "abstract_only"]
    outcome_summary = summarize_outcomes(outcomes[outcomes != "abstract_only"])
    drug_type_summary = summarize_drug_types(audited)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    outcome_summary.to_csv(output_path / "audit_summary.csv", index=False)
    drug_type_summary.to_csv(output_path / "drug_type_summary.csv", index=False)
    # utf-8-sig so Excel shows characters such as β and ′ correctly
    build_additional_file(table, outcomes).to_csv(
        output_path / "additional_file_annotations.csv", index=False, encoding="utf-8-sig")

    excluded = table.loc[outcomes == "abstract_only", "pmcid"]
    print(f"sampled {len(table)}; excluded {len(excluded)} as abstract-only "
          f"({', '.join(excluded)}); audited {len(audited)}")
    print("\n=== outcome (95% Wilson CI) ===")
    print(outcome_summary.round(1).to_string(index=False))
    print(f"\n=== drug types of the {drug_type_summary['n'].sum()} administered compounds ===")
    print(drug_type_summary.round(1).to_string(index=False))
    print("\n" + describe_outcomes(outcome_summary, len(audited)))
    print("\n" + describe_drug_types(drug_type_summary))
    return 0


# Resolved from this file rather than the working directory: string defaults
# relative to the CWD meant the script only ran from the repository root.
ROOT = Path(__file__).resolve().parents[3]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path,
                        default=ROOT / "data/PCDB/drug_mapping_audit/annotations.csv",
                        help="one row per sampled publication")
    parser.add_argument("--output-dir", "--output_dir", type=Path,
                        dest="output_dir",
                        default=ROOT / "outputs/pcdb/drug_mapping_audit")
    args = parser.parse_args()

    raise SystemExit(main(args.annotations, args.output_dir))
