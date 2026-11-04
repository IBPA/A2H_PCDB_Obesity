#!/usr/bin/env python3
"""Build the release's row tables from the labelled relations.

    outputs/pcdb/pcdb_complete.tsv          entity-linking output: one row per
                                            (publication x animal group), with
                                            the publication metadata
    outputs/relations/pcdb_relations_all.jsonl
                                            every asserted relation, labelled
        -> <release>/pcdb_complete.tsv      one row per asserted relation, with
                                            its label
        -> <release>/pcdb.tsv               one row per disease-drug-animal
                                            triple an experiment supports
        -> <release>/publication_extractions.tsv
                                            one row per publication: the
                                            extracted names, external ids and
                                            PCDB ids, resolved or not
        -> <release>/preclinical_database.xlsx

The entity-linking table is the stage input, not a release file: it carries
animal strain and group size, which the release does not publish. Its
publication metadata and extraction columns are carried into the release
collapsed to one row per publication, the animal groups unioned.

    python build_release_tables.py --entity-table /path/to/pcdb_complete.tsv
    python build_release_tables.py --pcdb-dir /data/PCDB --release preclinical-db-v2.0
"""

import argparse
import ast
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
METADATA_COLUMNS = ["pmcid", "date", "link", "confidence_score", "TITLE"]
ENTITY_COLUMNS = ["disease_name", "drug_name", "animal_name",
                  "disease_PCDB_id", "drug_PCDB_id", "animal_PCDB_id"]
# The model also gives a reason code for every `no`; the release does not
# carry it, since the label alone is what the tables are for.
COMPLETE_COLUMNS = METADATA_COLUMNS + ["relation_type"] + ENTITY_COLUMNS + ["label"]
PCDB_COLUMNS = METADATA_COLUMNS + ENTITY_COLUMNS
RELATION_TO_COLUMN = {"disease": "disease_name", "drug": "drug_name",
                      "animal": "animal_name", "disease_id": "disease_PCDB_id",
                      "drug_id": "drug_PCDB_id", "animal_id": "animal_PCDB_id"}
TRIPLE = "disease-drug-animal"
ENTITY_SHEETS = ("disease", "drug", "animal")
# Extraction columns that describe the publication, so every animal-group row
# repeats them; carried over under their original names.
PUBLICATION_LEVEL = ["disease_names", "disease_external_ids", "disease_PCDB_id",
                     "unresolved_disease_name", "unresolved_disease_external_ids",
                     "drug_names", "drug_external_ids", "drug_PCDB_id",
                     "unresolved_drug_names", "unresolved_drug_external_ids"]
# Per animal-group columns, unioned over the publication's groups; strain and
# group size are deliberately not among them.
ANIMAL_GROUP_LEVEL = {"animal_species": "animal_names",
                      "animal_external_ids": "animal_external_ids",
                      "animal_PCDB_id": "animal_PCDB_id"}
EXTRACTION_COLUMNS = METADATA_COLUMNS + PUBLICATION_LEVEL + list(ANIMAL_GROUP_LEVEL.values())


def main() -> int:
    args = resolve_paths(parse_arguments())
    entity_table = read_entity_table(args.entity_table)
    metadata = publication_metadata(entity_table)
    records = read_release_relations(args.relations)
    complete = relation_rows(records, metadata)
    pcdb = confirmed_triples(complete)
    extractions = publication_extractions(entity_table)
    verify(complete, pcdb, records, extractions)
    write_table(complete, args.pcdb_complete)
    write_table(pcdb, args.pcdb)
    write_table(extractions, args.extractions)
    write_workbook(pcdb, complete, extractions, args)
    report(complete, pcdb, extractions)
    return 0


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcdb-dir", type=Path, default=ROOT / "data/PCDB",
                        help="directory holding the release")
    parser.add_argument("--release", type=Path, default=Path("preclinical-db-v2.0"),
                        help="release directory, relative to --pcdb-dir or absolute")
    parser.add_argument("--entity-table", type=Path,
                        default=ROOT / "outputs/pcdb/pcdb_complete.tsv",
                        help="entity-linking output, one row per publication x animal group")
    parser.add_argument("--relations", type=Path,
                        default=ROOT / "outputs/relations/pcdb_relations_all.jsonl",
                        help="the labelled relations build_release_relations.py wrote")
    parser.add_argument("--pcdb-complete", type=Path)
    parser.add_argument("--pcdb", type=Path)
    parser.add_argument("--extractions", type=Path)
    parser.add_argument("--workbook", type=Path)
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    release = args.release if args.release.is_absolute() else args.pcdb_dir / args.release
    defaults = {
        "pcdb_complete": release / "pcdb_complete.tsv",
        "pcdb": release / "pcdb.tsv",
        "extractions": release / "publication_extractions.tsv",
        "workbook": release / "preclinical_database.xlsx",
    }
    for name, path in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, path)
    args.release_dir = release
    for name in ("entity_table", "relations"):
        if not getattr(args, name).exists():
            raise FileNotFoundError(f"{getattr(args, name)} not found; "
                                    f"set --{name.replace('_', '-')}")
    return args


# --- inputs ------------------------------------------------------------------

def read_entity_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def publication_metadata(entity_table: pd.DataFrame) -> pd.DataFrame:
    """One row per publication. Raises if the entity table gives a
    publication two dates or titles, since the release could then not say
    which one is right."""
    frame = entity_table[METADATA_COLUMNS].drop_duplicates()
    conflicting = frame[frame.duplicated("pmcid", keep=False)]
    if len(conflicting):
        raise ValueError(f"{conflicting.pmcid.nunique()} publications carry "
                         f"conflicting metadata, e.g. {conflicting.pmcid.iloc[0]}")
    return frame.set_index("pmcid")


def read_release_relations(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


# --- tables ------------------------------------------------------------------

def relation_rows(records: list[dict], metadata: pd.DataFrame) -> pd.DataFrame:
    """One row per asserted relation, in the relation file's order. Publications
    asserting no relation contribute no row."""
    missing = sorted({r["pmcid"] for r in records if r["relations"]} - set(metadata.index))
    if missing:
        raise ValueError(f"{len(missing)} publications with relations have no "
                         f"metadata in the entity table, e.g. {missing[0]}")
    rows = []
    for record in records:
        head = {"pmcid": record["pmcid"], **metadata.loc[record["pmcid"]].to_dict()} \
            if record["relations"] else None
        for relation in record["relations"]:
            rows.append({**head, "relation_type": relation["relation_type"],
                         **{column: relation[field]
                            for field, column in RELATION_TO_COLUMN.items()},
                         "label": relation["label"]})
    return pd.DataFrame(rows, columns=COMPLETE_COLUMNS)


def confirmed_triples(complete: pd.DataFrame) -> pd.DataFrame:
    supported = (complete["relation_type"] == TRIPLE) & (complete["label"] == "yes")
    return complete.loc[supported, PCDB_COLUMNS].reset_index(drop=True)


def publication_extractions(entity_table: pd.DataFrame) -> pd.DataFrame:
    """One row per publication, in first-seen order. Publication-level columns
    must agree across a publication's animal-group rows; the animal columns are
    unioned over the groups, duplicates dropped, first occurrence first."""
    varying = entity_table.groupby("pmcid")[PUBLICATION_LEVEL].nunique()
    conflicting = varying[(varying > 1).any(axis=1)]
    if len(conflicting):
        raise ValueError(f"{len(conflicting)} publications disagree across their "
                         f"rows on a publication-level column, e.g. "
                         f"{conflicting.index[0]}")

    rows = []
    for pmcid, group in entity_table.groupby("pmcid", sort=False):
        first = group.iloc[0]
        row = {column: first[column] for column in METADATA_COLUMNS + PUBLICATION_LEVEL}
        row["animal_names"] = literal_or_empty(unique_in_order(
            name.strip() for name in group["animal_species"] if name.strip()))
        row["animal_external_ids"] = literal_or_empty(unique_in_order(
            parse_literal(cell) for cell in group["animal_external_ids"] if cell))
        row["animal_PCDB_id"] = literal_or_empty(unique_in_order(
            item for cell in group["animal_PCDB_id"] for item in parse_literal(cell)))
        rows.append(row)
    return pd.DataFrame(rows, columns=EXTRACTION_COLUMNS)


def parse_literal(cell: str):
    return ast.literal_eval(cell) if cell else []


def unique_in_order(items) -> list:
    seen = []
    for item in items:
        if item not in seen:
            seen.append(item)
    return seen


def literal_or_empty(items: list) -> str:
    return str(items) if items else ""


# --- checks ------------------------------------------------------------------

def verify(complete: pd.DataFrame, pcdb: pd.DataFrame, records: list[dict],
           extractions: pd.DataFrame) -> None:
    if extractions["pmcid"].duplicated().any():
        raise ValueError("publication_extractions has a publication twice")
    if not set(r["pmcid"] for r in records) <= set(extractions["pmcid"]):
        raise ValueError("a publication in the relations file is missing from "
                         "publication_extractions")
    asserted = sum(r["n_relations"] for r in records)
    if len(complete) != asserted:
        raise ValueError(f"{len(complete):,} rows built for {asserted:,} relations")
    if complete["label"].isin(["yes", "no", "unclear"]).sum() != len(complete):
        raise ValueError("a relation row carries no valid label")
    confirmed = sum(1 for r in records for rel in r["relations"]
                    if rel["relation_type"] == TRIPLE and rel["label"] == "yes")
    if len(pcdb) != confirmed:
        raise ValueError(f"pcdb.tsv has {len(pcdb):,} rows for {confirmed:,} "
                         f"confirmed triples")
    if (pcdb[["disease_PCDB_id", "drug_PCDB_id", "animal_PCDB_id"]] == "").any().any():
        raise ValueError("a pcdb.tsv row lacks one of its three entity ids")


# --- output ------------------------------------------------------------------

def write_table(table: pd.DataFrame, out: Path) -> None:
    temporary = out.with_suffix(out.suffix + ".tmp")
    table.to_csv(temporary, sep="\t", index=False)
    temporary.replace(out)


def write_workbook(pcdb: pd.DataFrame, complete: pd.DataFrame,
                   extractions: pd.DataFrame, args: argparse.Namespace) -> None:
    """The spreadsheet view mirrors the TSVs: the three tables plus the catalogs."""
    temporary = args.workbook.with_suffix(".xlsx.tmp")
    with pd.ExcelWriter(temporary, engine="openpyxl") as writer:
        pcdb.to_excel(writer, sheet_name="pcdb", index=False)
        complete.to_excel(writer, sheet_name="pcdb_complete", index=False)
        extractions.to_excel(writer, sheet_name="publication_extractions", index=False)
        for kind in ENTITY_SHEETS:
            catalog = pd.read_csv(args.release_dir / f"{kind}_entities.tsv", sep="\t",
                                  dtype=str, keep_default_na=False)
            catalog.to_excel(writer, sheet_name=f"{kind}_entities", index=False)
    temporary.replace(args.workbook)


def report(complete: pd.DataFrame, pcdb: pd.DataFrame,
           extractions: pd.DataFrame) -> None:
    print(f"  pcdb_complete.tsv  {len(complete):>9,} relations, "
          f"{complete.pmcid.nunique():,} publications")
    for kind, count in complete.groupby("relation_type").size().items():
        confirmed = ((complete.relation_type == kind) & (complete.label == "yes")).sum()
        print(f"    {kind:22s} {count:>9,} asserted  {confirmed:>9,} confirmed")
    print(f"  pcdb.tsv           {len(pcdb):>9,} confirmed triples, "
          f"{pcdb.pmcid.nunique():,} publications")
    print(f"  publication_extractions.tsv  {len(extractions):>9,} publications")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
