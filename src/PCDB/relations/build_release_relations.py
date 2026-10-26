#!/usr/bin/env python3
"""Build the release's relation files from the predictions.

    data/PCDB/pcdb_relations/all_predictions.jsonl   every model verdict
    data/PCDB/pcdb_relations/publications.jsonl      every PCDB publication
    preclinical-db-v1.1/pcdb_complete.tsv            which relations are in scope
        -> preclinical-db-v1.1/pcdb_relations_all.jsonl
        -> preclinical-db-v1.1/pcdb_relations_all.csv

Three inputs, because none alone is enough. The predictions carry a verdict per
relation but no notion of which are in scope: enumeration resolved entities
through their external identifiers and recovered drugs PCDB extracted yet left
unmapped, so 40,532 verdicts describe relations PCDB does not assert. PCDB's
own columns decide scope. And the 10,194 publications asserting no relation have
no prediction at all, so the publication list has to come from elsewhere too.

Scope is the cross product of the entities PCDB records for a publication,
unioned over its rows - PCDB stores one row per (publication x animal group),
so reading any single row understates the animal axis.

    python build_release_relations.py
    python build_release_relations.py --pcdb-dir /data/PCDB --release preclinical-db-v1.2

--pcdb-dir is the directory holding the release and the predictions; --release
names the release inside it, or is an absolute path elsewhere. Every individual
path can still be overridden, so inputs and outputs need not live together.
"""

import argparse
import ast
import csv
import hashlib
import json
import sys
from collections import Counter
from itertools import product
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RELATION_FIELDS = ("row_id", "relation_type", "disease", "drug", "animal",
                   "disease_id", "drug_id", "animal_id")
VERDICTS = {"yes", "no", "unclear"}


def main() -> int:
    args = resolve_paths(parse_arguments())
    scope = pcdb_scope(args.pcdb_complete)
    publications = read_publications(args.publications)
    predictions = read_predictions(args.predictions)
    records = assemble(publications, predictions, scope)
    verify(records, scope)
    write_jsonl(records, args.out_jsonl)
    write_csv(records, args.out_csv)
    report(records, predictions, args)
    return 0


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcdb-dir", type=Path, default=ROOT / "data/PCDB",
                        help="directory holding the release and pcdb_relations/")
    parser.add_argument("--release", type=Path, default=Path("preclinical-db-v1.1"),
                        help="release directory, relative to --pcdb-dir or absolute")
    parser.add_argument("--relations-dir", type=Path,
                        help="where the prediction files live "
                             "(default: <pcdb-dir>/pcdb_relations)")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--publications", type=Path)
    parser.add_argument("--pcdb-complete", type=Path)
    parser.add_argument("--out-jsonl", type=Path)
    parser.add_argument("--out-csv", type=Path)
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    """Fill the paths the caller left unset from --pcdb-dir and --release.

    Derived after parsing rather than as argparse defaults, because a default
    computed at definition time cannot depend on another flag's value.
    """
    release = args.release if args.release.is_absolute() else args.pcdb_dir / args.release
    relations = args.relations_dir or args.pcdb_dir / "pcdb_relations"

    defaults = {
        "predictions": relations / "all_predictions.jsonl",
        "publications": relations / "publications.jsonl",
        "pcdb_complete": release / "pcdb_complete.tsv",
        "out_jsonl": release / "pcdb_relations_all.jsonl",
        "out_csv": release / "pcdb_relations_all.csv",
    }
    for name, path in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, path)

    for name in ("predictions", "publications", "pcdb_complete"):
        path = getattr(args, name)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found; set --{name.replace('_', '-')} or point "
                f"--pcdb-dir at a directory containing it")
    return args


def parse_list(cell) -> list:
    if not isinstance(cell, str) or not cell.strip() or cell == "[]":
        return []
    try:
        parsed = ast.literal_eval(cell)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def pcdb_scope(pcdb_complete: Path) -> dict[str, set[tuple]]:
    frame = pd.read_csv(pcdb_complete, sep="\t", dtype=str,
                        usecols=["pmcid", "disease_PCDB_id", "drug_PCDB_id",
                                 "animal_PCDB_id"])
    entities = {}
    for row in frame.itertuples():
        entry = entities.setdefault(row.pmcid, {"disease": set(), "drug": set(),
                                                "animal": set()})
        for kind in ("disease", "drug", "animal"):
            entry[kind].update(parse_list(getattr(row, f"{kind}_PCDB_id")))

    scope = {}
    for pmcid, entry in entities.items():
        diseases, drugs, animals = entry["disease"], entry["drug"], entry["animal"]
        keys = set()
        for d, r in product(diseases, drugs):
            keys.add(("disease-drug", d, r, ""))
        for d, a in product(diseases, animals):
            keys.add(("disease-animal", d, "", a))
        for r, a in product(drugs, animals):
            keys.add(("drug-animal", "", r, a))
        for d, r, a in product(diseases, drugs, animals):
            keys.add(("disease-drug-animal", d, r, a))
        scope[pmcid] = keys
    return scope


def read_publications(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def read_predictions(path: Path) -> dict[str, list[dict]]:
    by_publication = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            by_publication.setdefault(record["pmcid"], []).append(record)
    return by_publication


def key_of(record: dict) -> tuple:
    return (record["relation_type"], record["disease_id"],
            record["drug_id"], record["animal_id"])


def assemble(publications: list[dict], predictions: dict,
             scope: dict) -> list[dict]:
    records = []
    for publication in publications:
        pmcid = publication["pmcid"]
        allowed = scope.get(pmcid, set())
        relations = sorted(
            ({k: p[k] for k in RELATION_FIELDS} | {"label": p["label"],
                                                   "reason": p["reason"]}
             for p in predictions.get(pmcid, []) if key_of(p) in allowed),
            key=lambda r: int(r["row_id"]))
        records.append({
            "pmcid": pmcid,
            "link": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}",
            "input_status": publication["input_status"],
            "text_source": publication["text_source"],
            "n_relations": len(relations),
            "n_labelled": sum(1 for r in relations if r["label"]),
            "relations": relations,
        })
    return records


def verify(records: list[dict], scope: dict) -> None:
    """The result must equal PCDB's cross product exactly, every relation
    labelled, no row_id twice."""
    if len(records) != len(scope):
        raise ValueError(f"{len(records):,} publications built, PCDB has "
                         f"{len(scope):,}")

    missing = surplus = 0
    for record in records:
        allowed = scope.get(record["pmcid"], set())
        present = {key_of(r) for r in record["relations"]}
        missing += len(allowed - present)
        surplus += len(present - allowed)
    if missing or surplus:
        raise ValueError(f"scope not exact: {missing:,} missing, {surplus:,} surplus")

    invalid = sum(1 for r in records for rel in r["relations"]
                  if rel["label"] not in VERDICTS)
    if invalid:
        raise ValueError(f"{invalid:,} relations without a valid label")

    row_ids = Counter(int(rel["row_id"]) for r in records for rel in r["relations"])
    duplicates = [rid for rid, n in row_ids.items() if n > 1]
    if duplicates:
        raise ValueError(f"{len(duplicates)} duplicate row_ids")


def write_jsonl(records: list[dict], out: Path) -> None:
    temporary = out.with_suffix(out.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as sink:
        for record in records:
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(out)


def write_csv(records: list[dict], out: Path) -> None:
    columns = ["pmcid", "link", "input_status", *RELATION_FIELDS, "label", "reason"]
    temporary = out.with_suffix(out.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as sink:
        writer = csv.DictWriter(sink, fieldnames=columns)
        writer.writeheader()
        for record in records:
            head = {k: record[k] for k in ("pmcid", "link", "input_status")}
            for relation in record["relations"]:
                writer.writerow({**head, **relation})
    temporary.replace(out)


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def report(records, predictions, args) -> None:
    labels = Counter(rel["label"] for r in records for rel in r["relations"])
    total = sum(labels.values())
    predicted = sum(len(v) for v in predictions.values())
    print(f"  predictions read       {predicted:>9,}")
    print(f"  out of PCDB's scope    {predicted - total:>9,}  (dropped)")
    print(f"\n  publications           {len(records):>9,}")
    print(f"  relations              {total:>9,}")
    for label in ("yes", "no", "unclear"):
        print(f"    {label:18s} {labels[label]:>9,}  ({100*labels[label]/total:5.2f}%)")
    print(f"\n  {args.out_jsonl.name}  sha256 {digest(args.out_jsonl)[:32]}")
    print(f"  {args.out_csv.name}    sha256 {digest(args.out_csv)[:32]}")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
