#!/usr/bin/env python3
"""Check the release contains every relation PCDB's entities imply.

A relation set can be wrong in two directions and both have happened here:

  missing - the animal axis was truncated to each publication's first row, so
            27,476 relations were never enumerated. Nothing failed; the
            prompts were valid, only incomplete, and the gap stayed invisible
            until the counts were questioned. This is what the test exists to
            catch.
  surplus - the audit resolves entities through their external identifiers,
            which recovers drugs PCDB extracted but left unmapped in its own
            *_PCDB_id column. Those relations are real but PCDB does not
            assert them, so they are reported rather than treated as faults.

Expected relations are the cross product of the entities PCDB records for a
publication, unioned over all of its rows:

    |D|x|R| disease-drug + |D|x|A| disease-animal
  + |R|x|A| drug-animal  + |D|x|R|x|A| disease-drug-animal

Exits non-zero if anything PCDB asserts is missing.

    python verify_completeness.py --release ../../../data/PCDB/preclinical-db-v1.1
"""

import argparse
import ast
import json
import sys
from collections import Counter
from itertools import product
from pathlib import Path

import pandas as pd


def main() -> int:
    args = parse_arguments()
    expected = expected_relations(args.release)
    actual = actual_relations(args.release)
    missing, surplus = compare(expected, actual)
    report(expected, actual, missing, surplus)
    return 1 if missing else 0


def parse_arguments() -> argparse.Namespace:
    here = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path,
                        default=here / "data/PCDB/preclinical-db-v1.1")
    parser.add_argument("--show", type=int, default=5)
    return parser.parse_args()


def parse_list(cell) -> list:
    if not isinstance(cell, str) or not cell.strip() or cell == "[]":
        return []
    try:
        parsed = ast.literal_eval(cell)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def pcdb_entities(release: Path) -> dict[str, dict[str, set]]:
    """Every entity PCDB records for a publication, unioned over its rows.

    The union is the whole point: PCDB stores one row per (publication x animal
    group), so reading any single row understates the animal axis.
    """
    frame = pd.read_csv(release / "pcdb_complete.tsv", sep="\t", dtype=str,
                        usecols=["pmcid", "disease_PCDB_id", "drug_PCDB_id",
                                 "animal_PCDB_id"])
    entities = {}
    for row in frame.itertuples():
        entry = entities.setdefault(row.pmcid,
                                    {"disease": set(), "drug": set(), "animal": set()})
        for kind in ("disease", "drug", "animal"):
            entry[kind].update(parse_list(getattr(row, f"{kind}_PCDB_id")))
    return entities


def expected_relations(release: Path) -> set[tuple]:
    expected = set()
    for pmcid, entry in pcdb_entities(release).items():
        diseases, drugs, animals = entry["disease"], entry["drug"], entry["animal"]
        for d, r in product(diseases, drugs):
            expected.add((pmcid, "disease-drug", d, r, ""))
        for d, a in product(diseases, animals):
            expected.add((pmcid, "disease-animal", d, "", a))
        for r, a in product(drugs, animals):
            expected.add((pmcid, "drug-animal", "", r, a))
        for d, r, a in product(diseases, drugs, animals):
            expected.add((pmcid, "disease-drug-animal", d, r, a))
    return expected


def actual_relations(release: Path) -> set[tuple]:
    actual = set()
    with (release / "pcdb_relations_all.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            for relation in record["relations"]:
                actual.add((record["pmcid"], relation["relation_type"],
                            relation["disease_id"], relation["drug_id"],
                            relation["animal_id"]))
    return actual


def compare(expected: set, actual: set) -> tuple[set, set]:
    return expected - actual, actual - expected


def report(expected, actual, missing, surplus) -> None:
    print("Release completeness against PCDB's own entity columns\n")
    print(f"  relations PCDB implies      {len(expected):>9,}")
    print(f"  relations in the release     {len(actual):>9,}")
    print(f"  present in both              {len(expected & actual):>9,}")
    print(f"\n  MISSING (PCDB asserts, release lacks)  {len(missing):>9,}")
    if missing:
        by_type = Counter(m[1] for m in missing)
        for kind, count in by_type.most_common():
            print(f"    {kind:22s} {count:>8,}")
        print(f"    affected publications: {len({m[0] for m in missing}):,}")
        for m in sorted(missing)[:5]:
            print(f"      {m}")
    print(f"\n  surplus (release has, PCDB does not assert) {len(surplus):>9,}")
    if surplus:
        by_type = Counter(s[1] for s in surplus)
        for kind, count in by_type.most_common():
            print(f"    {kind:22s} {count:>8,}")
        print(f"    affected publications: {len({s[0] for s in surplus}):,}")
        print("    expected: the audit resolves drugs PCDB left unmapped")

    print(f"\n  {'FAIL - the cross product is incomplete' if missing else 'PASS - nothing PCDB asserts is missing'}")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
