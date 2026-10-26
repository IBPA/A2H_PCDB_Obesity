#!/usr/bin/env python3
"""Statistics derived from the relation-extraction results. None transcribed.

Run it to print the whole-database numbers, the per-relation-type table in
both frames, and the Results paragraph with its own statistics substituted. The
paragraph is generated rather than written down because a transcribed number
outlives the data it came from: nothing recomputes it, so a database rebuild
leaves it quietly wrong.

Two entry points are also imported elsewhere:

  corpus_statistics()   the whole-database numbers, including the
                        entity-resolved frame that Table 1 divides by
  per_disease_triples() asserted and experimentally confirmed triples per
                        disease, which cross_disease_summary.py reports as a
                        column

A disease-drug-animal triple is "experimentally confirmed" when the model
found that all three took part in the same in vivo experiment in the paper.
Rejection does not mean the paper is wrong - it means PCDB's row is a
co-mention rather than an experiment.

Two inputs, and only two: the database rows and the extraction verdicts. The
relations file holds exactly what PCDB asserts, so no statistic here depends on
how the audit resolved entities.

    python relation_statistics.py
    python relation_statistics.py \
        --pcdb-complete /path/to/pcdb_complete.tsv \
        --relations /path/to/pcdb_relations_all.jsonl
"""

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RELEASE = ROOT / "data/PCDB/preclinical-db-v1.1"
TRIPLE = "disease-drug-animal"


def main() -> int:
    args = parse_arguments()
    inputs = [require(args.pcdb_complete), require(args.relations)]
    statistics = {
        "inputs": {str(p): sha256(p) for p in inputs},
        **corpus_statistics(args.pcdb_complete, args.relations),
    }
    report(statistics)
    print_paper_sentence(statistics)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(statistics, indent=2) + "\n", encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return check(statistics, args.check) if args.check else 0


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcdb-complete", type=Path,
                        default=DEFAULT_RELEASE / "pcdb_complete.tsv",
                        help="PCDB rows: one per (publication x animal group)")
    parser.add_argument("--relations", type=Path,
                        default=DEFAULT_RELEASE / "pcdb_relations_all.jsonl",
                        help="extraction verdicts, one record per publication")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "outputs/relations/relation_statistics.json")
    parser.add_argument("--check", type=Path,
                        help="an earlier statistics file; exit non-zero "
                             "if any statistic moved")
    return parser.parse_args()


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def parse_list(value) -> list:
    if not isinstance(value, str) or not value.strip() or value == "[]":
        return []
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def entity_resolved(pcdb_complete: Path) -> tuple[set[str], dict]:
    """Publications with a disease, a drug and an animal all resolved to PCDB
    identifiers - the frame Table 1 reports rates against."""
    frame = pd.read_csv(require(pcdb_complete), sep="\t", dtype=str,
                        usecols=["pmcid", "disease_PCDB_id", "drug_PCDB_id",
                                 "animal_PCDB_id"])
    for kind in ("disease", "drug", "animal"):
        frame[kind] = frame[f"{kind}_PCDB_id"].map(parse_list).map(bool)
    per_pub = frame.groupby("pmcid")[["disease", "drug", "animal"]].any()
    resolved = set(per_pub[per_pub.disease & per_pub.drug & per_pub.animal].index)
    counts = {f"publications_with_{k}_id": int(per_pub[k].sum())
              for k in ("disease", "drug", "animal")}
    return resolved, {"publications_total": len(per_pub), **counts}


def read_relations(relations: Path):
    """Yield (pmcid, relation) for every relation in the file."""
    with require(relations).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            for relation in record["relations"]:
                yield record["pmcid"], relation


def corpus_statistics(pcdb_complete: Path, relations: Path) -> dict:
    """Counts in both frames, because the reported statistics use different
    ones and neither is a subset of the claim the other makes.

    A disease-drug-animal triple needs all three entities, so every triple
    already sits in a publication PCDB resolves on all three axes - the triple
    statistics are identical in either frame. A two-entity pair does not: a
    publication with a disease and an animal but no resolvable drug asserts
    disease-animal pairs and never reaches the three-entity subset. Reporting
    those pairs over the entity-resolved frame alone would silently drop pairs
    PCDB legitimately asserts, so they are also counted across all
    publications.

    Relations belonging to publications outside the three-entity subset are
    reported as out_of_frame. That count is expected, not a defect: it is
    simply the relations that cannot be triples, dominated by the 37.6% of
    publications with no resolvable drug."""
    resolved, universe = entity_resolved(pcdb_complete)
    asserted = confirmed = labelled = total = 0
    out_of_frame = 0
    out_of_frame_pubs = set()
    with_triple, with_confirmed = set(), set()
    by_type, by_type_all = {}, {}

    for pmcid, relation in read_relations(relations):
        total += 1
        labelled += bool(relation["label"])
        # Two frames, because the reported statistics use different ones: a
        # triple requires all three entities so it is inherently inside the
        # entity-resolved subset, while a two-entity pair exists in
        # publications that never resolve the third - restricting those would
        # discard pairs PCDB legitimately asserts.
        bucket_all = by_type_all.setdefault(relation["relation_type"],
                                            {"asserted": 0, "confirmed": 0})
        bucket_all["asserted"] += 1
        bucket_all["confirmed"] += relation["label"] == "yes"
        if pmcid not in resolved:
            out_of_frame += 1
            out_of_frame_pubs.add(pmcid)
            continue
        bucket = by_type.setdefault(relation["relation_type"],
                                    {"asserted": 0, "confirmed": 0})
        bucket["asserted"] += 1
        bucket["confirmed"] += relation["label"] == "yes"
        if relation["relation_type"] != TRIPLE:
            continue
        asserted += 1
        with_triple.add(pmcid)
        if relation["label"] == "yes":
            confirmed += 1
            with_confirmed.add(pmcid)

    for table in (by_type, by_type_all):
        for bucket in table.values():
            bucket["pct_confirmed"] = pct(bucket["confirmed"], bucket["asserted"])

    return {
        **universe,
        "publications_entity_resolved": len(resolved),
        "pct_entity_resolved": pct(len(resolved), universe["publications_total"]),
        "triples_asserted": asserted,
        "triples_confirmed": confirmed,
        "pct_triples_confirmed": pct(confirmed, asserted),
        "publications_asserting_triple": len(with_triple),
        "publications_with_confirmed_triple": len(with_confirmed),
        "pct_entity_resolved_with_confirmed_triple": pct(len(with_confirmed), len(resolved)),
        "publications_all_triples_rejected": len(with_triple - with_confirmed),
        "by_relation_type_entity_resolved": by_type,
        "by_relation_type_all_publications": by_type_all,
        "relations_asserted": total,
        "relations_labelled": labelled,
        "relations_unlabelled": total - labelled,
        "relations_outside_entity_resolved_frame": out_of_frame,
        "publications_outside_entity_resolved_frame": len(out_of_frame_pubs),
    }


def per_disease_triples(pcdb_complete: Path, relations: Path) -> dict[str, dict]:
    """Per disease PCDB id: triples asserted, confirmed, and the publications
    behind them. Restricted to the entity-resolved frame, so these percentages
    compose with the corpus statistics. Keyed on disease_id because PCDB ids are
    positional and disease names are not unique across a rebuild."""
    resolved, _ = entity_resolved(pcdb_complete)
    stats = {}
    for pmcid, relation in read_relations(relations):
        if pmcid not in resolved:
            continue
        if relation["relation_type"] != TRIPLE or not relation["disease_id"]:
            continue
        entry = stats.setdefault(relation["disease_id"],
                                 {"asserted": 0, "confirmed": 0,
                                  "pubs": set(), "pubs_confirmed": set()})
        entry["asserted"] += 1
        entry["pubs"].add(pmcid)
        if relation["label"] == "yes":
            entry["confirmed"] += 1
            entry["pubs_confirmed"].add(pmcid)

    return {disease_id: {
        "triples_asserted": e["asserted"],
        "triples_confirmed": e["confirmed"],
        "pct_triples_confirmed": pct(e["confirmed"], e["asserted"]),
        "publications": len(e["pubs"]),
        "publications_with_confirmed_triple": len(e["pubs_confirmed"]),
        "pct_publications_with_confirmed_triple": pct(len(e["pubs_confirmed"]),
                                                      len(e["pubs"])),
    } for disease_id, e in stats.items()}


def pct(numerator: int, denominator: int) -> float | None:
    return round(100 * numerator / denominator, 1) if denominator else None


def report(statistics: dict) -> None:
    print("Relation-extraction statistics - all derived, none transcribed\n")
    print(f"  publications in PCDB                  {statistics['publications_total']:>9,}")
    print(f"  entity-resolved (all three ids)       "
          f"{statistics['publications_entity_resolved']:>9,}"
          f"   {statistics['pct_entity_resolved']}%")
    for kind, article in (("disease", "a"), ("drug", "a"), ("animal", "an")):
        label = f"with {article} {kind} id"
        print(f"    {label:<34s} {statistics[f'publications_with_{kind}_id']:>9,}")
    print(f"\n  disease-drug-animal triples")
    print(f"    asserted                            {statistics['triples_asserted']:>9,}")
    print(f"    experimentally confirmed            {statistics['triples_confirmed']:>9,}"
          f"   {statistics['pct_triples_confirmed']}%")
    print(f"    publications with >=1 confirmed     "
          f"{statistics['publications_with_confirmed_triple']:>9,}"
          f"   {statistics['pct_entity_resolved_with_confirmed_triple']}% of the "
          f"{statistics['publications_asserting_triple']:,} asserting one")
    print(f"    publications all triples rejected   "
          f"{statistics['publications_all_triples_rejected']:>9,}")
    # Both frames, side by side and labelled: a triple needs all three
    # entities so it sits inside the entity-resolved subset either way, but a
    # two-entity pair also exists in publications that never resolve the
    # third. Printing one frame alone invites the reader to think the other
    # number is a mistake.
    print("\n  by relation type        "
          "--- entity-resolved ---    ----- all of PCDB -----")
    resolved = statistics["by_relation_type_entity_resolved"]
    every = statistics["by_relation_type_all_publications"]
    for kind in sorted(every):
        a, b = resolved.get(kind, {"confirmed": 0, "asserted": 0,
                                   "pct_confirmed": None}), every[kind]
        print(f"    {kind:22s} {a['confirmed']:>9,}/{a['asserted']:<9,}"
              f"{str(a['pct_confirmed']) + '%':>7s}"
              f"   {b['confirmed']:>9,}/{b['asserted']:<9,}"
              f"{str(b['pct_confirmed']) + '%':>7s}")
    print(f"\n  relations asserted (all publications)  {statistics['relations_asserted']:>9,}")
    print(f"  relations labelled                    {statistics['relations_labelled']:>9,}")
    print(f"  relations unlabelled                  {statistics['relations_unlabelled']:>9,}")
    print(f"  outside the entity-resolved frame     "
          f"{statistics['relations_outside_entity_resolved_frame']:>9,}"
          f"   in {statistics['publications_outside_entity_resolved_frame']:,} publications")
    print("\n  inputs:")
    for path, digest in statistics["inputs"].items():
        print(f"    {digest}  {path}")


def print_paper_sentence(statistics: dict) -> None:
    """Emit the Results sentence with its own numbers substituted.

    Every statistic in it is derived here on each run, so the paragraph cannot
    drift from the data the way a transcribed number does.
    """
    resolved = statistics["by_relation_type_entity_resolved"]
    every = statistics["by_relation_type_all_publications"]

    def clause(bucket, label):
        return (f"{bucket['pct_confirmed']}% (n = {bucket['confirmed']:,} out "
                f"of {bucket['asserted']:,}) of {label}")

    triple = resolved["disease-drug-animal"]
    pooled_confirmed = sum(b["confirmed"] for b in every.values())
    pooled_asserted = sum(b["asserted"] for b in every.values())

    print("\n=== RESULTS PARAGRAPH (derived, not transcribed) ===\n")
    print(f"Applied to the three-entity-resolved subset "
          f"({statistics['publications_entity_resolved']:,} publications), "
          f"{clause(triple, 'publication-level disease-drug-animal triples')} "
          f"were supported by an experiment-level relation. Across PCDB, "
          + clause(every["disease-drug"], "disease-drug pairs") + ", "
          + clause(every["disease-animal"], "disease-animal pairs") + " and "
          + clause(every["drug-animal"], "drug-animal pairs")
          + f" were similarly supported, as were "
            f"{pct(pooled_confirmed, pooled_asserted)}% (n = "
            f"{pooled_confirmed:,} out of {pooled_asserted:,}) of all four "
            f"relation types pooled; "
            f"{statistics['pct_entity_resolved_with_confirmed_triple']}% (n = "
            f"{statistics['publications_with_confirmed_triple']:,} out of "
            f"{statistics['publications_asserting_triple']:,}) of these "
            f"publications contained at least one supported triple.")

    print("\n=== INPUTS THAT MUST BE FIXED FOR THESE NUMBERS ===\n")
    for path, sha in statistics["inputs"].items():
        print(f"  sha256 {sha}  {path}")
    print("\n  Re-run with --check <earlier statistics json> to fail if any")
    print("  statistic moves. A changed digest explains why.")


def check(statistics: dict, earlier: Path) -> int:
    """Non-zero exit if any statistic moved, so a rebuild cannot silently change a
    published number."""
    was = json.loads(earlier.read_text(encoding="utf-8"))
    moved = [(k, was.get(k), statistics.get(k)) for k in statistics
             if k != "inputs" and was.get(k) != statistics.get(k)]
    if not moved:
        print(f"\n  all statistics unchanged against {earlier}")
        return 0
    print(f"\n  STATISTICS MOVED against {earlier}:", file=sys.stderr)
    for key, before, after in moved:
        print(f"    {key}: {before} -> {after}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
