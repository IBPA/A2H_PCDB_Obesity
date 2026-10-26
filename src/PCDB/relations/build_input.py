"""Build the annotation input for every PCDB publication.

One JSONL record per publication, all 134,565, so the output is a complete
image of the database rather than a filtered subset. `prompt` is empty where
no prompt can exist, and `status` says which case applies:

  ok              prompt built
  no_relations    fewer than two mapped entity types, so no relation exists
                  to audit - 11,017 publications
  no_text         relations exist but no article text could be retrieved
  oversized       more relations than MAX_RELATIONS; audit these separately
                  rather than sending a prompt that is mostly roster

Sharded because the whole set is ~4 GB: a single file that size invites silent
truncation in whatever reads it, and shards let annotation run in parallel.
Also writes an index CSV with every field except `prompt`, so the set can be
inspected and filtered without touching gigabytes.

    python build_input.py --pcdb-dir /path/to/preclinical-db-v1.1 \
        --text-dir /path/to/pmc_articles --outdir full_input/

Article text is supplied as directories of PMC files named by accession -
PMC1234567.xml (JATS) or PMC1234567.txt (BioC plain text) - searched
recursively. Pass --text-dir more than once to span several sources. A paper
with no file gets status no_text and an empty prompt rather than being
dropped, so the output stays a complete image of the database.
"""

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from itertools import product
from pathlib import Path

import pandas as pd

import pcdb_entities
import prompt as prompt_module

MAX_RELATIONS = 40          # above this the roster dwarfs the paper; max in PCDB is 175
SHARD_SIZE = 10_000
MIN_BODY_CHARS = 1_000      # below this the 'paper' is an abstract with noise
BODY_SECTION = re.compile(r"^METHODS AND RESULTS:(.*?)(?=\n\n[A-Z][A-Z /]+:|\Z)", re.S | re.M)


def main() -> int:
    args = parse_arguments()
    args.outdir.mkdir(parents=True, exist_ok=True)

    pcdb = pcdb_entities.load_pcdb(args.pcdb_dir)
    catalogs = pcdb_entities.load_catalogs(args.pcdb_dir)
    print("indexing PCDB...", flush=True)
    resolver = pcdb_entities.build_resolver(pcdb, catalogs)
    verify_animal_coverage(pcdb, resolver)
    names = {k: catalogs[k].set_index("PCDB_id")[f"{k}_name"] for k in catalogs}
    text_paths = locate_text(args.text_dir)
    pmcids = sorted(resolver["rows"])
    print(f"{len(pmcids):,} publications, text available for {len(text_paths):,}", flush=True)

    written = write_shards(pmcids, resolver, names, text_paths, pcdb, catalogs, args)
    report(written, args.outdir)
    return 0


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcdb-dir", type=Path, required=True,
                        help="a PCDB release directory (holds pcdb_complete.tsv)")
    parser.add_argument("--text-dir", type=Path, nargs="+", required=True,
                        help="directories of PMC*.xml / PMC*.txt, searched recursively")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    parser.add_argument("--limit", type=int, default=0, help="0 = all")
    return parser.parse_args()


def verify_animal_coverage(pcdb: pd.DataFrame, resolver: dict) -> None:
    """Every animal PCDB records for a publication must reach the prompt.
    """
    recorded = {}
    for row in pcdb.itertuples():
        recorded.setdefault(row.pmcid, set()).update(
            pcdb_entities.parse_list(row.animal_PCDB_id))

    short = []
    for pmcid, animals in recorded.items():
        entry = resolver["rows"].get(pmcid)
        if entry is None:
            continue
        if set(entry["animals"]) != animals:
            short.append((pmcid, sorted(animals), sorted(entry["animals"])))

    if short:
        detail = "; ".join(f"{p}: PCDB has {a}, resolver has {b}"
                           for p, a, b in short[:5])
        raise ValueError(
            f"{len(short):,} publications would be enumerated with the wrong "
            f"animal set - the animal axis is being truncated. First: {detail}")
    print(f"  animal coverage verified over {len(recorded):,} publications",
          flush=True)


def locate_text(text_dirs: list[Path]) -> dict[str, Path]:
    """pmcid -> the file its article text comes from, preferring JATS.

    Both passes run over every directory before the .txt pass begins, so a
    JATS file in a later directory still wins over BioC text in an earlier
    one: the XML carries section structure that the plain text has already
    flattened."""
    located = {}
    for suffix in (".xml", ".txt"):
        for text_dir in text_dirs:
            if not text_dir.is_dir():
                raise NotADirectoryError(f"--text-dir {text_dir} is not a directory")
            for path in text_dir.rglob(f"PMC*{suffix}"):
                located.setdefault(path.stem, path)
    return located


def relations_for(pmcid: str, entities: dict, names: dict) -> list[dict]:
    """Every combination PCDB's entities imply, which is what the audit tests."""
    diseases, drugs, animals = list(entities["diseases"]), list(entities["drugs"]), entities["animals"]
    rows = []

    def add(kind, **fields):
        rows.append({"relation_type": kind, "disease": "", "drug": "", "animal": "",
                     "disease_id": "", "drug_id": "", "animal_id": "", **fields})

    for d, r in product(diseases, drugs):
        add("disease-drug", disease=names["disease"][d], disease_id=d,
            drug=names["drug"][r], drug_id=r)
    for d, a in product(diseases, animals):
        add("disease-animal", disease=names["disease"][d], disease_id=d,
            animal=names["animal"][a], animal_id=a)
    for r, a in product(drugs, animals):
        add("drug-animal", drug=names["drug"][r], drug_id=r,
            animal=names["animal"][a], animal_id=a)
    for d, r, a in product(diseases, drugs, animals):
        add("disease-drug-animal", disease=names["disease"][d], disease_id=d,
            drug=names["drug"][r], drug_id=r, animal=names["animal"][a], animal_id=a)
    return rows


def build_record(pmcid: str, row_id_base: int, resolver, names, text_paths,
                 pcdb, catalogs) -> dict:
    entities = pcdb_entities.resolve_publication(
        pmcid, pcdb, catalogs, resolver=resolver)
    relations = relations_for(pmcid, entities, names)
    for offset, relation in enumerate(relations):
        relation["row_id"] = row_id_base + offset

    record = {"pmcid": pmcid,
              "link": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}",
              "n_relations": len(relations),
              "text_source": source_of(text_paths.get(pmcid)),
              "body_chars": 0, "status": "", "relations": relations, "prompt": ""}

    # Cases where no prompt can exist. `prompt` stays empty and status says why.
    if not relations:
        record["status"] = "no_relations"
        return record
    if pmcid not in text_paths:
        record["status"] = "no_text"
        return record
    if len(relations) > MAX_RELATIONS:
        record["status"] = "oversized"
        return record

    try:
        article = prompt_module.article_text(text_paths[pmcid])
    except Exception as error:                 # noqa: BLE001 - never abort the sweep
        record["status"] = "unparseable"
        record["detail"] = f"{type(error).__name__}: {error}"[:140]
        return record

    # Text-quality check folded in here rather than run as a separate sweep:
    # both passes read the same files, and I/O is the scarce resource on this
    # host. A prompt is still emitted when the body is thin or absent - it is
    # answerable, and the caller can filter on status - so `prompt` is empty
    # only for the three cases above.
    found = BODY_SECTION.search(article)
    record["body_chars"] = len(found.group(1).strip()) if found else 0
    record["prompt"] = prompt_module.build_prompt(
        pmcid, relations, article, entities)
    if record["body_chars"] >= MIN_BODY_CHARS:
        record["status"] = "ok"
    elif record["body_chars"] > 0:
        record["status"] = "thin"
    else:
        record["status"] = "no_body"
    return record


def source_of(path: Path | None) -> str:
    if path is None:
        return "none"
    return "bioc" if path.suffix == ".txt" else "jats"


def write_shards(pmcids, resolver, names, text_paths, pcdb, catalogs, args) -> list[dict]:
    """Streams to disk: the full set is ~4 GB and must never be held in memory."""
    if args.limit:
        pmcids = pmcids[: args.limit]
    index_path = args.outdir / "index.csv"
    columns = ["pmcid", "link", "status", "text_source", "n_relations",
               "first_row_id", "prompt_chars", "body_chars", "shard"]
    summary, row_id = [], 1
    sink = None
    started = time.time()

    with index_path.open("w", newline="", encoding="utf-8") as index_file:
        index = csv.DictWriter(index_file, fieldnames=columns)
        index.writeheader()
        for position, pmcid in enumerate(pmcids):
            if position % args.shard_size == 0:
                if sink:
                    sink.close()
                shard = args.outdir / f"input_{position // args.shard_size:03d}.jsonl"
                sink = shard.open("w", encoding="utf-8")
                print(f"  {shard.name} ...", flush=True)
            record = build_record(pmcid, row_id, resolver, names, text_paths, pcdb, catalogs)
            row_id += max(len(record["relations"]), 1)
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            entry = {"pmcid": pmcid, "link": record["link"], "status": record["status"],
                     "text_source": record["text_source"], "n_relations": record["n_relations"],
                     "first_row_id": record["relations"][0]["row_id"] if record["relations"] else "",
                     "prompt_chars": len(record["prompt"]),
                     "body_chars": record["body_chars"],
                     "shard": args.outdir.name + "/" + shard.name}
            index.writerow(entry)
            summary.append(entry)
            if len(summary) % 5_000 == 0:
                rate = len(summary) / max(time.time() - started, 1e-9)
                print(f"    {len(summary):,}/{len(pmcids):,} "
                      f"({100*len(summary)/len(pmcids):.0f}%) at {rate:.0f}/s, "
                      f"~{(len(pmcids)-len(summary))/rate/60:.0f} min left", flush=True)
    if sink:
        sink.close()
    return summary


def report(written: list[dict], outdir: Path) -> None:
    import statistics as st
    status = Counter(r["status"] for r in written)
    source = Counter(r["text_source"] for r in written)
    sized = [r["prompt_chars"] for r in written if r["prompt_chars"]]
    shards = sorted(outdir.glob("input_*.jsonl"))

    print(f"\n{len(written):,} publications -> {len(shards)} shards in {outdir}")
    print(f"index: {outdir/'index.csv'}\n")
    print("status:")
    for name, n in status.most_common():
        print(f"  {name:<14} {n:>8,}  ({100*n/len(written):5.2f}%)")
    print("\ntext source:")
    for name, n in source.most_common():
        print(f"  {name:<14} {n:>8,}")
    if sized:
        print(f"\nprompts built: {len(sized):,}")
        print(f"  size: median {st.median(sized):,.0f} chars (~{st.median(sized)/3.3:,.0f} tok), "
              f"max {max(sized):,}")
        print(f"  total input: ~{sum(sized)/3.3/1e6:,.1f}M tokens")
    print(f"\non disk: {sum(s.stat().st_size for s in shards)/1e9:.2f} GB")
    relations = sum(r["n_relations"] for r in written)
    print(f"relations enumerated: {relations:,}")

    # The folded-in text verification: these three statuses all carry a prompt,
    # so they are graded on whether the methods/results section actually
    # arrived rather than on whether a file existed.
    graded = [r for r in written if r["status"] in ("ok", "thin", "no_body")]
    if graded:
        body = [r["body_chars"] for r in graded if r["body_chars"]]
        print("\ntext coverage (folded-in verification):")
        print(f"  publications with a prompt     {len(graded):>8,}")
        print(f"  usable methods+results (>=1k)  "
              f"{sum(1 for r in graded if r['status'] == 'ok'):>8,}")
        print(f"  thin (<1k chars)               "
              f"{sum(1 for r in graded if r['status'] == 'thin'):>8,}")
        print(f"  no methods/results section     "
              f"{sum(1 for r in graded if r['status'] == 'no_body'):>8,}")
        if body:
            print(f"  methods+results: median {st.median(body):,.0f} chars, "
                  f"min {min(body):,}, max {max(body):,}")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
