#!/usr/bin/env python3
"""Extraction accuracy against the human-adjudicated validation sample.

One input, and only one:

    data/PCDB/pcdb_relations/relation_validation_sample.csv

Each row is one relation PCDB asserts. `gemma_4_e4b_label` is the extraction
model's verdict and `author_label` is ground truth - the authors' adjudication
where two independent annotators disagreed, and their agreed label elsewhere.

    python score_relation_validation.py
    python score_relation_validation.py --sample /path/to/sample.csv

The sample is stratified by the model's label, 60 per label, so it measures
the model's conditional error rates and not the database's label mix. A
rejection rate read straight off it would describe the sample, not PCDB, so
none is reported here. The two conditional rates needed to correct a
database-wide rate are printed instead, since they are what the sample can
support; reweighting them belongs with the file that holds the database.

    python relation_statistics.py     # the database-wide statistics

Per-relation-type accuracy is shown because a reviewer will ask for it, but
the sample was stratified by label and not by type, so each type rests on
roughly a quarter of the rows. Those rows are too few for a stratified claim
and are reported as counts for that reason.
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VERDICTS = {"yes", "no", "unclear"}


def main() -> int:
    args = parse_arguments()
    graded = read_sample(args.sample)
    overall = score(graded)
    by_type = {kind: score([g for g in graded if g["relation_type"] == kind])
               for kind in sorted({g["relation_type"] for g in graded})}
    conditional = conditional_rates(graded)
    report(graded, overall, by_type, conditional, args.sample)
    write_json(args.out, overall, by_type, conditional)
    return 0


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path,
                        default=ROOT / "data/PCDB/pcdb_relations"
                                       "/relation_validation_sample.csv")
    parser.add_argument("--model-column", default="gemma_4_e4b_label")
    parser.add_argument("--gold-column", default="author_label")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "outputs/relations/relation_validation_scores.json")
    return parser.parse_args()


def read_sample(sample: Path) -> list[dict]:
    """Accepts either the published column names or the working ones, so the
    workbook export and the csv both load."""
    csv.field_size_limit(10**9)
    with sample.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{sample} holds no rows")

    model_column = pick(rows[0], "gemma_4_e4b_label", "gemma_label")
    gold_column = pick(rows[0], "author_label", "final_annotation_authors")

    graded = []
    for row in rows:
        model = (row[model_column] or "").strip().lower()
        gold = (row[gold_column] or "").strip().lower()
        if gold not in VERDICTS:
            raise ValueError(f"{row['pmcid']}: gold label {gold!r} is not a verdict")
        graded.append({"pmcid": row["pmcid"],
                       "relation_type": row["relation_type"],
                       "model": model, "gold": gold})
    return graded


def pick(row: dict, *candidates: str) -> str:
    for name in candidates:
        if name in row:
            return name
    raise KeyError(f"none of {candidates} present; columns are {sorted(row)}")


def score(graded: list[dict]) -> dict:
    """Treats 'yes' as the positive class; anything else counts as a rejection,
    so the handful of 'unclear' verdicts are not silently dropped."""
    tp = sum(1 for g in graded if g["model"] == "yes" and g["gold"] == "yes")
    fp = sum(1 for g in graded if g["model"] == "yes" and g["gold"] != "yes")
    fn = sum(1 for g in graded if g["model"] != "yes" and g["gold"] == "yes")
    tn = sum(1 for g in graded if g["model"] != "yes" and g["gold"] != "yes")
    n = len(graded)
    observed = (tp + tn) / n
    expected = (((tp + fp) / n) * ((tp + fn) / n)
                + ((fn + tn) / n) * ((fp + tn) / n))
    return {
        "n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": round(observed, 3),
        "precision": round(tp / (tp + fp), 3) if tp + fp else None,
        "recall": round(tp / (tp + fn), 3) if tp + fn else None,
        "specificity": round(tn / (tn + fp), 3) if tn + fp else None,
        "f1": round(2 * tp / (2 * tp + fp + fn), 3) if tp else None,
        "kappa": round((observed - expected) / (1 - expected), 3)
        if expected != 1 else None,
    }


def conditional_rates(graded: list[dict]) -> dict:
    """P(truly no | the model said X), the two quantities a label-stratified
    sample estimates and the only ones that transfer to the database."""
    rates = {}
    for said in ("yes", "no"):
        stratum = [g for g in graded if g["model"] == said]
        wrong = sum(1 for g in stratum if g["gold"] == "no")
        rates[said] = {"n": len(stratum), "truly_no": wrong,
                       "p_truly_no": round(wrong / len(stratum), 3)
                       if stratum else None}
    return rates


def report(graded, overall, by_type, conditional, sample: Path) -> None:
    gold = Counter(g["gold"] for g in graded)
    print(f"Extraction accuracy on {len(graded)} adjudicated relations "
          f"in {len({g['pmcid'] for g in graded})} publications")
    print(f"  source: {sample}")
    print(f"  ground truth: {gold['yes']} yes / {gold['no']} no\n")

    print(f"  accuracy    {overall['accuracy']}")
    print(f"  precision   {overall['precision']}")
    print(f"  recall      {overall['recall']}")
    print(f"  specificity {overall['specificity']}")
    print(f"  F1          {overall['f1']}")
    print(f"  Cohen kappa {overall['kappa']}")
    print(f"  confusion   TP {overall['tp']}  FP {overall['fp']}  "
          f"FN {overall['fn']}  TN {overall['tn']}")

    print("\n  by relation type (n per type is small; counts, not claims)")
    print(f"    {'type':22s} {'n':>4s} {'acc':>6s} {'prec':>6s} {'rec':>6s}")
    for kind, s in by_type.items():
        print(f"    {kind:22s} {s['n']:>4} {str(s['accuracy']):>6s} "
              f"{str(s['precision']):>6s} {str(s['recall']):>6s}")

    print("\n  conditional error rates, for correcting a database-wide rate")
    for said, r in conditional.items():
        print(f"    P(truly no | model said {said!r:5s}) = {r['p_truly_no']}"
              f"   ({r['truly_no']}/{r['n']})")


def write_json(out: Path, overall, by_type, conditional) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"overall": overall, "by_relation_type": by_type,
                               "conditional_rates": conditional}, indent=2) + "\n",
                   encoding="utf-8")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
