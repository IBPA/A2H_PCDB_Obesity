"""Cross-disease PCDB summary tables (revision edit #1).

Reads outputs/pcdb/ and prints the main-text table to stdout. Nothing is saved.

Two universes underlie the reported figures:
  * "Classified" = all preclinical articles in pcdb_complete.tsv whose
    disease_PCDB_id list contains the target disease (134,565-article frame).
  * "Structured" = the subset of classified pubs with all three entity types
    (disease + drug + animal) successfully normalized; equal to pcdb.tsv
    filtered by disease (72,134-article frame). Verified by set equality.
"""
from __future__ import annotations
import ast
from collections import Counter
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PCDB_DIR = ROOT / "outputs" / "pcdb"

# PCDB IDs are positional and shift when the pipeline is rebuilt, so each disease is
# keyed on its (stable) entity name and the PCDB_id is resolved at runtime. First field
# is the canonical disease_name in disease_entities.tsv; second is the display label.
DISEASES = [
    ("Obesity", "Obesity", "Proof-of-concept disease used for A2H analysis"),
    ("Diabetes Mellitus, Type 2", "Type 2 diabetes mellitus", "Metabolic disease comparator"),
    ("Alzheimer Disease", "Alzheimer disease", "Neurodegeneration comparator"),
    ("Arthritis, Rheumatoid", "Rheumatoid arthritis", "Inflammatory/autoimmune comparator"),
    ("Breast Neoplasms", "Breast neoplasms", "High-evidence oncology comparator"),
]


def parse_list(x):
    if pd.isna(x) or x == "" or x == "[]":
        return []
    try:
        v = ast.literal_eval(x)
        return v if isinstance(v, list) else []
    except (ValueError, SyntaxError):
        return []


def has_entity(x):
    return bool(parse_list(x))


def load_structured():
    df = pd.read_csv(
        PCDB_DIR / "pcdb.tsv", sep="\t",
        usecols=["pmcid", "disease_PCDB_id", "drug_PCDB_id",
                 "animal_PCDB_id", "animal_species", "date"],
    )
    for col in ["disease_PCDB_id", "drug_PCDB_id", "animal_PCDB_id"]:
        df[col] = df[col].map(parse_list)
    df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year
    df["species_set"] = df["animal_species"].map(
        lambda s: {s.strip().lower()} if isinstance(s, str) and s.strip()
        else set()
    )
    return df


def load_classified():
    df = pd.read_csv(
        PCDB_DIR / "pcdb_complete.tsv", sep="\t",
        usecols=["pmcid", "disease_PCDB_id", "drug_PCDB_id",
                 "animal_PCDB_id", "date"],
    )
    df["disease_ids"] = df["disease_PCDB_id"].map(parse_list)
    df["has_drug"] = df["drug_PCDB_id"].map(has_entity)
    df["has_animal"] = df["animal_PCDB_id"].map(has_entity)
    df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year
    return df


def classified_stats(classified, did):
    """Return per-pmcid classified-universe stats for one disease."""
    sub = classified[classified["disease_ids"].apply(lambda l: did in l)]
    pm = sub.groupby("pmcid").agg(
        has_dr=("has_drug", "any"),
        has_a=("has_animal", "any"),
        year_min=("year", "min"),
        year_max=("year", "max"),
    )
    return {
        "n_classified": len(pm),
        "n_structured": int((pm["has_dr"] & pm["has_a"]).sum()),
        "pct_drug_mapped": 100.0 * pm["has_dr"].mean() if len(pm) else 0.0,
        "pct_animal_mapped": 100.0 * pm["has_a"].mean() if len(pm) else 0.0,
        "year_min": (int(pm["year_min"].min())
                     if pm["year_min"].notna().any() else None),
        "year_max": (int(pm["year_max"].max())
                     if pm["year_max"].notna().any() else None),
    }


def structured_stats(structured, did, drug_name):
    sub = structured[structured["disease_PCDB_id"].apply(lambda l: did in l)]
    n_pubs = sub["pmcid"].nunique()
    n_drugs = len({d for lst in sub["drug_PCDB_id"] for d in lst})
    n_species = len({a for lst in sub["animal_PCDB_id"] for a in lst})

    with_sp = sub[sub["species_set"].map(bool)]
    denom_sp = with_sp["pmcid"].nunique()

    def pct(name):
        if denom_sp == 0:
            return 0.0
        n = with_sp[with_sp["species_set"].apply(lambda s: name in s)][
            "pmcid"].nunique()
        return 100.0 * n / denom_sp

    pct_mouse = pct("mouse")
    pct_rat = pct("rat")

    other_counter: Counter = Counter()
    for s_set in with_sp["species_set"]:
        for s in s_set - {"mouse", "rat"}:
            other_counter[s] += 1
    top_other = other_counter.most_common(3)
    pct_other = 0.0
    if denom_sp:
        n_other = with_sp[with_sp["species_set"]
            .apply(lambda s: bool(s - {"mouse", "rat"}))]["pmcid"].nunique()
        pct_other = 100.0 * n_other / denom_sp

    pub_drug = sub[["pmcid", "drug_PCDB_id"]].explode("drug_PCDB_id") \
        .dropna(subset=["drug_PCDB_id"])
    denom_dr = pub_drug["pmcid"].nunique()
    if denom_dr > 0:
        top = (pub_drug.groupby("drug_PCDB_id")["pmcid"].nunique()
               .sort_values(ascending=False))
        top1 = top.index[0]
        dom_cell = (f"{drug_name.get(top1, top1)} "
                    f"({100.0*top.iloc[0]/denom_dr:.1f}%)")
        top3 = "; ".join(f"{drug_name.get(i, i)} ({int(n)})"
                         for i, n in top.head(3).items())
    else:
        dom_cell = "—"
        top3 = "—"

    yr_min = int(sub["year"].min()) if sub["year"].notna().any() else None
    yr_max = int(sub["year"].max()) if sub["year"].notna().any() else None

    return dict(
        n_pubs=n_pubs, n_drugs=n_drugs, n_species=n_species,
        denom_sp=denom_sp,
        pct_mouse=pct_mouse, pct_rat=pct_rat, pct_other=pct_other,
        top_other=top_other,
        dom_cell=dom_cell, top3=top3,
        year_min=yr_min, year_max=yr_max,
    )


def resolve_disease_ids() -> dict[str, str]:
    """Map each case disease's entity name to its current PCDB_id, failing if a name
    is absent or ambiguous. Resolving by name keeps the report correct across rebuilds,
    which reshuffle the positional PCDB IDs."""
    entities = pd.read_csv(PCDB_DIR / "disease_entities.tsv", sep="\t",
                           usecols=["PCDB_id", "disease_name"])
    id_of = {}
    for entity_name, label, _ in DISEASES:
        match = entities[entities["disease_name"].str.lower() == entity_name.lower()]
        if len(match) != 1:
            raise ValueError(
                f"'{entity_name}' matched {len(match)} disease entities (expected 1); "
                f"check the DISEASES list against disease_entities.tsv")
        did = match["PCDB_id"].iloc[0]
        id_of[entity_name] = did
        print(f"  {did:12} -> {label} ({entity_name})")
    return id_of


def main():
    print("=== DISEASE ID CHECK ===")
    disease_id = resolve_disease_ids()
    structured = load_structured()
    classified = load_classified()
    drugs = pd.read_csv(PCDB_DIR / "drug_entities.tsv", sep="\t",
                        usecols=["PCDB_id", "drug_name"])
    drug_name = dict(zip(drugs["PCDB_id"], drugs["drug_name"]))

    main_rows = []
    for entity_name, label, interp in DISEASES:
        did = disease_id[entity_name]
        c = classified_stats(classified, did)
        s = structured_stats(structured, did, drug_name)
        assert c["n_structured"] == s["n_pubs"], (
            f"Universe mismatch for {did}: classified-derived structured "
            f"{c['n_structured']} != pcdb.tsv structured {s['n_pubs']}")
        rate = 100.0 * s["n_pubs"] / c["n_classified"]

        main_rows.append({
            "Disease area": label,
            "PCDB-structured publications": s["n_pubs"],
            "Structuring rate (%)": f"{rate:.1f}",
            "Unique mapped drugs": s["n_drugs"],
            "Unique animal species": s["n_species"],
            "% mouse": f"{s['pct_mouse']:.1f}",
            "% rat": f"{s['pct_rat']:.1f}",
            "Dominant compound (%)": s["dom_cell"],
            "Interpretation": interp,
        })
    print("=== MAIN TABLE ===")
    print(pd.DataFrame(main_rows).to_string(index=False))


if __name__ == "__main__":
    main()
