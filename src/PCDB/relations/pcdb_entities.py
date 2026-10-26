"""PCDB entity resolution and article text, for building extraction prompts.

Two jobs, both prerequisites of a prompt:

  * resolve a publication to the disease, drug and animal entities PCDB
    recorded for it, keeping the mention each canonical name was extracted
    from - a model cannot match 'DOX' in the text to 'Doxorubicin' without it;
  * assemble the article text the model reads: title, abstract, methods and
    results, plus table and caption text, each capped so the roster never
    dwarfs the paper.

Trimmed from the audit library to the inference path only. The resolution and
text functions are byte-identical to the ones that produced the published
labels.
"""

import ast
import re
from pathlib import Path

import pandas as pd
from lxml import etree

SECTION_PATTERN = re.compile(
    r"method|material|result|experimental|procedure|animal|treatment|in vivo|protocol|design", re.I
)

METHODS_CHAR_CAP = 60_000
TABLE_CHAR_CAP = 6_000
CAPTION_CHAR_CAP = 6_000


def load_pcdb(release: Path) -> pd.DataFrame:
    return pd.read_csv(require(release / "pcdb_complete.tsv"), sep="\t", dtype=str)


def load_catalogs(release: Path) -> dict:
    return {kind: pd.read_csv(require(release / f"{kind}_entities.tsv"),
                              sep="\t", dtype=str)
            for kind in ("disease", "drug", "animal")}


def require(path: Path) -> Path:
    """Fail naming the missing file, so a wrong --pcdb-dir is obvious rather
    than surfacing later as an empty catalog."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; is --pcdb-dir a PCDB release?")
    return path


def parse_list(cell) -> list:
    if not isinstance(cell, str) or not cell.strip():
        return []
    try:
        return list(ast.literal_eval(cell))
    except (ValueError, SyntaxError):
        return []


def build_lookup(catalog: pd.DataFrame, id_column: str) -> dict[str, list[str]]:
    """External ID -> every PCDB_id carrying it, mirroring
    build_pcdb.create_pcdb_id_LUT.

    One external identifier can belong to several PCDB entities, and PCDB
    assigns all of them. Keeping only the first - which `setdefault` to a
    scalar did - lost 2,600 relations across 839 publications that PCDB
    asserts, because the entity the audit resolved to was not always the one
    PCDB recorded.
    """
    lookup = {}
    for row in catalog.itertuples():
        for external in parse_list(getattr(row, id_column)):
            lookup.setdefault(external, []).append(row.PCDB_id)
    return lookup


def resolve(mentions, external_ids, lookups) -> tuple[dict, list]:
    """Group mentions under the PCDB id they resolved to; keep what didn't resolve.

    PCDB stores only the union of resolved ids, discarding which mention produced
    which. The *_external_ids columns are positionally aligned with the name
    lists, so the alignment is recoverable by redoing the lookup per mention.
    """
    resolved, unmapped = {}, []
    for mention, ids in zip(mentions, external_ids):
        hits = []
        for source, lookup in lookups:
            if source in ids:
                for external in ids[source]:
                    hits.extend(lookup.get(external, []))
                if hits:
                    break
        if hits:
            for pcdb_id in dict.fromkeys(hits):
                resolved.setdefault(pcdb_id, []).append(mention)
        else:
            unmapped.append(mention)
    return resolved, unmapped


def build_resolver(pcdb: pd.DataFrame, catalogs: dict) -> dict:
    """Precompute what resolve_publication otherwise rebuilds on every call.

    resolve_publication is fine for a handful of publications and quadratic for
    the whole database: it scans all 186k PCDB rows to find one pmcid and
    rebuilds five catalogue lookups each time. Pass the result as `resolver=`
    to make a full-database sweep linear.
    """
    rows = {}
    for row in pcdb.itertuples():
        entry = rows.get(row.pmcid)
        if entry is None:
            rows[row.pmcid] = entry = {"first": row, "strains": set(),
                                       "animals": {}, "disease_ids": set(),
                                       "drug_ids": set()}
        # PCDB's own columns define which entities the audit may enumerate.
        entry["disease_ids"].update(parse_list(row.disease_PCDB_id))
        entry["drug_ids"].update(parse_list(row.drug_PCDB_id))
        strain = getattr(row, "animal_strain", None)
        if isinstance(strain, str) and strain.strip():
            entry["strains"].add(strain.strip())
        # PCDB stores one row per (publication x animal group), so a study in
        # mice and rats has two rows. Taking the first row alone truncated the
        # animal axis to one species. Diseases and drugs are repeated verbatim
        # on every row, so those still come from `first`. Insertion order is
        # kept so a single-species publication enumerates exactly as before.
        species = getattr(row, "animal_species", None)
        species = species.strip() if isinstance(species, str) else ""
        for animal_id in parse_list(row.animal_PCDB_id):
            entry["animals"].setdefault(animal_id, species)
    return {
        "rows": rows,
        "disease": [("mesh", build_lookup(catalogs["disease"], "mesh_id")),
                    ("diseaseontology", build_lookup(catalogs["disease"], "diseaseontology_id"))],
        "drug": [("drugbank", build_lookup(catalogs["drug"], "drugbank_id")),
                 ("mesh", build_lookup(catalogs["drug"], "mesh_id")),
                 ("UMLS", build_lookup(catalogs["drug"], "UMLS_id"))],
    }


def resolve_publication(pmcid: str, pcdb: pd.DataFrame, catalogs: dict,
                        resolver: dict | None = None) -> dict:
    if resolver is not None:
        entry = resolver["rows"].get(pmcid)
        if entry is None:
            raise KeyError(f"{pmcid} not in PCDB")
        row, strains = entry["first"], sorted(entry["strains"])
        animals = dict(entry["animals"])
        asserted = {"disease": set(entry["disease_ids"]),
                    "drug": set(entry["drug_ids"])}
        disease_lookups, drug_lookups = resolver["disease"], resolver["drug"]
    else:
        rows = pcdb[pcdb.pmcid == pmcid]
        if rows.empty:
            raise KeyError(f"{pmcid} not in PCDB")
        row = rows.iloc[0]
        strains = sorted({s.strip() for s in rows.animal_strain.dropna() if s.strip()})
        animals = {}
        asserted = {"disease": set(), "drug": set()}
        for _, each in rows.iterrows():
            species = each.animal_species
            species = species.strip() if isinstance(species, str) else ""
            for animal_id in parse_list(each.animal_PCDB_id):
                animals.setdefault(animal_id, species)
            asserted["disease"].update(parse_list(each.disease_PCDB_id))
            asserted["drug"].update(parse_list(each.drug_PCDB_id))
        disease_lookups = [("mesh", build_lookup(catalogs["disease"], "mesh_id")),
                           ("diseaseontology",
                            build_lookup(catalogs["disease"], "diseaseontology_id"))]
        drug_lookups = [("drugbank", build_lookup(catalogs["drug"], "drugbank_id")),
                        ("mesh", build_lookup(catalogs["drug"], "mesh_id")),
                        ("UMLS", build_lookup(catalogs["drug"], "UMLS_id"))]

    diseases, disease_unmapped = resolve(
        parse_list(row.disease_names), parse_list(row.disease_external_ids), disease_lookups)
    drugs, drug_unmapped = resolve(
        parse_list(row.drug_names), parse_list(row.drug_external_ids), drug_lookups)

    # The identifier lookup exists to attach each mention to its entity, so the
    # prompt can say which wording a canonical name came from. It must not
    # decide *which* entities are in scope: resolving through external ids finds
    # drugs PCDB extracted but left unmapped in its own column, and auditing
    # those would report rates against a database PCDB is not. PCDB's columns
    # are therefore authoritative - resolved ids outside them are dropped, and
    # asserted ids the resolution missed are kept with no mention hint.
    diseases = restrict(diseases, asserted["disease"])
    drugs = restrict(drugs, asserted["drug"])
    return {
        "diseases": diseases,
        "drugs": drugs,
        "animals": list(animals),
        # Each animal keeps the species string from the row that carried it.
        # One shared string would mislabel every species after the first.
        "animal_species_by_id": animals,
        "animal_species": (row.animal_species or "").strip() if isinstance(row.animal_species, str) else "",
        "strains": strains,
        "unmapped": disease_unmapped + drug_unmapped,
    }


def restrict(resolved: dict, asserted: set) -> dict:
    """Exactly the entities PCDB asserts, carrying any mentions found for them."""
    return {pcdb_id: resolved.get(pcdb_id, []) for pcdb_id in sorted(asserted)}


def flatten(node) -> str:
    if node is None:
        return ""
    return re.sub(r"\s+", " ", " ".join(node.itertext())).strip()


def build_article_text(xml_path: str) -> str:
    root = etree.parse(xml_path).getroot()
    sections = [
        flatten(section) for section in root.findall(".//body/sec")
        if SECTION_PATTERN.search(flatten(section.find("title")))
    ]
    selected = " ".join(sections) or flatten(root.find(".//body"))
    tables = " ".join(flatten(t) for t in root.findall(".//table-wrap/table"))
    captions = " ".join(flatten(c) for c in root.findall(".//caption"))
    return (
        f"TITLE: {flatten(root.find('.//article-title'))}\n\n"
        f"ABSTRACT: {flatten(root.find('.//abstract'))}\n\n"
        f"METHODS AND RESULTS: {selected[:METHODS_CHAR_CAP]}\n\n"
        f"TABLES: {tables[:TABLE_CHAR_CAP]}\n\n"
        f"CAPTIONS: {captions[:CAPTION_CHAR_CAP]}"
    )
