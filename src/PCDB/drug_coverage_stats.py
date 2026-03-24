import pandas as pd
import ast
import argparse


def load_pcdb(db_file_path: str):
    pcdb = pd.read_excel(
        db_file_path, sheet_name="pcdb_mapped", dtype=str, keep_default_na=False
    )
    for col in ["drug_external_ids", "disease_external_ids", "disease_names", "drug_names"]:
        pcdb[col] = pcdb[col].apply(lambda x: ast.literal_eval(x) if x != "" else [])
    pcdb["animal_external_ids"] = pcdb["animal_external_ids"].apply(
        lambda x: ast.literal_eval(x) if x != "" else {}
    )
    return pcdb


def drug_coverage(pcdb, drug_entities):
    db_lut = {
        row["drugbank_id"]: row["PCDB_id"]
        for _, row in drug_entities[drug_entities["drugbank_id"] != ""].iterrows()
    }
    mesh_lut = {
        row["mesh_id"]: row["PCDB_id"]
        for _, row in drug_entities[
            drug_entities["mesh_id"].apply(lambda x: x != "" and not x.startswith("["))
        ].iterrows()
    }
    umls_lut = {
        row["UMLS_id"]: row["PCDB_id"]
        for _, row in drug_entities[
            drug_entities["UMLS_id"].apply(lambda x: x != "" and not x.startswith("["))
        ].iterrows()
    }

    mapped_to_ext = pcdb[pcdb["drug_external_ids"].apply(lambda x: len(x) > 0)]["pmcid"].nunique()

    def has_pcdb_id(d_eids):
        for d_eid in d_eids:
            if "drugbank" in d_eid and any(db_id in db_lut for db_id in d_eid["drugbank"]):
                return True
            if "mesh" in d_eid and any(m_id in mesh_lut for m_id in d_eid["mesh"]):
                return True
            if "UMLS" in d_eid and any(u_id in umls_lut for u_id in d_eid["UMLS"]):
                return True
        return False

    mapped_to_pcdb = pcdb[pcdb["drug_external_ids"].apply(has_pcdb_id)]["pmcid"].nunique()
    return mapped_to_ext, mapped_to_pcdb


def disease_coverage(pcdb, disease_entities):
    mesh_lut = {
        row["disease_mesh_id"]: row["PCDB_id"]
        for _, row in disease_entities[disease_entities["disease_mesh_id"] != ""].iterrows()
    }
    do_lut = {
        row["diseaseontology_ids"]: row["PCDB_id"]
        for _, row in disease_entities[disease_entities["disease_mesh_id"] == ""].iterrows()
        if row["diseaseontology_ids"] != ""
    }

    mapped_to_ext = pcdb[pcdb["disease_external_ids"].apply(lambda x: len(x) > 0)]["pmcid"].nunique()

    def has_pcdb_id(d_eids):
        for d_eid in d_eids:
            if "mesh" in d_eid and any(m_id in mesh_lut for m_id in d_eid["mesh"]):
                return True
            if "diseaseontology" in d_eid and any(do_id in do_lut for do_id in d_eid["diseaseontology"]):
                return True
        return False

    mapped_to_pcdb = pcdb[pcdb["disease_external_ids"].apply(has_pcdb_id)]["pmcid"].nunique()
    return mapped_to_ext, mapped_to_pcdb


def animal_coverage(pcdb, animal_entities):
    mesh_lut = {
        row["mesh_id"]: row["PCDB_id"]
        for _, row in animal_entities[animal_entities["mesh_id"] != ""].iterrows()
    }

    mapped_to_ext = pcdb[pcdb["animal_external_ids"].apply(lambda x: len(x) > 0)]["pmcid"].nunique()

    def has_pcdb_id(a_eid):
        if "mesh" in a_eid and a_eid["mesh"] in mesh_lut:
            return True
        return False

    mapped_to_pcdb = pcdb[pcdb["animal_external_ids"].apply(has_pcdb_id)]["pmcid"].nunique()
    return mapped_to_ext, mapped_to_pcdb


def print_stats(label, total, mapped_to_ext, mapped_to_pcdb):
    print(f"\n{label}")
    print(f"  Mapped to external DB:  {mapped_to_ext:>7,}  ({mapped_to_ext/total*100:.1f}% of all)")
    print(f"  Resolved to PCDB ID:    {mapped_to_pcdb:>7,}  ({mapped_to_pcdb/total*100:.1f}% of all, "
          f"{mapped_to_pcdb/mapped_to_ext*100:.1f}% of ext-mapped)")


def main(db_file_path, drug_entities_path, disease_entities_path, animal_entities_path):
    pcdb = load_pcdb(db_file_path)
    total = pcdb["pmcid"].nunique()
    print(f"Total publications: {total:,}")

    drug_entities = pd.read_csv(drug_entities_path, sep="\t", dtype=str, keep_default_na=False)
    disease_entities = pd.read_csv(disease_entities_path, sep="\t", dtype=str, keep_default_na=False)
    animal_entities = pd.read_csv(animal_entities_path, sep="\t", dtype=str, keep_default_na=False)

    print_stats("Drug", total, *drug_coverage(pcdb, drug_entities))
    print_stats("Disease", total, *disease_coverage(pcdb, disease_entities))
    print_stats("Animal", total, *animal_coverage(pcdb, animal_entities))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute entity mapping coverage statistics.")
    parser.add_argument("--db_file", type=str, default="outputs/pcdb/pcdb_mapped.xlsx")
    parser.add_argument("--drug_entities", type=str, default="outputs/pcdb/drug_entities.tsv")
    parser.add_argument("--disease_entities", type=str, default="outputs/pcdb/disease_entities.tsv")
    parser.add_argument("--animal_entities", type=str, default="outputs/pcdb/animal_entities.tsv")
    args = parser.parse_args()

    main(args.db_file, args.drug_entities, args.disease_entities, args.animal_entities)
