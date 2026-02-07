import pandas as pd
import ast
import argparse
from pathlib import Path


def create_LUT(keys, values):
    lut = {}
    for k, v in zip(keys, values):
        lut[k] = v
    return lut


def find_PCDB_id_for_disease(df, pcdb_disease_entities):
    first_cover_mesh = pcdb_disease_entities[
        pcdb_disease_entities["disease_mesh_id"] != ""
    ].copy()
    first_cover_mesh_lut = create_LUT(
        first_cover_mesh["disease_mesh_id"].to_list(),
        first_cover_mesh["PCDB_id"].to_list(),
    )

    second_cover_DO = pcdb_disease_entities[
        pcdb_disease_entities["disease_mesh_id"] == ""
    ].copy()
    second_cover_DO_lut = create_LUT(
        second_cover_DO["diseaseontology_ids"].to_list(),
        second_cover_DO["PCDB_id"].to_list(),
    )

    def _find(row):
        disease_external_ids = row["disease_external_ids"]
        disease_names = row["disease_names"]
        unresolved_disease_names = []
        unresolved_disease_external_ids = []
        pcdb_ids = set()
        for d_name, d_eid in zip(disease_names, disease_external_ids):
            resolved = False
            if "mesh" in d_eid:
                for m_id in d_eid["mesh"]:
                    if m_id in first_cover_mesh_lut:
                        pcdb_ids.add(first_cover_mesh_lut[m_id])
                        resolved = True
            elif "diseaseontology" in d_eid:
                for do_id in d_eid["diseaseontology"]:
                    if do_id in second_cover_DO_lut:
                        pcdb_ids.add(second_cover_DO_lut[do_id])
                        resolved = True
            if not resolved:
                unresolved_disease_names.append(d_name)
                unresolved_disease_external_ids.append(d_eid)

        if len(pcdb_ids) == 0:
            pcdb_ids = ""
        if len(unresolved_disease_names) == 0:
            unresolved_disease_names = ""
        if len(unresolved_disease_external_ids) == 0:
            unresolved_disease_external_ids = ""

        return pcdb_ids, unresolved_disease_names, unresolved_disease_external_ids

    df[["disease_PCDB_id", "unresolved_disease_name", "unresolved_disease_external_ids"]] = (
        df.apply(lambda x: _find(x), axis=1, result_type="expand")
    )
    df["disease_PCDB_id"] = df["disease_PCDB_id"].apply(
        lambda x: list(x) if x != "" else x
    )
    return df


def find_PCDB_id_for_drug(df, pcdb_drug_entities):
    first_cover_drugbank = pcdb_drug_entities[
        pcdb_drug_entities["drugbank_id"] != ""
    ].copy()
    pcdb_id_lut_only_db_ids = create_LUT(
        first_cover_drugbank["drugbank_id"].tolist(),
        first_cover_drugbank["PCDB_id"].tolist(),
    )

    second_cover_mesh = pcdb_drug_entities[
        pcdb_drug_entities["mesh_id"].apply(lambda x: (x != "") and isinstance(x, str))
    ]
    pcdb_id_lut_only_mesh_ids = create_LUT(
        second_cover_mesh["mesh_id"].tolist(),
        second_cover_mesh["PCDB_id"].tolist(),
    )

    third_cover_umls = pcdb_drug_entities[
        pcdb_drug_entities["UMLS_id"].apply(lambda x: (x != "") and isinstance(x, str))
    ]
    pcdb_id_lut_only_umls_ids = create_LUT(
        third_cover_umls["UMLS_id"].tolist(),
        third_cover_umls["PCDB_id"].tolist(),
    )

    def _find(row):
        d_eids = row["drug_external_ids"]
        drug_names = row["drug_names"]
        unresolved_drug_names = []
        unresolved_drug_external_ids = []
        pcdb_ids = []
        for drug_name, drug_eid in zip(drug_names, d_eids):
            resolved = False
            if "drugbank" in drug_eid:
                for db_id in drug_eid["drugbank"]:
                    pcdb_ids.append(pcdb_id_lut_only_db_ids[db_id])
                    resolved = True
            elif "mesh" in drug_eid:
                for mesh_id in drug_eid["mesh"]:
                    if mesh_id in pcdb_id_lut_only_mesh_ids:
                        pcdb_ids.append(pcdb_id_lut_only_mesh_ids[mesh_id])
                        resolved = True
            elif "UMLS" in drug_eid:
                for umls_id in drug_eid["UMLS"]:
                    if umls_id in pcdb_id_lut_only_umls_ids:
                        pcdb_ids.append(pcdb_id_lut_only_umls_ids[umls_id])
                        resolved = True
            if not resolved:
                unresolved_drug_names.append(drug_name)
                unresolved_drug_external_ids.append(drug_eid)

        if len(pcdb_ids) == 0:
            pcdb_ids = ""
        if len(unresolved_drug_names) == 0:
            unresolved_drug_names = ""
            unresolved_drug_external_ids = ""

        return pcdb_ids, unresolved_drug_names, unresolved_drug_external_ids

    df[["drug_PCDB_id", "unresolved_drug_names", "unresolved_drug_external_ids"]] = (
        df.apply(lambda x: _find(x), result_type="expand", axis=1)
    )


def find_PCDB_id_for_animal(df, pcdb_animal_entities):
    animal_model_LUT = create_LUT(
        pcdb_animal_entities["mesh_id"].tolist(),
        pcdb_animal_entities["PCDB_id"].tolist(),
    )

    def _find(row):
        animal_external_ids = row["animal_external_ids"]
        animal_name = row["animal_species"]
        unresolved_animal_names = ""
        unresolved_animal_external_ids = ""
        pcdb_ids = set()

        resolved = False
        if "mesh" in animal_external_ids:
            m_id = animal_external_ids["mesh"]
            if m_id in animal_model_LUT:
                pcdb_ids.add(animal_model_LUT[m_id])
                resolved = True
        if not resolved:
            unresolved_animal_names = animal_name
            unresolved_animal_external_ids = animal_external_ids

        if len(pcdb_ids) == 0:
            pcdb_ids = ""

        return pcdb_ids, unresolved_animal_names, unresolved_animal_external_ids

    df[["animal_PCDB_id", "unresolved_animal_name", "unresolved_animal_external_ids"]] = (
        df.apply(lambda x: _find(x), axis=1, result_type="expand")
    )
    df["animal_PCDB_id"] = df["animal_PCDB_id"].apply(
        lambda x: list(x) if x != "" else x
    )
    return df


def find_entity_name(target_df, entity_df, target_col, entity_name_col, pcdb_id_col):
    entity_df_ = entity_df.set_index("PCDB_id")
    target_df[target_col] = target_df[pcdb_id_col].apply(
        lambda x: [entity_df_.loc[x_][entity_name_col] for x_ in x]
    )


def main(db_file_path: str, entities_dir: str, output_dir: str):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    entities_path = Path(entities_dir)

    # Load data
    pcdb = pd.read_excel(
        db_file_path, engine="openpyxl", dtype=str,
        sheet_name="pcdb_mapped", keep_default_na=False,
    )
    unmapped_extractions = pd.read_excel(
        db_file_path, engine="openpyxl", dtype=str,
        sheet_name="unmapped_extractions", keep_default_na=False,
    )

    disease_entities = pd.read_csv(
        entities_path / "disease_entities.tsv", sep="\t", dtype=str, keep_default_na=False
    )
    drug_entities = pd.read_csv(
        entities_path / "drug_entities.tsv", sep="\t", dtype=str, keep_default_na=False
    )
    animal_entities = pd.read_csv(
        entities_path / "animal_entities.tsv", sep="\t", dtype=str, keep_default_na=False
    )

    # Parse list/dict columns
    for c in ["disease_names", "disease_external_ids", "drug_names",
              "drug_external_ids", "animal_external_ids"]:
        pcdb[c] = pcdb[c].apply(
            lambda x: ast.literal_eval(x) if (x != "" and isinstance(x, str)) else x
        )

    # Resolve PCDB IDs for each entity type
    print("Resolving disease PCDB IDs...")
    find_PCDB_id_for_disease(pcdb, disease_entities)

    print("Resolving drug PCDB IDs...")
    find_PCDB_id_for_drug(pcdb, drug_entities)

    print("Resolving animal PCDB IDs...")
    find_PCDB_id_for_animal(pcdb, animal_entities)

    # Select columns for complete output
    pcdb = pcdb[[
        "pmcid", "date", "link", "confidence_score", "TITLE",
        "disease_PCDB_id", "drug_PCDB_id", "animal_PCDB_id",
        "unresolved_disease_name", "unresolved_disease_external_ids",
        "unresolved_drug_names", "unresolved_drug_external_ids",
        "unresolved_animal_name", "unresolved_animal_external_ids",
        "disease_names", "disease_external_ids", "drug_names",
        "drug_external_ids", "animal_strain", "animal_species",
        "total_subject_size", "animal_external_ids",
    ]]

    # Filter to rows with all three entity types resolved
    pcdb_clean = pcdb[
        (pcdb["disease_PCDB_id"] != "")
        & (pcdb["drug_PCDB_id"] != "")
        & (pcdb["animal_PCDB_id"] != "")
    ].copy()

    n_complete = pcdb_clean["pmcid"].nunique()
    print(f"Publications with all entities resolved: {n_complete}")

    pcdb_clean = pcdb_clean[[
        "pmcid", "date", "link", "confidence_score", "TITLE",
        "disease_PCDB_id", "drug_PCDB_id", "animal_PCDB_id",
        "animal_strain", "animal_species", "total_subject_size",
    ]].copy()

    # Add entity names
    find_entity_name(pcdb_clean, disease_entities, "disease_name", "disease_name", "disease_PCDB_id")
    find_entity_name(pcdb_clean, drug_entities, "drug_name", "drug_name", "drug_PCDB_id")
    find_entity_name(pcdb_clean, animal_entities, "animal_name", "animal_name", "animal_PCDB_id")

    pcdb_clean = pcdb_clean[[
        "pmcid", "date", "link", "confidence_score", "TITLE",
        "disease_name", "drug_name", "animal_name",
        "disease_PCDB_id", "drug_PCDB_id", "animal_PCDB_id",
        "animal_strain", "animal_species", "total_subject_size",
    ]].copy()

    # Save outputs
    pcdb_clean.to_csv(output_path / "pcdb.tsv", sep="\t", index=False)
    pcdb.to_csv(output_path / "pcdb_complete.tsv", sep="\t", index=False)
    unmapped_extractions.to_csv(
        output_path / "unmapped_entity_extractions.tsv", sep="\t", index=False
    )

    with pd.ExcelWriter(output_path / "preclinical_database.xlsx", engine="openpyxl") as writer:
        pcdb_clean.to_excel(writer, sheet_name="preclinical_database", index=False)
        disease_entities.to_excel(writer, sheet_name="disease_entities", index=False)
        drug_entities.to_excel(writer, sheet_name="drug_entities", index=False)
        animal_entities.to_excel(writer, sheet_name="animal_entities", index=False)

    print(f"Outputs saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the PCDB from mapped data and entity tables.")
    parser.add_argument("--db_file", type=str,
                        default="outputs/pcdb/pcdb_mapped.xlsx",
                        help="Path to the mapped PCDB Excel file.")
    parser.add_argument("--entities_dir", type=str,
                        default="outputs/pcdb",
                        help="Directory containing entity TSV files.")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/pcdb",
                        help="Output directory.")
    args = parser.parse_args()

    main(args.db_file, args.entities_dir, args.output_dir)
