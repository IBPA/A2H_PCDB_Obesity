import os
import pandas as pd
import ast
import pickle
import argparse
from pathlib import Path
from itertools import chain
from tqdm import tqdm

tqdm.pandas()


def create_disease_tree_id_lut(mesh_desc_path: str):
    mesh_desc_data = pd.read_csv(mesh_desc_path, sep="\t")
    mesh_desc_data["tree_numbers"] = mesh_desc_data["tree_numbers"].apply(
        lambda x: ast.literal_eval(x)
    )
    tree_id_lut = {}
    for _, row in mesh_desc_data.iterrows():
        tree_id_lut[row["mesh_id"]] = row["tree_numbers"]
    return tree_id_lut


def get_disease_mesh_tree_id(df: pd.DataFrame, mesh_desc_path: str):
    disease_tree_id_lut = create_disease_tree_id_lut(mesh_desc_path)
    df["disease_mesh_tree_id"] = df["disease_mesh_id"].apply(
        lambda x: disease_tree_id_lut[x] if x in disease_tree_id_lut else ""
    )
    print(f"Unfound mesh tree id: {len(df[df['disease_mesh_tree_id'] == ''])}")
    return df


def retrieve_DO_synonyms(DO_ids, DO_synonyms_LUT):
    if DO_ids == "":
        return ""
    if isinstance(DO_ids, str):
        DO_ids = [DO_ids]
    return [DO_synonyms_LUT[i] for i in DO_ids]


def combine_synonyms(row):
    synonym_groups = [row["mesh_synonyms"], row["diseaseontology_synonyms"]]
    all_synonyms = []
    for s_group in synonym_groups:
        if s_group == "":
            continue
        else:
            s_group = [s.lower() for s in s_group]
            all_synonyms = all_synonyms + s_group
    if len(all_synonyms) == 0:
        return ""
    return list(set(all_synonyms))


def main(db_file_path: str, mesh_synonyms_path: str, do_synonyms_path: str,
         mesh_desc_path: str, output_dir: str):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    pcdb = pd.read_excel(
        db_file_path, engine="openpyxl", dtype=str,
        sheet_name="pcdb_mapped", keep_default_na=False,
    )

    with open(mesh_synonyms_path, "rb") as f:
        mesh_synonyms_LUT = pickle.load(f)

    with open(do_synonyms_path, "rb") as f:
        DO_synonyms_LUT = pickle.load(f)

    pcdb["disease_external_ids"] = pcdb["disease_external_ids"].apply(
        lambda x: ast.literal_eval(x) if x != "" else x
    )

    # Extract unique disease MeSH IDs
    pcdb_disease = pcdb.explode("disease_external_ids")
    pcdb_disease = pcdb_disease[pcdb_disease["disease_external_ids"] != ""]
    pcdb_disease["disease_mesh_id"] = pcdb_disease["disease_external_ids"].apply(
        lambda x: x["mesh"] if "mesh" in x else ""
    )
    pcdb_disease = pcdb_disease[pcdb_disease["disease_mesh_id"] != ""]
    pcdb_disease = pcdb_disease.explode("disease_mesh_id")
    all_mapped_disease_mesh_ids = pcdb_disease["disease_mesh_id"].unique().tolist()

    # Build disease entities from MeSH IDs
    pcdb_disease_entities = pd.DataFrame.from_dict(
        {"disease_mesh_id": all_mapped_disease_mesh_ids}
    )
    pcdb_disease_entities = get_disease_mesh_tree_id(pcdb_disease_entities, mesh_desc_path)

    # Only C (disease category) and F03 (Mental disorders) are allowed
    pcdb_disease_entities["disease_categories"] = pcdb_disease_entities[
        "disease_mesh_tree_id"
    ].apply(lambda x: [i.split(".")[0] for i in x if i.startswith("C") or i.startswith("F03")])

    pcdb_disease_entities["disease_categories"] = pcdb_disease_entities[
        "disease_categories"
    ].apply(lambda x: "" if len(x) == 0 else x)
    pcdb_disease_entities = pcdb_disease_entities[
        pcdb_disease_entities["disease_categories"] != ""
    ].copy()
    pcdb_disease_entities["disease_categories"] = pcdb_disease_entities[
        "disease_categories"
    ].apply(lambda x: list(set(x)))

    # Exclude C22 - animal diseases
    pcdb_disease_entities["disease_categories"] = pcdb_disease_entities[
        "disease_categories"
    ].apply(lambda x: "" if (len(x) == 1 and x[0] == "C22") else x)

    # Assign PCDB IDs
    disease_PCDB_ids = [f"PCDB_DI{i}" for i in range(1, len(pcdb_disease_entities) + 1)]
    pcdb_disease_entities["PCDB_id"] = disease_PCDB_ids

    # Build MeSH-to-DO mapping
    mesh_ids_to_DO_ids = {
        m_id: set() for m_id in pcdb_disease_entities["disease_mesh_id"].unique()
    }

    for d_eids in tqdm(pcdb["disease_external_ids"].to_list(), total=len(pcdb)):
        for d_eid in d_eids:
            if "mesh" in d_eid and "diseaseontology" in d_eid:
                for m_id in d_eid["mesh"]:
                    if m_id in mesh_ids_to_DO_ids:
                        for do_id in d_eid["diseaseontology"]:
                            mesh_ids_to_DO_ids[m_id].add(do_id)

    pcdb_disease_entities["diseaseontology_ids"] = pcdb_disease_entities[
        "disease_mesh_id"
    ].apply(lambda x: list(mesh_ids_to_DO_ids[x]) if len(mesh_ids_to_DO_ids[x]) != 0 else "")

    # Find DO-only disease IDs (not linked to any MeSH)
    DO_ids_linked_mesh = set(
        chain.from_iterable(pcdb_disease_entities["diseaseontology_ids"].to_list())
    )

    pcdb_disease_DO = pcdb.explode("disease_external_ids")
    pcdb_disease_DO = pcdb_disease_DO[pcdb_disease_DO["disease_external_ids"] != ""]
    pcdb_disease_DO["disease_DO_id"] = pcdb_disease_DO["disease_external_ids"].apply(
        lambda x: x["diseaseontology"] if "diseaseontology" in x else ""
    )
    pcdb_disease_DO = pcdb_disease_DO[pcdb_disease_DO["disease_DO_id"] != ""]
    pcdb_disease_DO = pcdb_disease_DO.explode("disease_DO_id")
    all_mapped_disease_DO_ids = pcdb_disease_DO["disease_DO_id"].unique().tolist()

    mapped_disease_DO_ids_not_linked_mesh = [
        x for x in all_mapped_disease_DO_ids if x not in DO_ids_linked_mesh
    ]

    # Append DO-only entities
    current_id_max = len(pcdb_disease_entities)
    for do_id in mapped_disease_DO_ids_not_linked_mesh:
        new_row = pd.DataFrame.from_dict({
            "disease_mesh_id": "",
            "disease_mesh_tree_id": "",
            "disease_categories": "",
            "PCDB_id": f"PCDB_DI{current_id_max + 1}",
            "diseaseontology_ids": list([do_id]),
        })
        current_id_max += 1
        pcdb_disease_entities = pd.concat(
            [pcdb_disease_entities, new_row], ignore_index=True
        )

    # Add synonym columns
    pcdb_disease_entities["mesh_synonyms"] = pcdb_disease_entities["disease_mesh_id"].apply(
        lambda x: mesh_synonyms_LUT[x] if x != "" else ""
    )
    pcdb_disease_entities["mesh_descriptor_name"] = pcdb_disease_entities[
        "mesh_synonyms"
    ].apply(lambda x: x[0] if x != "" else "")

    pcdb_disease_entities["diseaseontology_synonyms"] = pcdb_disease_entities[
        "diseaseontology_ids"
    ].apply(lambda x: retrieve_DO_synonyms(x, DO_synonyms_LUT))
    pcdb_disease_entities["diseaseontology_synonyms"] = pcdb_disease_entities[
        "diseaseontology_synonyms"
    ].apply(lambda x: list(chain.from_iterable(x)) if x != "" else x)

    pcdb_disease_entities["disease_name"] = pcdb_disease_entities.apply(
        lambda x: x["mesh_descriptor_name"]
        if x["mesh_descriptor_name"] != ""
        else x["diseaseontology_synonyms"][0],
        axis=1,
    )

    pcdb_disease_entities["synonyms"] = pcdb_disease_entities.apply(combine_synonyms, axis=1)

    # Reorder columns and save
    pcdb_disease_entities = pcdb_disease_entities[
        [
            "PCDB_id", "disease_name", "synonyms",
            "mesh_descriptor_name",
            "disease_mesh_id", "diseaseontology_ids",
            "mesh_synonyms", "diseaseontology_synonyms",
            "disease_mesh_tree_id", "disease_categories",
        ]
    ]

    output_file = output_path / "disease_entities.tsv"
    pcdb_disease_entities.to_csv(output_file, sep="\t", index=False)
    print(f"Disease entities saved to {output_file} ({len(pcdb_disease_entities)} rows)")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent

    parser = argparse.ArgumentParser(description="Create disease entities for PCDB.")
    parser.add_argument("--db_file", type=str,
                        default="outputs/pcdb/pcdb_mapped.xlsx",
                        help="Path to the mapped PCDB Excel file.")
    parser.add_argument("--mesh_synonyms", type=str,
                        default="data/PCDB/external_database_mapping_utils/external_database_lut/mesh_synonyms_LUT.pkl",
                        help="Path to MeSH synonyms lookup table pickle.")
    parser.add_argument("--do_synonyms", type=str,
                        default="data/PCDB/external_database_mapping_utils/external_database_lut/disease_ontology_synonyms_LUT.pkl",
                        help="Path to Disease Ontology synonyms lookup table pickle.")
    parser.add_argument("--mesh_desc", type=str,
                        default="data/PCDB/external_database_mapping_utils/external_database_lut/mesh_descriptive_data.tsv",
                        help="Path to MeSH descriptive data TSV.")
    parser.add_argument("--output_dir", type=str,
                        default= "outputs/pcdb",
                        help="Output directory for disease_entities.tsv.")
    args = parser.parse_args()

    main(args.db_file, args.mesh_synonyms, args.do_synonyms, args.mesh_desc, args.output_dir)
