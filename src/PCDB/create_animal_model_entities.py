import pandas as pd
import ast
import pickle
import argparse
from pathlib import Path


def create_tree_id_lut(mesh_desc_path: str):
    mesh_desc_data = pd.read_csv(mesh_desc_path, sep="\t")
    mesh_desc_data["tree_numbers"] = mesh_desc_data["tree_numbers"].apply(
        lambda x: ast.literal_eval(x)
    )
    tree_id_lut = {}
    for _, row in mesh_desc_data.iterrows():
        tree_id_lut[row["mesh_id"]] = row["tree_numbers"]
    return tree_id_lut


def get_mesh_tree_id(df: pd.DataFrame, column_str: str, mesh_desc_path: str):
    tree_id_lut = create_tree_id_lut(mesh_desc_path)
    df["mesh_tree_id"] = df[column_str].apply(
        lambda x: tree_id_lut[x] if x in tree_id_lut else ""
    )
    print(f"Unfound mesh tree id: {len(df[df['mesh_tree_id'] == ''])}")
    return df


def main(db_file_path: str, mesh_synonyms_path: str, mesh_desc_path: str,
         output_dir: str):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    pcdb = pd.read_excel(
        db_file_path, engine="openpyxl", dtype=str,
        sheet_name="pcdb_mapped", keep_default_na=False,
    )

    with open(mesh_synonyms_path, "rb") as f:
        mesh_synonyms_LUT = pickle.load(f)

    pcdb["animal_external_ids"] = pcdb["animal_external_ids"].apply(
        lambda x: ast.literal_eval(x) if x != "" else x
    )
    pcdb["animal_mesh_id"] = pcdb["animal_external_ids"].apply(
        lambda x: x["mesh"] if x != "" else ""
    )

    # Build animal model entities from unique MeSH IDs
    unique_animal_mesh_ids = pcdb[pcdb["animal_mesh_id"] != ""]["animal_mesh_id"].unique().tolist()
    animal_model_entities = pd.DataFrame.from_dict({"mesh_id": unique_animal_mesh_ids})

    # Add tree IDs
    animal_model_entities = get_mesh_tree_id(animal_model_entities, "mesh_id", mesh_desc_path)

    # Add names, synonyms, and PCDB IDs
    animal_model_entities["animal_name"] = animal_model_entities["mesh_id"].apply(
        lambda x: mesh_synonyms_LUT[x][0]
    )
    animal_model_entities["synonyms"] = animal_model_entities["mesh_id"].apply(
        lambda x: mesh_synonyms_LUT[x]
    )
    animal_model_entities["PCDB_id"] = [
        f"PCDB_AN{i}" for i in range(1, len(animal_model_entities) + 1)
    ]

    # Reorder columns and save
    animal_model_entities = animal_model_entities[
        ["PCDB_id", "animal_name", "synonyms", "mesh_id", "mesh_tree_id"]
    ]

    output_file = output_path / "animal_entities.tsv"
    animal_model_entities.to_csv(output_file, sep="\t", index=False)
    print(f"Animal model entities saved to {output_file} ({len(animal_model_entities)} rows)")


if __name__ == "__main__":
    lut_dir = "data/PCDB/external_database_mapping_utils/external_database_lut"

    parser = argparse.ArgumentParser(description="Create animal model entities for PCDB.")
    parser.add_argument("--db_file", type=str,
                        default="outputs/pcdb/pcdb_mapped.xlsx",
                        help="Path to the mapped PCDB Excel file.")
    parser.add_argument("--mesh_synonyms", type=str,
                        default=f"{lut_dir}/mesh_synonyms_LUT.pkl",
                        help="Path to MeSH synonyms lookup table pickle.")
    parser.add_argument("--mesh_desc", type=str,
                        default=f"{lut_dir}/mesh_descriptive_data.tsv",
                        help="Path to MeSH descriptive data TSV.")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/pcdb",
                        help="Output directory for animal_entities.tsv.")
    args = parser.parse_args()

    main(args.db_file, args.mesh_synonyms, args.mesh_desc, args.output_dir)
