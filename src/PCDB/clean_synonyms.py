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


def verify_mesh_ids(df, mesh_desc_path: str, mesh_data_path: str):
    tree_id_lut = create_tree_id_lut(mesh_desc_path)
    mesh_data = pd.read_csv(mesh_data_path, sep="\t")
    mesh_data = mesh_data.set_index("mesh_id")

    def _verify_mesh_id(mesh_ids):
        if mesh_ids == "":
            return ""
        if mesh_ids.startswith("["):
            mesh_ids = ast.literal_eval(mesh_ids)
        else:
            return mesh_ids
        verified_mesh_ids = []
        for mesh_id in mesh_ids:
            verified = False
            if mesh_id not in tree_id_lut:
                pas = mesh_data.loc[mesh_id]["pharmacological_actions"]
                pas = ast.literal_eval(pas)
                if len(pas) > 0:
                    verified = True
            else:
                mesh_tree_ids = tree_id_lut[mesh_id]
                for tree_id in mesh_tree_ids:
                    if tree_id.startswith("D"):
                        verified = True
            if verified:
                verified_mesh_ids.append(mesh_id)
        if len(verified_mesh_ids) == 0:
            return ""
        return str(verified_mesh_ids)

    df["mesh_id"] = df["mesh_id"].apply(_verify_mesh_id)


def verify_umls_ids(df, umls_semantic_path: str):
    with open(umls_semantic_path, "rb") as f:
        umls_semantic_lut = pickle.load(f)
    allowed_semantic_types = ["T195", "T200", "T121"]  # Antibiotic, Clinical drugs, Pharmacological substance

    def _verify_umls_id(umls_ids):
        if umls_ids == "":
            return ""
        if umls_ids.startswith("["):
            umls_ids = ast.literal_eval(umls_ids)
        else:
            return umls_ids
        verified_umls_ids = []
        for umls_id in umls_ids:
            s_types = umls_semantic_lut[umls_id]
            for s_t in s_types:
                if s_t in allowed_semantic_types:
                    verified_umls_ids.append(umls_id)
                    break
        if len(verified_umls_ids) == 0:
            return ""
        return str(verified_umls_ids)

    df["UMLS_id"] = df["UMLS_id"].apply(_verify_umls_id)


def find_drug_synonyms(row, mesh_synonyms_lut, umls_synonyms_lut, drugbank_synonyms_lut):
    mesh_ids = row["mesh_id"] if row["mesh_id"] != "" else []
    umls_ids = row["UMLS_id"] if row["UMLS_id"] != "" else []
    drugbank_ids = row["drugbank_id"] if row["drugbank_id"] != "" else []
    synonyms = []
    for drugbank_id in drugbank_ids:
        if drugbank_id in drugbank_synonyms_lut:
            synonyms.extend(drugbank_synonyms_lut[drugbank_id])
    for mesh_id in mesh_ids:
        if mesh_id in mesh_synonyms_lut:
            synonyms.extend(mesh_synonyms_lut[mesh_id])
    for umls_id in umls_ids:
        if umls_id in umls_synonyms_lut:
            synonyms.extend(umls_synonyms_lut[umls_id])
    synonyms = [s.lower() for s in synonyms]
    unique = list(dict.fromkeys(synonyms))[:10]
    return sorted(unique, key=len)


def clean_disease_synonyms(row):
    mesh_synonyms = ast.literal_eval(row["mesh_synonyms"]) if row["mesh_synonyms"] != "" else []
    do_synonyms = ast.literal_eval(row["diseaseontology_synonyms"]) if row["diseaseontology_synonyms"] != "" else []
    all_synonyms = [s.lower() for s in mesh_synonyms + do_synonyms]
    unique = list(dict.fromkeys(all_synonyms))[:10]
    sorted_unique = sorted(unique, key=len)
    return sorted_unique if sorted_unique else ""


def clean_animal_synonyms(row):
    synonyms = ast.literal_eval(row["synonyms"]) if row["synonyms"] != "" else []
    all_synonyms = [s.lower() for s in synonyms]
    unique = list(dict.fromkeys(all_synonyms))[:10]
    sorted_unique = sorted(unique, key=len)
    return sorted_unique if sorted_unique else ""


def main(input_dir: str, mesh_desc_path: str, mesh_data_path: str,
         umls_semantic_path: str, mesh_synonyms_path: str,
         umls_synonyms_path: str, drugbank_synonyms_path: str,
         output_dir: str):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load entity tables
    drug_entities = pd.read_csv(input_path / "drug_entities.tsv", sep="\t", dtype=str, keep_default_na=False)
    disease_entities = pd.read_csv(input_path / "disease_entities.tsv", sep="\t", dtype=str, keep_default_na=False)
    animal_entities = pd.read_csv(input_path / "animal_entities.tsv", sep="\t", dtype=str, keep_default_na=False)

    # --- Drug entities ---
    verify_mesh_ids(drug_entities, mesh_desc_path, mesh_data_path)
    verify_umls_ids(drug_entities, umls_semantic_path)

    # Parse ID columns to lists
    for col in ["mesh_id", "UMLS_id", "drugbank_id"]:
        drug_entities[col] = drug_entities[col].apply(
            lambda x: [x] if not x.startswith("[") and x != "" else x
        )
        drug_entities[col] = drug_entities[col].apply(
            lambda x: ast.literal_eval(x) if not isinstance(x, list) and x != "" else x
        )

    with open(mesh_synonyms_path, "rb") as f:
        mesh_synonyms_lut = pickle.load(f)
    with open(umls_synonyms_path, "rb") as f:
        umls_synonyms_lut = pickle.load(f)
    with open(drugbank_synonyms_path, "rb") as f:
        drugbank_synonyms_lut = pickle.load(f)

    drug_entities["drug_synonyms"] = drug_entities.apply(
        find_drug_synonyms, axis=1,
        args=(mesh_synonyms_lut, umls_synonyms_lut, drugbank_synonyms_lut),
    )

    # --- Disease entities ---
    disease_entities["disease_synonyms"] = disease_entities.apply(clean_disease_synonyms, axis=1)
    disease_entities.drop(columns=["synonyms"], inplace=True)
    disease_entities["disease_mesh_id"] = disease_entities.apply(
        lambda row: [row["disease_mesh_id"]] if row["disease_mesh_id"] != "" else "", axis=1
    )
    disease_entities["diseaseontology_ids"] = disease_entities["diseaseontology_ids"].apply(
        lambda x: ast.literal_eval(x) if x.startswith("[") else [x]
    )
    disease_entities["diseaseontology_ids"] = disease_entities["diseaseontology_ids"].apply(
        lambda x: "" if len(x) == 1 and x[0] == "" else x
    )
    disease_entities.rename(
        columns={"disease_mesh_id": "mesh_id", "diseaseontology_ids": "diseaseontology_id"},
        inplace=True,
    )

    # --- Animal entities ---
    animal_entities["animal_synonyms"] = animal_entities.apply(clean_animal_synonyms, axis=1)
    animal_entities.drop(columns=["synonyms"], inplace=True)

    # Save outputs
    drug_entities.to_csv(output_path / "drug_entities.tsv", sep="\t", index=False)
    disease_entities.to_csv(output_path / "disease_entities.tsv", sep="\t", index=False)
    animal_entities.to_csv(output_path / "animal_entities.tsv", sep="\t", index=False)
    print(f"Saved cleaned entities to {output_path}")


if __name__ == "__main__":
    lut_dir = "data/PCDB/external_database_mapping_utils/external_database_lut"

    parser = argparse.ArgumentParser(description="Clean and verify synonyms for PCDB entities.")
    parser.add_argument("--input_dir", type=str, default="outputs/pcdb",
                        help="Directory containing drug_entities.tsv, disease_entities.tsv, animal_entities.tsv.")
    parser.add_argument("--mesh_desc", type=str, default=f"{lut_dir}/mesh_descriptive_data.tsv",
                        help="Path to MeSH descriptive data TSV.")
    parser.add_argument("--mesh_data", type=str, default=f"{lut_dir}/mesh_data.tsv",
                        help="Path to MeSH data TSV.")
    parser.add_argument("--umls_semantic", type=str, default=f"{lut_dir}/UMLS_semantic_LUT.pkl",
                        help="Path to UMLS semantic type lookup table pickle.")
    parser.add_argument("--mesh_synonyms", type=str, default=f"{lut_dir}/mesh_synonyms_LUT.pkl",
                        help="Path to MeSH synonyms lookup table pickle.")
    parser.add_argument("--umls_synonyms", type=str, default=f"{lut_dir}/UMLS_synonyms_LUT.pkl",
                        help="Path to UMLS synonyms lookup table pickle.")
    parser.add_argument("--drugbank_synonyms", type=str, default=f"{lut_dir}/drugbank_synonyms_LUT.pkl",
                        help="Path to DrugBank synonyms lookup table pickle.")
    parser.add_argument("--output_dir", type=str, default="outputs/pcdb",
                        help="Output directory for cleaned entity TSVs.")
    args = parser.parse_args()

    main(args.input_dir, args.mesh_desc, args.mesh_data, args.umls_semantic,
         args.mesh_synonyms, args.umls_synonyms, args.drugbank_synonyms, args.output_dir)
