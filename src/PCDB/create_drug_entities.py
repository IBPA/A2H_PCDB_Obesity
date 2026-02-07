import pandas as pd
import ast
import pickle
import argparse
from pathlib import Path
from itertools import chain
from tqdm import tqdm
from pandarallel import pandarallel

tqdm.pandas()


def create_tree_id_lut(mesh_desc_path: str):
    mesh_desc_data = pd.read_csv(mesh_desc_path, sep="\t")
    mesh_desc_data["tree_numbers"] = mesh_desc_data["tree_numbers"].apply(
        lambda x: ast.literal_eval(x)
    )
    tree_id_lut = {}
    for _, row in mesh_desc_data.iterrows():
        tree_id_lut[row["mesh_id"]] = row["tree_numbers"]
    return tree_id_lut


def representative_name_for_cui(
    mrconso,
    cui,
    tty_priority=("PT", "PN", "MH", "HT", "SY"),
    sab_priority=None,
):
    df = mrconso[mrconso["CUI"] == cui]

    tty_rank = {t: (len(tty_priority) - i) for i, t in enumerate(tty_priority)}
    sab_rank = (
        {s: (len(sab_priority) - i) for i, s in enumerate(sab_priority)}
        if sab_priority
        else {}
    )

    def score_row(r) -> tuple:
        ispref = 1 if str(r.get("ISPREF", "")).upper() == "Y" else 0
        ts_pref = 1 if str(r.get("TS", "")).upper() == "P" else 0
        tty = str(r.get("TTY", ""))
        sab = str(r.get("SAB", ""))
        strval = str(r.get("STR", ""))
        return (
            ispref,
            ts_pref,
            tty_rank.get(tty, 0),
            sab_rank.get(sab, 0),
            -len(strval),
            strval.lower(),
        )

    best = max(df.to_dict("records"), key=score_row)
    return best.get("STR")


def get_drug_synonyms(row, drugbank_common_name_LUT, drugbank_synonyms_LUT,
                      mesh_synonyms_LUT, umls_synonyms_LUT):
    drugbank_id = row["drugbank_id"]
    mesh_id = row["mesh_id"]
    umls_id = row["UMLS_id"]
    official_drug_name = ""
    official_synonyms = []

    if drugbank_id != "":
        official_drug_name = drugbank_common_name_LUT[drugbank_id]
    elif mesh_id != "":
        official_drug_name = mesh_synonyms_LUT[mesh_id][0]
    else:
        official_drug_name = f"UMLS_{umls_id}"

    if drugbank_id != "":
        official_synonyms = official_synonyms + drugbank_synonyms_LUT[drugbank_id]
    if mesh_id != "":
        if isinstance(mesh_id, str):
            mesh_id = [mesh_id]
        for m_id in mesh_id:
            official_synonyms = official_synonyms + mesh_synonyms_LUT[m_id]
    if umls_id != "":
        if isinstance(umls_id, str):
            umls_id = [umls_id]
        for u_id in umls_id:
            official_synonyms = official_synonyms + umls_synonyms_LUT[u_id]

    official_synonyms = [s.lower() for s in official_synonyms]
    official_synonyms = list(set(official_synonyms))
    return official_drug_name, official_synonyms


def main(db_file_path: str, mesh_synonyms_path: str, drugbank_synonyms_path: str,
         drugbank_common_name_path: str, umls_synonyms_path: str,
         umls_semantic_path: str, mesh_data_path: str, mesh_desc_path: str,
         mrconso_path: str, output_dir: str):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    pcdb = pd.read_excel(
        db_file_path, engine="openpyxl", dtype=str,
        sheet_name="pcdb_mapped", keep_default_na=False,
    )

    with open(mesh_synonyms_path, "rb") as f:
        mesh_synonyms_LUT = pickle.load(f)
    with open(drugbank_synonyms_path, "rb") as f:
        drugbank_synonyms_LUT = pickle.load(f)
    with open(drugbank_common_name_path, "rb") as f:
        drugbank_common_name_LUT = pickle.load(f)
    with open(umls_synonyms_path, "rb") as f:
        umls_synonyms_LUT = pickle.load(f)
    with open(umls_semantic_path, "rb") as f:
        umls_semantic_lut = pickle.load(f)

    pcdb["drug_external_ids"] = pcdb["drug_external_ids"].apply(
        lambda x: ast.literal_eval(x) if x != "" else x
    )

    # Extract unique DrugBank IDs
    pcdb_drug = pcdb.explode("drug_external_ids")
    pcdb_drug = pcdb_drug[pcdb_drug["drug_external_ids"] != ""].copy()
    pcdb_drug["drugbank_id"] = pcdb_drug["drug_external_ids"].apply(
        lambda x: x["drugbank"] if "drugbank" in x else ""
    )
    pcdb_drug = pcdb_drug[pcdb_drug["drugbank_id"] != ""]

    drugbank_ids = pcdb_drug["drugbank_id"].to_list()
    drugbank_ids = list(set(chain.from_iterable(drugbank_ids)))

    # Build initial drug entities from DrugBank IDs
    pcdb_drug_entities = pd.DataFrame.from_dict({"drugbank_id": drugbank_ids})
    drug_PCDB_ids = [f"PCDB_DR{i}" for i in range(1, len(pcdb_drug_entities) + 1)]
    pcdb_drug_entities["PCDB_id"] = drug_PCDB_ids

    # Map DrugBank -> MeSH
    drugbank_ids_to_mesh_ids = {
        db_id: set() for db_id in pcdb_drug_entities["drugbank_id"].unique()
    }
    for d_eids in tqdm(pcdb["drug_external_ids"].to_list(), total=len(pcdb),
                       desc="DrugBank->MeSH mapping"):
        for d_eid in d_eids:
            if "drugbank" in d_eid and "mesh" in d_eid:
                for db_id in d_eid["drugbank"]:
                    if db_id in drugbank_ids_to_mesh_ids:
                        for mesh_id in d_eid["mesh"]:
                            drugbank_ids_to_mesh_ids[db_id].add(mesh_id)

    pcdb_drug_entities["mesh_id"] = pcdb_drug_entities["drugbank_id"].apply(
        lambda x: list(drugbank_ids_to_mesh_ids[x]) if len(drugbank_ids_to_mesh_ids[x]) != 0 else ""
    )

    # Find MeSH-only drug IDs (not linked to any DrugBank)
    mesh_ids_linked_drugbank = set(
        chain.from_iterable(pcdb_drug_entities["mesh_id"].to_list())
    )

    pcdb_drug_MeSH = pcdb.explode("drug_external_ids")
    pcdb_drug_MeSH = pcdb_drug_MeSH[pcdb_drug_MeSH["drug_external_ids"] != ""]
    pcdb_drug_MeSH["drug_mesh_id"] = pcdb_drug_MeSH["drug_external_ids"].apply(
        lambda x: x["mesh"] if "mesh" in x else ""
    )
    pcdb_drug_MeSH = pcdb_drug_MeSH[pcdb_drug_MeSH["drug_mesh_id"] != ""]
    pcdb_drug_MeSH = pcdb_drug_MeSH.explode("drug_mesh_id")
    all_mapped_drug_mesh_ids = pcdb_drug_MeSH["drug_mesh_id"].unique().tolist()
    mapped_drug_mesh_ids_not_linked_drugbank = [
        x for x in all_mapped_drug_mesh_ids if x not in mesh_ids_linked_drugbank
    ]

    # Filter MeSH-only IDs: keep those with pharmacological actions or D-tree
    mesh_data = pd.read_csv(mesh_data_path, sep="\t")
    mesh_data = mesh_data.set_index("mesh_id")
    mesh_tree_id_lut = create_tree_id_lut(mesh_desc_path)

    drug_mesh_ids_with_PA = set()
    for m_id in mapped_drug_mesh_ids_not_linked_drugbank:
        if m_id not in mesh_tree_id_lut:
            pas = mesh_data.loc[m_id]["pharmacological_actions"]
            pas = ast.literal_eval(pas)
            if len(pas) > 0:
                drug_mesh_ids_with_PA.add(m_id)
        else:
            tree_ids = mesh_tree_id_lut[m_id]
            top_tree_id = set([x.split(".")[0] for x in tree_ids])
            for tid in top_tree_id:
                if tid.startswith("D"):
                    drug_mesh_ids_with_PA.add(m_id)
                    break

    # Append MeSH-only entities
    current_id_max = len(pcdb_drug_entities)
    for mesh_id in drug_mesh_ids_with_PA:
        new_row = pd.DataFrame.from_dict({
            "drugbank_id": "",
            "PCDB_id": f"PCDB_DR{current_id_max + 1}",
            "mesh_id": [mesh_id],
        })
        current_id_max += 1
        pcdb_drug_entities = pd.concat([pcdb_drug_entities, new_row], ignore_index=True)

    # Map DrugBank/MeSH -> UMLS
    drugbank_id_to_UMLS_id = {
        db_id: set()
        for db_id in pcdb_drug_entities[pcdb_drug_entities["drugbank_id"] != ""]["drugbank_id"].unique()
    }
    all_mesh_id = pcdb_drug_entities.explode("mesh_id")
    mesh_id_to_UMLS_id = {
        m_id: set()
        for m_id in all_mesh_id[all_mesh_id["mesh_id"] != ""]["mesh_id"].unique()
    }

    for d_eids in tqdm(pcdb["drug_external_ids"].to_list(), total=len(pcdb),
                       desc="UMLS mapping"):
        for d_eid in d_eids:
            if "drugbank" in d_eid and "UMLS" in d_eid:
                for db_id in d_eid["drugbank"]:
                    if db_id in drugbank_id_to_UMLS_id:
                        for umls_id in d_eid["UMLS"]:
                            drugbank_id_to_UMLS_id[db_id].add(umls_id)
            if "mesh" in d_eid and "UMLS" in d_eid:
                for mesh_id in d_eid["mesh"]:
                    if mesh_id in mesh_id_to_UMLS_id:
                        for umls_id in d_eid["UMLS"]:
                            mesh_id_to_UMLS_id[mesh_id].add(umls_id)

    pcdb_drug_entities["UMLS_id"] = pcdb_drug_entities["drugbank_id"].apply(
        lambda x: drugbank_id_to_UMLS_id[x] if x != "" else ""
    )

    def find_UMLS_id_by_mesh_id(row):
        mesh_ids = row["mesh_id"]
        if mesh_ids == "":
            return ""
        if isinstance(mesh_ids, str):
            mesh_ids = [mesh_ids]
        curr_umls_ids = row["UMLS_id"]
        if curr_umls_ids == "":
            curr_umls_ids = set()
        for mesh_id in mesh_ids:
            if mesh_id in mesh_id_to_UMLS_id:
                curr_umls_ids.update(mesh_id_to_UMLS_id[mesh_id])
        if len(curr_umls_ids) == 0:
            return ""
        return curr_umls_ids

    pcdb_drug_entities["UMLS_id"] = pcdb_drug_entities.apply(find_UMLS_id_by_mesh_id, axis=1)
    pcdb_drug_entities["UMLS_id"] = pcdb_drug_entities["UMLS_id"].apply(
        lambda x: list(x) if x != "" else x
    )

    # Find unresolved UMLS IDs (not linked to DrugBank or MeSH)
    resolved_UMLS_ids = pcdb_drug_entities[pcdb_drug_entities["UMLS_id"] != ""]["UMLS_id"].to_list()
    resolved_UMLS_ids = set(chain.from_iterable(resolved_UMLS_ids))

    unresolved_UMLS_ids = set()
    for d_eids in tqdm(pcdb["drug_external_ids"].to_list(), total=len(pcdb),
                       desc="Finding unresolved UMLS"):
        for d_eid in d_eids:
            if "UMLS" in d_eid:
                for umls_id in d_eid["UMLS"]:
                    if umls_id not in resolved_UMLS_ids:
                        unresolved_UMLS_ids.add(umls_id)

    # Filter unresolved UMLS by allowed semantic types
    allowed_semantic_types = ["T195", "T200", "T121"]  # Antibiotic, Clinical drugs, Pharmacological substance
    unresolved_drug_UMLS_ids = set()
    for umls_id in unresolved_UMLS_ids:
        s_types = umls_semantic_lut[umls_id]
        for s_t in s_types:
            if s_t in allowed_semantic_types:
                unresolved_drug_UMLS_ids.add(umls_id)

    # Append UMLS-only entities
    current_id_max = len(pcdb_drug_entities)
    for new_umls_id in unresolved_drug_UMLS_ids:
        new_row = pd.DataFrame.from_dict({
            "drugbank_id": "",
            "PCDB_id": f"PCDB_DR{current_id_max + 1}",
            "mesh_id": "",
            "UMLS_id": [new_umls_id],
        })
        current_id_max += 1
        pcdb_drug_entities = pd.concat([pcdb_drug_entities, new_row], ignore_index=True)

    # Get drug names and synonyms
    pcdb_drug_entities[["drug_name", "drug_synonyms"]] = pcdb_drug_entities.apply(
        get_drug_synonyms, axis=1, result_type="expand",
        args=(drugbank_common_name_LUT, drugbank_synonyms_LUT, mesh_synonyms_LUT, umls_synonyms_LUT),
    )

    # Resolve UMLS-only drug names using MRCONSO
    print("Loading MRCONSO for UMLS name resolution...")
    mrconso_df = pd.read_csv(mrconso_path, sep="|", header=None, dtype=str).iloc[:, :-1]
    mrconso_df.columns = [
        "CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF", "AUI", "SAUI",
        "SCUI", "SDUI", "SAB", "TTY", "CODE", "STR", "SRL", "SUPPRESS", "CVF",
    ]

    pandarallel.initialize(progress_bar=True)
    pcdb_drug_entities["drug_name"] = pcdb_drug_entities["drug_name"].parallel_apply(
        lambda x: representative_name_for_cui(mrconso_df, x[5:]) if x.startswith("UMLS") else x
    )

    # Reorder columns and save
    pcdb_drug_entities = pcdb_drug_entities[
        ["PCDB_id", "drug_name", "drug_synonyms", "drugbank_id", "mesh_id", "UMLS_id"]
    ]

    output_file = output_path / "drug_entities.tsv"
    pcdb_drug_entities.to_csv(output_file, sep="\t", index=False)
    print(f"Drug entities saved to {output_file} ({len(pcdb_drug_entities)} rows)")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    lut_dir = "data/PCDB/external_database_mapping_utils/external_database_lut"

    parser = argparse.ArgumentParser(description="Create drug entities for PCDB.")
    parser.add_argument("--db_file", type=str,
                        default="outputs/pcdb/pcdb_mapped.xlsx",
                        help="Path to the mapped PCDB Excel file.")
    parser.add_argument("--mesh_synonyms", type=str,
                        default=f"{lut_dir}/mesh_synonyms_LUT.pkl",
                        help="Path to MeSH synonyms lookup table pickle.")
    parser.add_argument("--drugbank_synonyms", type=str,
                        default=f"{lut_dir}/drugbank_synonyms_LUT.pkl",
                        help="Path to DrugBank synonyms lookup table pickle.")
    parser.add_argument("--drugbank_common_name", type=str,
                        default=f"{lut_dir}/drugbank_common_name_LUT.pkl",
                        help="Path to DrugBank common name lookup table pickle.")
    parser.add_argument("--umls_synonyms", type=str,
                        default=f"{lut_dir}/UMLS_synonyms_LUT.pkl",
                        help="Path to UMLS synonyms lookup table pickle.")
    parser.add_argument("--umls_semantic", type=str,
                        default=f"{lut_dir}/UMLS_semantic_LUT.pkl",
                        help="Path to UMLS semantic type lookup table pickle.")
    parser.add_argument("--mesh_data", type=str,
                        default=f"{lut_dir}/mesh_data.tsv",
                        help="Path to MeSH data TSV.")
    parser.add_argument("--mesh_desc", type=str,
                        default=f"{lut_dir}/mesh_descriptive_data.tsv",
                        help="Path to MeSH descriptive data TSV.")
    parser.add_argument("--mrconso", type=str,
                        default="data/PCDB/external_database_mapping_utils/UMLS/2024AB/META/MRCONSO.RRF",
                        help="Path to UMLS MRCONSO.RRF file.")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/pcdb",
                        help="Output directory for drug_entities.tsv.")
    args = parser.parse_args()

    main(args.db_file, args.mesh_synonyms, args.drugbank_synonyms,
         args.drugbank_common_name, args.umls_synonyms, args.umls_semantic,
         args.mesh_data, args.mesh_desc, args.mrconso, args.output_dir)
