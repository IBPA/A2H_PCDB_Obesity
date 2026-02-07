import pandas as pd
import ast
import re
import itertools
import pickle
from collections import Counter
from tqdm import tqdm

tqdm.pandas()


def disease_mapping(row, lookup_table, ontology_name):
    if row.disease == "":
        return "", ""
    diseases = row.disease
    mapped_diseases = row['mapped_diseases'] if ('mapped_diseases' in row and row['mapped_diseases'] != '') else {}
    missed_diseases = []
    parsed_diseases = []
    for disease in diseases:
        disease = re.sub(r"\s*\(.*?\)", "", disease)
        disease = disease.replace("’", "'").replace("–", "-").lower()
        parsed_diseases.append(disease)
        if disease in lookup_table:
            uids = lookup_table[disease]
            if (disease in mapped_diseases):
                if (ontology_name in mapped_diseases[disease]):
                    mapped_diseases[disease][ontology_name] = list(set(mapped_diseases[disease][ontology_name] + uids))
                else:
                    mapped_diseases[disease][ontology_name] = uids
            else:
                mapped_diseases[disease] = {ontology_name: uids}

    for p_disease in parsed_diseases:
        if p_disease not in mapped_diseases:
            missed_diseases.append(p_disease)

    if len(mapped_diseases) == 0:
        mapped_diseases = ""
    if len(missed_diseases) == 0:
        missed_diseases = ""

    return mapped_diseases, missed_diseases


# def unmapped_disease_stats(pcdb):
#     unmapped_disease_all = \
#         list(itertools.chain(*(pcdb[pcdb['unmapped_diseases'] != ""]["unmapped_diseases"].to_list())))
#     unmapped_disease_counter = Counter(unmapped_disease_all)
#     return unmapped_disease_counter


def get_llm_mapped_disease_mesh_id(df, llm_mapped_mesh_LUT):
    for i, row in df.iterrows():
        unmapped_diseases = row["unmapped_diseases"]
        if unmapped_diseases == "":
            continue
        mapped_diseases = row["mapped_diseases"]

        for ud in unmapped_diseases:
            if ud in llm_mapped_mesh_LUT:
                found_mesh_id = llm_mapped_mesh_LUT[ud]
                if mapped_diseases == "":
                    mapped_diseases = {ud: {"mesh": [found_mesh_id]}}
                else:
                    mapped_diseases[ud] = {"mesh": [found_mesh_id]}

        remaining_diseases = [ud for ud in unmapped_diseases if ud not in mapped_diseases]
        if len(remaining_diseases) == 0:
            remaining_diseases = ""
        df.at[i, "mapped_diseases"] = mapped_diseases
        df.at[i, "unmapped_diseases"] = remaining_diseases
    return df


def load_mesh_lookup_table():
    """Load MeSH lookup table from pickle file."""
    with open("data/PCDB/external_database_mapping_utils/external_database_lut/mesh_ids_LUT.pkl", "rb") as f:
        mesh_lookup_table = pickle.load(f)
    mesh_lookup_table["diabetes"] = ["D003920"]
    return mesh_lookup_table


def load_disease_ontology_lookup_table():
    """Load Disease Ontology lookup table from pickle file."""
    with open("data/PCDB/external_database_mapping_utils/external_database_lut/disease_ontology_ids_LUT.pkl", "rb") as f:
        DO_lookup_table = pickle.load(f)
    return DO_lookup_table


def load_umls_lookup_table():
    """Load UMLS lookup table from pickle file."""
    with open("data/PCDB/external_database_mapping_utils/external_database_lut/UMLS_ids_LUT.pkl", "rb") as f:
        UMLS_LUT = pickle.load(f)
    return UMLS_LUT


def load_llm_mapped_mesh_lookup_table():
    """Load LLM mapped mesh disease lookup table from pickle file."""
    with open("data/PCDB/external_database_mapping_utils/external_database_lut/llm_mapping_disease_mesh_id_LUT.pkl", "rb") as f:
        llm_mapped_mesh_LUT = pickle.load(f)
    return llm_mapped_mesh_LUT


def main():
    # Load PCDB database
    pcdb = pd.read_csv("data/PCDB/preclinical_database_raw.tsv", sep="\t", keep_default_na=False, dtype=str)

    # Process disease column
    pcdb['disease'] = pcdb['disease'].apply(lambda x: ast.literal_eval(x) if x != '' else x)
    pcdb['disease'] = pcdb['disease'].apply(lambda x: list(set(x)) if x != '' else x)

    # Load MeSH disease terms and synonyms
    mesh_lookup_table = load_mesh_lookup_table()

    # Load Disease Ontology terms and synonyms
    DO_lookup_table = load_disease_ontology_lookup_table()

    # Map diseases to MeSH
    pcdb[["mapped_diseases", "unmapped_diseases"]] = pcdb.progress_apply(lambda x: disease_mapping(x, mesh_lookup_table, "mesh"),
                                                                              axis=1,
                                                                              result_type='expand')

    # Map diseases to Disease Ontology
    pcdb[["mapped_diseases", "unmapped_diseases"]] = pcdb.progress_apply(lambda x: disease_mapping(x, DO_lookup_table, "diseaseontology"),
                                                                              axis=1,
                                                                              result_type='expand')

    # Load UMLS lookup table
    UMLS_LUT = load_umls_lookup_table()
    pcdb[["mapped_diseases", "unmapped_diseases"]] = pcdb.progress_apply(lambda x: disease_mapping(x, UMLS_LUT, "UMLS"),
                                                                              axis=1,
                                                                              result_type='expand')

    # Clean up mapped and unmapped diseases
    pcdb["mapped_diseases"] = pcdb["mapped_diseases"].apply(lambda x: "" if len(x)==0 else x)
    pcdb["unmapped_diseases"] = pcdb["unmapped_diseases"].apply(lambda x: "" if len(x)==0 else x)

    # Load and apply LLM mapped mesh disease results
    llm_mapped_mesh_LUT = load_llm_mapped_mesh_lookup_table()
    pcdb = get_llm_mapped_disease_mesh_id(pcdb, llm_mapped_mesh_LUT)

    # Calculate and print final percentage of articles with mapped diseases
    pcdb_articles_with_mapped_disease = pcdb[pcdb["mapped_diseases"].astype(str) != '']
    print(f"\nFinal percentage of articles with mapped diseases: {len(pcdb_articles_with_mapped_disease) / len(pcdb)}")

    # Save results
    pcdb.to_csv("outputs/pcdb/mapped_diseases.tsv", sep="\t", index=False)
    print("\nResults saved to: outputs/pcdb/mapped_diseases.tsv")


if __name__ == "__main__":
    main()
