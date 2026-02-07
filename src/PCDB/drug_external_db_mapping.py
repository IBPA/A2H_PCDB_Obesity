import pandas as pd
import ast
import re
import itertools
import pickle
from collections import Counter
from tqdm import tqdm

tqdm.pandas()


def normalize_greek_letters(drug_name):
    """Normalize Greek letters in drug names to their English equivalents."""
    greek_letter_map = {
        'α': 'alpha',
        'β': 'beta',
        'γ': 'gamma',
        'δ': 'delta',
        'ε': 'epsilon',
        'ζ': 'zeta',
        'η': 'eta',
        'θ': 'theta',
        'ι': 'iota',
        'κ': 'kappa',
        'λ': 'lambda',
        'μ': 'mu',
        'ν': 'nu',
        'ξ': 'xi',
        'ο': 'omicron',
        'π': 'pi',
        'ρ': 'rho',
        'σ': 'sigma',
        'τ': 'tau',
        'υ': 'upsilon',
        'φ': 'phi',
        'χ': 'chi',
        'ψ': 'psi',
        'ω': 'omega'
    }
    for greek_char, name in greek_letter_map.items():
        drug_name = drug_name.replace(greek_char, name)
    return drug_name


def detect_end_abbreviation(text):
    """Detects an abbreviation in the format 'word (abbreviation)' at the end of a string."""
    match = re.search(r"\s\((\w+)\)\s*$", text)
    return match.group(1) if match else None


def has_special_char(s):
    """Check if string contains special characters."""
    return (("\t" in s) or ("\r" in s) or ("\n" in s))


def contain_abrrv(k, v):
    """Check if key is contained in value (to avoid circular replacements)."""
    k = k.strip()
    v = v.strip()
    if v.startswith(k) and (" " not in v):
        return False
    return (k in v)


def clean_drug_names_list(names):
    """Clean drug names by removing irrelevant entries."""
    cleaned = []
    for name in names:
        original = name
        name = name.strip().lower()
        # Criteria 1: Remove 'compound {number or letter}'
        if re.fullmatch(r"compound\s+[a-z0-9]+", name):
            continue
        # Criteria 2: Only numbers
        if name.isdigit():
            continue
        # Criteria 3: Contains 'control' or 'vehicle'
        if 'control' in name or 'vehicle' in name:
            continue
        if name == 'pbs':
            continue
        # Criteria 4: Length less than 3
        if len(name) < 3:
            continue
        # Criteria 5: Contains ' igg'
        if ' igg' in name:
            continue
        # Criteria 6: Matches format '{number}{one letter}' (e.g., 14a, 2b)
        if re.fullmatch(r"\d+[a-zA-Z]", name):
            continue
        cleaned.append(original)
    if len(cleaned) == 0:
        cleaned = ""
    return cleaned


def load_pubchem_cids():
    """Load PubChem CID lookup table from file."""
    cid_result_df_raw = pd.read_csv(
        "data/PCDB/external_database_mapping_utils/pubchem/pubchem_cids.txt",
        sep="\t", keep_default_na=False, dtype=str, header=None
    )
    cid_result_df_raw.columns = ["drug", "CID"]
    cid_result_df_raw["drug"] = pd.Categorical(
        cid_result_df_raw["drug"],
        categories=cid_result_df_raw["drug"].unique(),
        ordered=True
    )
    cid_result_df_grouped = cid_result_df_raw.groupby(
        "drug", observed=True, as_index=False
    ).agg({"CID": lambda x: list(dict.fromkeys(x))})
    print("Retrieved PubChem CID:")
    print(cid_result_df_grouped.head())
    return cid_result_df_grouped


def load_pubchem_sids():
    """Load PubChem SID lookup table from file."""
    sid_result_df_raw = pd.read_csv(
        "data/PCDB/external_database_mapping_utils/pubchem/pubchem_sids.txt",
        sep="\t", keep_default_na=False, dtype=str, header=None
    )
    sid_result_df_raw.columns = ["drug", "SID"]
    sid_result_df_raw["drug"] = pd.Categorical(
        sid_result_df_raw["drug"],
        categories=sid_result_df_raw["drug"].unique(),
        ordered=True
    )
    sid_result_df_grouped = sid_result_df_raw.groupby(
        "drug", observed=True, as_index=False
    ).agg({"SID": lambda x: list(dict.fromkeys(x))})
    print("Retrieved PubChem SID:")
    print(sid_result_df_grouped.head())
    return sid_result_df_grouped


def load_drugbank_lookup_table():
    """Load DrugBank lookup table from pickle file."""
    with open("data/PCDB/external_database_mapping_utils/external_database_lut/drugbank_ids_LUT.pkl", "rb") as f:
        drugbank_LUT = pickle.load(f)
    return drugbank_LUT


def load_mesh_lookup_table():
    """Load MeSH lookup table from pickle file."""
    with open("data/PCDB/external_database_mapping_utils/external_database_lut/mesh_ids_LUT.pkl", "rb") as f:
        mesh_LUT = pickle.load(f)
    print(f"MeSH data: {len(mesh_LUT)}")
    return mesh_LUT


def load_umls_lookup_table():
    """Load UMLS lookup table from pickle file."""
    with open("data/PCDB/external_database_mapping_utils/external_database_lut/UMLS_ids_LUT.pkl", "rb") as f:
        UMLS_LUT = pickle.load(f)
    return UMLS_LUT


def get_pubchem_cids(df, cid_df):
    """Map drugs to PubChem CIDs."""
    def _get_cids(drug_name):
        matched_row = cid_df[cid_df["drug"] == drug_name]
        if len(matched_row) > 0:
            if len(matched_row.iloc[0]['CID']) == 1 and matched_row.iloc[0]['CID'][0] == '':
                return {}
            return {"pubchem": ["CID:" + cid for cid in matched_row.iloc[0]['CID']]}
        else:
            return {}
    df["external_ids"] = df["drug"].apply(_get_cids)
    return df


def get_drugbank_ids(df, drugbank_LUT):
    """Map drugs to DrugBank IDs."""
    for i, row in df.iterrows():
        external_ids = row["external_ids"]
        drug_name = row["drug"]
        if drug_name in drugbank_LUT:
            external_ids["drugbank"] = drugbank_LUT[drug_name]
            df.at[i, "external_ids"] = external_ids
    return df


def get_mesh_ids(df, mesh_LUT):
    """Map drugs to MeSH IDs."""
    for i, row in df.iterrows():
        external_ids = row["external_ids"]
        drug_name = row["drug"]
        if drug_name in mesh_LUT:
            external_ids["mesh"] = mesh_LUT[drug_name]
            df.at[i, "external_ids"] = external_ids
    return df


def get_umls_ids(df, UMLS_LUT):
    """Map drugs to UMLS IDs."""
    for i, row in df.iterrows():
        external_ids = row["external_ids"]
        drug_name = row["drug"]
        if drug_name in UMLS_LUT:
            external_ids["UMLS"] = UMLS_LUT[drug_name]
            df.at[i, "external_ids"] = external_ids
    return df


def get_pubchem_sids(df, sid_df):
    """Map drugs to PubChem SIDs (only if no CID exists)."""
    for i, row in df.iterrows():
        external_ids = row["external_ids"]
        drug_name = row["drug"]
        matched_row = sid_df[sid_df["drug"] == drug_name]
        if len(matched_row) > 0:
            if len(matched_row.iloc[0]['SID']) == 1 and matched_row.iloc[0]['SID'][0] == '':
                continue
            else:
                if "pubchem" not in external_ids:
                    # print(f"Found SID for: {drug_name}")
                    external_ids["pubchem"] = ["SID:" + sid for sid in matched_row.iloc[0]['SID']]
                    df.at[i, "external_ids"] = external_ids
    return df


def unmapped_drugs_stats(pcdb_mapped_drugs):
    """Calculate statistics for unmapped drugs."""
    unmapped_drug_list = pcdb_mapped_drugs[
        pcdb_mapped_drugs['unmapped_drugs'] != ""
    ]["unmapped_drugs"].to_list()
    unmapped_drugs_all = list(itertools.chain(*unmapped_drug_list))
    unmapped_drugs_counter = Counter(unmapped_drugs_all)
    print(f"Total unique unmapped drugs: {len(unmapped_drugs_counter.keys())}")
    return unmapped_drugs_counter


def clean_and_normalize_drug_names(df):
    """
    Clean and normalize drug names using abbreviation dictionary.

    For each row, this function:
    1. Replaces abbreviations with full names from the abbreviation dictionary
    2. Normalizes Greek letters to English equivalents
    3. Normalizes special characters (prime, hyphen)
    4. Removes trailing abbreviations in parentheses
    5. Converts to lowercase

    Args:
        df: DataFrame with 'drugs' and 'abbreviation_dict' columns

    Returns:
        DataFrame with added 'cleaned_drug_mapping' and 'cleaned_drug_name' columns
    """
    cleaned_drug_mapping = []
    cleaned_drug_name = []

    for i, row in df.iterrows():
        abbrv_dict = row["abbreviation_dict"]
        drugs = row["drugs"]

        replaced_drugs = {}
        # Handle abbreviation and replace with full name
        for d in drugs:
            if d in abbrv_dict and \
                (not has_special_char(abbrv_dict[d])) and \
                    (not contain_abrrv(d, abbrv_dict[d])):
                replaced_drugs[d] = abbrv_dict[d]
            else:
                replaced_drugs[d] = d

        # Clean the drug name
        for original_d, cleaned_d in replaced_drugs.items():
            cleaned_d = cleaned_d.lower()
            cleaned_d = normalize_greek_letters(cleaned_d)
            cleaned_d = re.sub(r"′", "'", cleaned_d)
            cleaned_d = re.sub("‐", "-", cleaned_d)
            if detect_end_abbreviation(cleaned_d):
                rm_parentheses = re.sub(r"\s\(\w+\)\s*$", "", cleaned_d)
                cleaned_d = rm_parentheses
            replaced_drugs[original_d] = cleaned_d

        replaced_drugs = {key: value.lower() for key, value in replaced_drugs.items()}
        cleaned_drug_mapping.append(replaced_drugs)
        cleaned_drug_name.append(list(replaced_drugs.values()))

    df["cleaned_drug_mapping"] = cleaned_drug_mapping
    df["cleaned_drug_name"] = cleaned_drug_name

    return df


def main():
    # Load PCDB database and abbreviation dictionary
    print("Loading data...")
    abbrv_dict_df = pd.read_csv(
        "data/PCDB/external_database_mapping_utils/document_abbreviation_conversion_dict.tsv",
        keep_default_na=False, dtype=str, sep="\t"
    )

    pcdb = pd.read_csv(
        "data/PCDB/preclinical_database_raw.tsv",
        sep="\t", keep_default_na=False, dtype=str
    )

    pcdb["drugs"] = pcdb["drugs"].apply(lambda x: ast.literal_eval(x) if x != "" else x)

    pcdb_with_abbrv_dict = pd.merge(
        left=pcdb, right=abbrv_dict_df,
        on="pmcid", how='outer'
    )
    pcdb_with_abbrv_dict["abbreviation_dict"] = pcdb_with_abbrv_dict["abbreviation_dict"].apply(ast.literal_eval)

    # Clean drug names
    print("Cleaning drug names...")
    pcdb_with_abbrv_dict = clean_and_normalize_drug_names(pcdb_with_abbrv_dict)

    # Collect all unique drugs
    print("Collecting unique drugs...")
    all_drugs = []
    for i, row in pcdb_with_abbrv_dict.iterrows():
        drug_names = row["cleaned_drug_name"]
        all_drugs.extend(drug_names)

    all_drugs_counter = Counter(all_drugs)
    unique_drugs = sorted(all_drugs_counter.keys(), key=lambda x: all_drugs_counter[x], reverse=True)
    if "" in unique_drugs:
        unique_drugs.remove("")
    print(f"Total unique drugs found: {len(unique_drugs)}")

    # Create drug dataframe
    pcdb_drug_df = pd.DataFrame({"drug": unique_drugs})

    # Load PubChem data and map CIDs
    print("\n--- PubChem CID Mapping ---")
    cid_df = load_pubchem_cids()
    pcdb_drug_df = get_pubchem_cids(pcdb_drug_df, cid_df)

    # Remove "control" from external_ids
    control_rows = pcdb_drug_df[pcdb_drug_df["drug"] == 'control']
    if len(control_rows) > 0:
        control_index = control_rows.index[0]
        pcdb_drug_df.at[control_index, "external_ids"] = {}

    print(pcdb_drug_df)

    # Load and map DrugBank IDs
    print("\n--- DrugBank ID Mapping ---")
    drugbank_LUT = load_drugbank_lookup_table()
    pcdb_drug_df = get_drugbank_ids(pcdb_drug_df, drugbank_LUT)
    print(pcdb_drug_df)

    # Load and map MeSH IDs
    print("\n--- MeSH ID Mapping ---")
    mesh_LUT = load_mesh_lookup_table()
    pcdb_drug_df = get_mesh_ids(pcdb_drug_df, mesh_LUT)
    print(pcdb_drug_df)

    # Load and map UMLS IDs
    print("\n--- UMLS ID Mapping ---")
    UMLS_LUT = load_umls_lookup_table()
    pcdb_drug_df = get_umls_ids(pcdb_drug_df, UMLS_LUT)
    print(pcdb_drug_df)

    # Load and map PubChem SIDs (for drugs without CIDs)
    print("\n--- PubChem SID Mapping ---")
    sid_df = load_pubchem_sids()
    pcdb_drug_df = get_pubchem_sids(pcdb_drug_df, sid_df)
    print(pcdb_drug_df)

    # Create drug external IDs lookup table
    pcdb_drug_external_ids_LUT = {}
    for i, row in pcdb_drug_df.iterrows():
        drug_name = row['drug']
        external_ids = row['external_ids']
        if isinstance(external_ids, str):
            external_ids = ast.literal_eval(external_ids)
        pcdb_drug_external_ids_LUT[drug_name] = external_ids

    # Map drugs in database to external IDs
    print("\n--- Mapping Database Drugs to External IDs ---")
    def get_external_ids_mapping(row):
        drug_names = row["cleaned_drug_name"]
        mapped_drugs = {}
        for drug in drug_names:
            if drug == "":
                return "", ""
            drug_normalized = normalize_greek_letters(drug)
            if drug_normalized in pcdb_drug_external_ids_LUT and len(pcdb_drug_external_ids_LUT[drug_normalized]) != 0:
                mapped_drugs[drug] = pcdb_drug_external_ids_LUT[drug_normalized]
        unmapped_drugs = []
        for d in drug_names:
            if d not in mapped_drugs:
                unmapped_drugs.append(d)

        if len(mapped_drugs) == 0:
            mapped_drugs = ""
        if len(unmapped_drugs) == 0:
            unmapped_drugs = ""

        return mapped_drugs, unmapped_drugs

    pcdb_with_abbrv_dict[["mapped_drugs", "unmapped_drugs"]] = pcdb_with_abbrv_dict.progress_apply(
        get_external_ids_mapping, axis=1, result_type="expand"
    )

    # Create result dataframe
    pcdb_mapped_drugs = pcdb_with_abbrv_dict[[
        'pmcid', 'cleaned_drug_mapping', 'cleaned_drug_name', 'mapped_drugs', 'unmapped_drugs'
    ]].copy()

    # Clean unmapped drugs
    print("\nCleaning unmapped drugs...")
    pcdb_mapped_drugs['unmapped_drugs'] = pcdb_mapped_drugs["unmapped_drugs"].apply(
        lambda x: clean_drug_names_list(x) if x != "" else x
    )

    # Print unmapped drug statistics
    # print("\n--- Unmapped Drug Statistics ---")
    # unmapped_counter = unmapped_drugs_stats(pcdb_mapped_drugs)
    # print(unmapped_counter)

    # Save results
    pcdb_mapped_drugs.to_csv("outputs/pcdb/mapped_drugs.tsv", sep="\t", index=False)
    print("\nResults saved to: outputs/pcdb/mapped_drugs.tsv")


if __name__ == "__main__":
    main()
