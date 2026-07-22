import pandas as pd
import ast
import re
import argparse
from pathlib import Path
from tqdm import tqdm

tqdm.pandas()

def create_disease_names_and_disease_eids(row):
    mapped_diseases = row["mapped_diseases"]
    if mapped_diseases == "":
        return "", ""
    if isinstance(mapped_diseases, str):
        mapped_diseases = ast.literal_eval(mapped_diseases)
    disease_names = []
    disease_eids = []
    for name, ids in mapped_diseases.items():
        disease_names.append(name)
        for db_name, db_ids in ids.items():
            ids[db_name] = list(set(db_ids))
        disease_eids.append(ids)
    return disease_names, disease_eids


def create_drug_names_and_drug_eids(row):
    mapped_drugs = row["mapped_drugs"]
    if mapped_drugs == "":
        return "", ""
    if isinstance(mapped_drugs, str):
        mapped_drugs = ast.literal_eval(mapped_drugs)
    drug_names = []
    drug_eids = []
    for name, ids in mapped_drugs.items():
        drug_names.append(name)
        drug_eids.append(ids)
    return drug_names, drug_eids


def create_animal_columns(row):
    mapped_animal = row["mapped_animals"]
    if mapped_animal == "":
        return "", "", "", ""
    if isinstance(mapped_animal, str):
        mapped_animal = ast.literal_eval(mapped_animal)
    strain = mapped_animal["strain"]
    species = mapped_animal["species"]
    total_subject_size = mapped_animal["total_subject_size"]
    eid = {"mesh": mapped_animal["mesh"]}
    return strain, species, total_subject_size, eid


def clean_total_subject_size(subject_size_str):
    def _evaluate_math_expression(expr):
        parts = expr.split("=")
        try:
            if len(parts) == 2:
                right = parts[1].strip()
                return right
            return str(eval(expr))
        except:
            return ""

    def _is_math_equation(m):
        if not re.compile(r"[A-Za-z]").search(m):
            if ('+' in m or "*" in m) and ('-' not in m):
                return True
        return False

    if subject_size_str == "":
        return ""
    pattern = re.compile(r"^\d+$|^\d+\s*-\s*\d+$")
    matched = pattern.match(subject_size_str)

    cleaned_tss = subject_size_str
    if not matched:
        if _is_math_equation(subject_size_str):
            cleaned_tss = _evaluate_math_expression(subject_size_str)
            # print(subject_size_str, " -> ", cleaned_tss)
        else:
            cleaned_tss = ""

    return cleaned_tss


def create_animal_model_columns(df: pd.DataFrame):
    df["mapped_animals"] = df["mapped_animals"].apply(
        lambda x: ast.literal_eval(x) if (isinstance(x, str) and x != "") else x
    )
    df_exploded = df.explode(column="mapped_animals")
    df_exploded[["animal_strain", "animal_species", "total_subject_size", "animal_external_ids"]] = (
        df_exploded.apply(create_animal_columns, axis=1, result_type='expand')
    )
    print("Clean total subject size. Only number, range of numbers and valid math equations (after evaluated) are kept.")
    df_exploded["total_subject_size"] = df_exploded["total_subject_size"].apply(clean_total_subject_size)
    return df_exploded



def main(input_dir: str, output_dir: str):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    diseases = pd.read_csv(input_path / "mapped_diseases.tsv", sep="\t", dtype=str, keep_default_na=False)
    drugs = pd.read_csv(input_path / "mapped_drugs.tsv", sep="\t", dtype=str, keep_default_na=False)
    animal_models = pd.read_csv(input_path / "mapped_animals.tsv", sep="\t", dtype=str, keep_default_na=False)
    document_dates = pd.read_csv("data/PCDB/external_database_mapping_utils/document_dates.tsv", sep="\t", dtype=str, keep_default_na=False)

    # Merge
    drugs = drugs[["pmcid", "mapped_drugs", "unmapped_drugs"]]
    animal_models = animal_models[["pmcid", "mapped_animals", "unmapped_animals"]]

    a2h_db = pd.merge(left=diseases, right=drugs, on='pmcid', how='outer')
    a2h_db = pd.merge(left=a2h_db, right=animal_models, on='pmcid', how='outer')

    # Create structured columns
    print("Processing Disease entities")
    a2h_db[["disease_names", "disease_external_ids"]] = a2h_db.progress_apply(
        create_disease_names_and_disease_eids, axis=1, result_type='expand'
    )
    print("Processing Drug entities")
    a2h_db[["drug_names", "drug_external_ids"]] = a2h_db.progress_apply(
        create_drug_names_and_drug_eids, axis=1, result_type='expand'
    )
    print("Processing Animal Model entities")
    a2h_db = create_animal_model_columns(a2h_db)

    # Merge document dates
    a2h_db = pd.merge(left=a2h_db, right=document_dates[['pmcid', 'date']], on='pmcid', how='left')

    print("Columns:", list(a2h_db.columns))

    # Split into mapped and unmapped
    a2h_db_mapped = a2h_db[['pmcid', "date", 'link', "probability", "TITLE", "disease_names",
                             "disease_external_ids", "drug_names", "drug_external_ids", "animal_strain",
                             "animal_species", "total_subject_size", "animal_external_ids"]]

    a2h_db_unmapped = a2h_db[['pmcid', "unmapped_diseases", "unmapped_drugs", "unmapped_animals"]]
    a2h_db_unmapped = a2h_db_unmapped.drop_duplicates()

    a2h_db_mapped = a2h_db_mapped.rename(columns={"probability": "confidence_score"})
    a2h_db_unmapped = a2h_db_unmapped.rename(columns={
        "disease": "extracted_diseases_from_LLM",
        "drugs": "extracted_drugs_from_LLM",
        "animals": "extracted_animals_from_LLM",
        "probability": "confidence_score",
    })

    # Save to Excel
    output_file = output_path / "pcdb_mapped.xlsx"
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        a2h_db_mapped.to_excel(writer, sheet_name='pcdb_mapped', index=False)
        a2h_db_unmapped.to_excel(writer, sheet_name='unmapped_extractions', index=False)
    print(f"Output saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Integrate mapping results into the A2H database.")
    default_dir = str(Path(__file__).resolve().parent.parent.parent / "outputs" / "pcdb")
    parser.add_argument("--input_dir", type=str, default=default_dir,
                        help="Directory containing mapped TSV files.")
    parser.add_argument("--output_dir", type=str, default=default_dir,
                        help="Directory to save the output Excel file.")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
