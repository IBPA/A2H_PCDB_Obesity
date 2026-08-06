import pandas as pd
import ast

# Load input files
database_df = pd.read_csv("database.tsv", sep="\t")
output_df = pd.read_csv("output.tsv", sep="\t")

# Step 1: Filter out rows with empty `matched_clinical_studies`
filtered_df = database_df.dropna(subset=["matched_clinical_studies"])

# Step 2: Only keep rows where disease_external_ids has valid MeSH IDs
def has_valid_mesh(entry):
    try:
        parsed = ast.literal_eval(entry)
        return any('mesh' in item and isinstance(item['mesh'], list) and item['mesh'] for item in parsed)
    except Exception:
        return False

filtered_df = filtered_df[filtered_df["disease_external_ids"].apply(has_valid_mesh)]

# Step 3: Only include rows where at least one MeSH ID is also in output.tsv
output_mesh_ids = set(output_df["mesh_id"].dropna().astype(str))

def mesh_id_in_output(entry):
    try:
        parsed = ast.literal_eval(entry)
        for item in parsed:
            if "mesh" in item:
                for mid in item["mesh"]:
                    if mid in output_mesh_ids:
                        return True
    except Exception:
        return False
    return False

filtered_df = filtered_df[filtered_df["disease_external_ids"].apply(mesh_id_in_output)]

# Save result to filtered_data.tsv
filtered_df.to_csv("filtered_data.tsv", sep="\t", index=False)
