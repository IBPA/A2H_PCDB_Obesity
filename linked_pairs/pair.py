import pandas as pd
import ast

# Step 1: Load the TSV file
df = pd.read_csv("filtered_data.tsv", sep="\t")

# Step 2: Extract pmcid | nct_id pairs along with mesh_ids
output_rows = []

for _, row in df.iterrows():
    pmcid = row["pmcid"]
    
    try:
        disease_data = ast.literal_eval(str(row["matched_clinical_studies"]))
        mesh_data = ast.literal_eval(str(row["disease_external_ids"]))
    except Exception:
        continue

    # Collect MeSH IDs
    mesh_ids = set()
    for disease_entry in mesh_data:
        mesh_ids.update(disease_entry.get("mesh", []))

    # Collect pmcid|nct_id pairs
    for disease, drugs in disease_data.items():
        for nct_ids in drugs.values():
            for nct_id in nct_ids:
                for mesh_id in mesh_ids:
                    output_rows.append({
                        "mesh_id": mesh_id,
                        "matched_clinical_study": f"{pmcid}|{nct_id}"
                    })

# Step 3: Create a DataFrame from extracted pairs
pairs_df = pd.DataFrame(output_rows)

# Step 4: Count unique pmcid|nct_id pairs for each mesh_id
unique_counts_df = pairs_df.groupby("mesh_id")["matched_clinical_study"] \
                           .nunique() \
                           .reset_index() \
                           .rename(columns={"matched_clinical_study": "linked_pair_count"})

# Step 5: Sort by linked_pair_count in descending order
sorted_unique_counts_df = unique_counts_df.sort_values(by="linked_pair_count", ascending=False)

# Step 6: Save the result to a TSV file
sorted_unique_counts_df.to_csv("pairs_count.tsv", sep="\t", index=False)

print("Process complete. Output saved to 'pairs_count.tsv'")
