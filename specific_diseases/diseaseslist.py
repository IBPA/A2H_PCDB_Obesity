import pandas as pd
import ast

# === Load the nodes.tsv file ===
nodes_df = pd.read_csv("nodes.tsv", sep="\t")

# === Step 1: Clean and parse child_nodes column safely ===
nodes_df['child_nodes'] = nodes_df['child_nodes'].apply(
    lambda x: ast.literal_eval(x) if pd.notnull(x) and x.strip() else []
)

# === Step 2: Build a set of all mesh_term values in the dataset for lookup ===
all_mesh_terms = set(nodes_df['mesh_term'])

# === Step 3: Define function to determine if a disease meets the new criteria ===
def keep_disease(row):
    children = row['child_nodes']
    if not children:
        return True  # No child nodes → keep
    for child in children:
        if child in all_mesh_terms:
            return False  # Has child in database → do not keep
    return True  # Has children, none in database → keep

# === Step 4: Filter rows accordingly ===
filtered_df = nodes_df[nodes_df.apply(keep_disease, axis=1)][['mesh_term', 'mesh_id']].drop_duplicates()

# === Step 5: Output to TSV file ===
filtered_df.to_csv("output.tsv", sep="\t", index=False)