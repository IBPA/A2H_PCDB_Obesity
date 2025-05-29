import pandas as pd

# Load your input files
pair_count_df = pd.read_csv("pair_count.tsv", sep="\t")
output_df = pd.read_csv("output.tsv", sep="\t")

# Merge on 'mesh_id' to bring in the 'mesh_term'
merged_df = pair_count_df.merge(output_df, on="mesh_id", how="left")

# Remove rows where 'mesh_term' is missing or empty
filtered_df = merged_df.dropna(subset=["mesh_term"])
filtered_df = filtered_df[filtered_df["mesh_term"].str.strip() != ""]

# Reorder columns if needed
filtered_df = filtered_df[["mesh_id", "mesh_term", "linked_pair_count"]]

# Save the result
filtered_df.to_csv("final.tsv", sep="\t", index=False)

print("Filtered file saved as 'final.tsv'")
