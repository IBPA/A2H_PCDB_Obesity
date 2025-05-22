import pandas as pd

# Load only the first and third columns (index 0 and 2)
df = pd.read_csv("bc_data.csv", usecols=[0, 2], header=1)
df.columns = ['col_label', 'translation_success']

# Drop non-numeric entries in the translation_success column
df = df[pd.to_numeric(df['translation_success'], errors='coerce').notna()]
df['translation_success'] = df['translation_success'].astype(float).astype(int)

# Calculate percentage of 0s and 1s per col_label
result = (
    df.groupby('col_label')['translation_success']
      .value_counts(normalize=True)
      .unstack(fill_value=0) * 100
)

# Rename columns for clarity
result.columns = ['Translation Fail', 'Translation Success']

# Reset index and rename col_label
result = result.reset_index().rename(columns={'col_label': 'Cluster'})

# Sort by Translation Success in descending order
result = result.sort_values(by='Translation Success', ascending=False)

# Keep only Cluster and Translation Success, and format the percentage
result = result[['Cluster', 'Translation Success']]
result['Translation Success'] = result['Translation Success'].map(lambda x: f"{x:.1f} %")

# Export to CSV
result.to_csv("ts_list.csv", index=False)
