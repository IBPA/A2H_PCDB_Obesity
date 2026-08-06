# Disease-MeSH Clinical Pairing Toolkit

This toolkit processes biomedical research data to identify and analyze matched disease-clinical study pairs based on MeSH (Medical Subject Headings) identifiers. The scripts are used to extract, filter, and count disease-study pairings and generate useful TSV files for downstream analysis.

## 🔧 Prerequisites

- Python 3.7+
- Required Python libraries: `pandas`, `tqdm`

You can install dependencies using:

```bash
pip install pandas tqdm
```

---

## 📁 Files Overview

### `final.py`
- **Purpose:** Main processing script to extract, clean, and generate a TSV file (`final.tsv`) with structured clinical-study pairings using MeSH terms.
- **Input:** `filtered_data.tsv`
- **Output:** `final.tsv`

### `pair.py`
- **Purpose:** Computes the count of unique clinical-study pairings per disease based on MeSH ID.
- **Input:** `final.tsv`
- **Output:** `pair_count.tsv`

### `specific_data.py`
- **Purpose:** Further processes `final.tsv` by adding disease names (MeSH terms) using `output.tsv`, and removes rows where `mesh_term` is empty.
- **Input:** `final.tsv`, `output.tsv`
- **Output:** `filtered_data.tsv`

---

## 📊 Data Files

| File Name           | Description |
|---------------------|-------------|
| `filtered_data.tsv` | Preprocessed data where each row includes a MeSH term. Empty MeSH terms are removed. |
| `final.tsv`         | Structured dataset with disease-study pairings and cleaned metadata. |
| `output.tsv`        | Lookup table for mapping MeSH IDs to human-readable disease names. |
| `pair_count.tsv`    | Final count of unique matched clinical studies per disease (with `mesh_id` and `mesh_term`). |

---

## 🚀 How to Run

Run the scripts in the following order:

```bash
# Step 1: Extract and clean final TSV data
python final.py

# Step 2: Generate disease-study pair count
python pair.py

# Step 3: Add disease names and filter rows
python specific_data.py
```

---

## 📌 Notes

- Ensure all `.tsv` files are UTF-8 encoded and tab-separated.
- The MeSH mapping in `output.tsv` must contain valid `mesh_id` and corresponding `mesh_term`.
- The final output `pair_count.tsv` can be sorted for analysis or visualization.

---

## 📫 Contact

For questions, reach out to the project maintainer.
