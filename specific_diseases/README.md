 MeSH Disease Hierarchy Parser

This project processes a dataset of diseases with MeSH (Medical Subject Headings) identifiers to extract parent-child relationships and filter relevant diseases based on a curated MeSH hierarchy.

## Overview

The project is structured around two scripts:

### 1. `childnodes.py`

This script processes a TSV file named `database.tsv` that contains disease records, including external IDs. It:

- Extracts MeSH IDs from the `disease_external_ids` column.
- Queries `https://www.ncbi.nlm.nih.gov/mesh/{mesh_id}` to obtain MeSH XML data (Tree Numbers).
- Uses a local `mtree.txt` file to extract all descendant terms for each MeSH term found.
- Outputs a file `output.tsv` that maps each input disease to:
  - Its `mesh_id`
  - Its corresponding `mesh_term`
  - A list of its child node MeSH terms in the hierarchy.

### 2. `diseaseslist.py`

This script uses the output from the previous step and a `nodes.tsv` file which lists diseases with their child node mappings. It:

- Parses the `child_nodes` column from `nodes.tsv`.
- Filters diseases that have child nodes present in the overall MeSH term set.
- Produces a final disease list for further analysis or visualization.

## Input Files

- `database.tsv`: Contains raw disease data with external identifiers.
- `mtree.txt`: A text file representing the full MeSH hierarchical tree.
- `nodes.tsv`: Contains processed disease nodes with child relationships.

## Output Files

- `output.tsv`: Result from `childnodes.py`, listing each disease's MeSH term and its child nodes.
- Final disease list printed or saved by `diseaseslist.py` (typically narrowed to diseases with defined child nodes in MeSH).

## Requirements

- Python 3.7+
- pandas
- requests (if making live queries to NCBI)
- Internet connection (for fetching MeSH pages unless pre-downloaded)

## Usage

1. Place `database.tsv` and `mtree.txt` in the same directory as `childnodes.py`, then run:
   ```bash
   python childnodes.py
   ```

2. Place `nodes.tsv` in the same directory as `diseaseslist.py`, then run:
   ```bash
   python diseaseslist.py
   ```

## Purpose

This code is useful for:

- Building a MeSH-based disease ontology tree.
- Filtering and structuring biomedical datasets according to hierarchical disease relationships.
