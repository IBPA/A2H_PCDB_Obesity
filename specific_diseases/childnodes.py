import pandas as pd
import ast
import xml.etree.ElementTree as ET
from collections import defaultdict

# === Step 1: Load and extract MeSH IDs from mini.tsv ===
df = pd.read_csv("database.tsv", sep="\t")

def extract_mesh_id(entry):
    try:
        parsed = ast.literal_eval(entry)
        for d in parsed:
            if 'mesh' in d:
                return d['mesh'][0]
    except:
        return None

df['mesh_id'] = df['disease_external_ids'].apply(extract_mesh_id)
df = df[df['mesh_id'].notnull()][['pmcid', 'disease_names', 'mesh_id']]

# === Step 2: Parse desc2025.xml to get MeSH ID → MeSH Term mapping ===
mesh_id_to_term = {}

tree = ET.parse("desc2025.xml")
root = tree.getroot()

for record in root.findall(".//DescriptorRecord"):
    descriptor_ui = record.findtext("DescriptorUI")
    descriptor_name = record.find("DescriptorName")
    if descriptor_ui and descriptor_name is not None:
        term = descriptor_name.findtext("String")
        mesh_id_to_term[descriptor_ui] = term

# Add mesh_term column using the local XML mapping
df['mesh_term'] = df['mesh_id'].map(mesh_id_to_term)

# === Step 3: Parse mtree file to build tree structure ===
term_to_tree_numbers = defaultdict(list)
tree_number_to_term = {}

with open("mtrees2025.bin", "r", encoding="utf-8") as f:
    for line in f:
        if ";" in line:
            term, tree = line.strip().split(";")
            term = term.strip()
            tree = tree.strip()
            term_to_tree_numbers[term].append(tree)
            tree_number_to_term[tree] = term

# === Step 4: Build tree_number → children mapping ===
tree_children = defaultdict(list)
tree_numbers_sorted = sorted(tree_number_to_term)

for i, parent in enumerate(tree_numbers_sorted):
    prefix = parent + "."
    for j in range(i + 1, len(tree_numbers_sorted)):
        child = tree_numbers_sorted[j]
        if child.startswith(prefix):
            tree_children[parent].append(child)
        elif not child.startswith(parent):
            break

# === Step 5: Map tree numbers and child nodes ===
df['tree_numbers'] = df['mesh_term'].map(lambda t: term_to_tree_numbers.get(t, []))
df['child_nodes'] = df['tree_numbers'].apply(
    lambda trees: [tree_number_to_term[child]
                   for tree in trees
                   for child in tree_children.get(tree, [])]
)

# === Step 6: Save to TSV ===
df.to_csv("nodes.tsv", sep="\t", index=False)
print("Output written to nodes.tsv")