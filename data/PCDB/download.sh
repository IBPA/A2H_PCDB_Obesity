#!/bin/bash
set -euo pipefail

# Please download the UMLS data to the current folder before running this script:
# (Please register your account in UMLS first: https://www.nlm.nih.gov/research/umls/index.html)
# 1. https://download.nlm.nih.gov/umls/kss/2024AB/umls-2024AB-mrconso.zip
# 2. https://download.nlm.nih.gov/umls/kss/2025AB/umls-2025AB-metathesaurus-full.zip

echo 'Downloading external databases (look up tables for synonyms and ids & related helper files)...'

curl -fL "https://ucdavis.app.box.com/index.php?rm=box_download_shared_file&shared_name=8gvttyc2ea2gwvwhn9qqwe2txq4yhft1&file_id=f_2130807650205" --output PCDB_utils.zip
curl -fL "https://ucdavis.app.box.com/index.php?rm=box_download_shared_file&shared_name=ftb4p48cai84ks58wr2tpf98kiix082n&file_id=f_2130781103647" --output preclinical_database_raw.tsv

# Unzip the utils file, creates data/PCDB/external_database_mapping_utils
unzip PCDB_utils.zip

echo 'Downloading the relation-extraction predictions (frozen model outputs, see src/PCDB/relations/README.md)...'

# all_predictions.jsonl: one verdict per asserted relation from the google/gemma-4-E4B-it sweep (181 MB).
# publications.jsonl: all 134,565 PCDB publications with the sweep's input status and text source (10 MB).
# Together they let scripts/run_pcdb_relations.sh release rebuild the v2.0 tables without re-running the model.
mkdir -p pcdb_relations
curl -fL "https://ucdavis.box.com/shared/static/ztf7s0n2m1x54lswrp5d7azk8r3feo0i.jsonl" --output pcdb_relations/all_predictions.jsonl
curl -fL "https://ucdavis.box.com/shared/static/27dkeb7cx53jjau6zqu5vs9y2weho9sg.jsonl" --output pcdb_relations/publications.jsonl

# Unzip MRCONSO.RRF and MRSTY.RRF to the current folder data/PCDB/
unzip umls-2024AB-mrconso.zip
unzip umls-2025AB-metathesaurus-full.zip '*MRSTY.RRF'

# Create look-up tables for UMLS
python -m external_database_mapping_utils.external_database_lut.create_UMLS_lut \
  --mrconso 'UMLS/2024AB/META/MRCONSO.RRF' \
  --mrsty 'UMLS/2025AB/META/MRSTY.RRF' \
  --output-dir 'external_database_mapping_utils/external_database_lut'
