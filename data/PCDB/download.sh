#!/bin/bash

# Please download the UMLS data to the current folder before running this script:
# (Please register your account in UMLS first: https://www.nlm.nih.gov/research/umls/index.html)
# 1. https://download.nlm.nih.gov/umls/kss/2024AB/umls-2024AB-mrconso.zip
# 2. https://download.nlm.nih.gov/umls/kss/2025AB/umls-2025AB-metathesaurus-full.zip

echo 'Downloading external databases (look up tables for synonyms and ids & related helper files)...'
curl -L  https://app.box.com/s/8gvttyc2ea2gwvwhn9qqwe2txq4yhft1 --output PCDB_utils.zip

# Unzip the utils file, creates data/PCDB/external_database_mapping_utils
unzip PCDB_utils.zip

# Unzip MRCONSO.RRF and MRSTY.RRF to the current folder data/PCDB/
unzip umls-2024AB-mrconso.zip
unzip umls-2025AB-metathesaurus-full.zip '*MRSTY.RRF'

# Create look-up tables for UMLS
python -m external_database_mapping_utils.external_database_lut.create_UMLS_lut.py \
  --mrconso 'UMLS/2024AB/META/MRCONSO.RRF' \
  --mrsty 'UMLS/2025AB/META/MRSTY.RRF' \
  --output-dir 'external_database_mapping_utils/external_database_lut'
