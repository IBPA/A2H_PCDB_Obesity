#!/bin/bash
set -e

python -m src.PCDB.disease_external_db_mapping

python -m src.PCDB.drug_external_db_mapping

python -m src.PCDB.animal_model_external_db_mapping

python -m src.PCDB.integrate_mapping_results

echo "Step 1/5 : Create disease entities..."
python -m src.PCDB.create_disease_entities

echo "Step 2/5 : Create drug entities..."
python -m src.PCDB.create_drug_entities

echo "Step 3/5 : Create animal model entities..."
python -m src.PCDB.create_animal_model_entities

echo "Step 4/5 : Clean synonyms and normalize entity IDs..."
python -m src.PCDB.clean_synonyms

echo "Step 5/5 : Create preclinical databases..."
python -m src.PCDB.build_pcdb