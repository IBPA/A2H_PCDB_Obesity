#!/bin/bash
set -e

echo "Step 1/4 : Create disease entities..."
python -m src.PCDB.create_disease_entities

echo "Step 2/4 : Create drug entities..."
python -m src.PCDB.create_drug_entities

echo "Step 3/4 : Create animal model entities..."
python -m src.PCDB.create_animal_model_entities

echo "Step 4/4 : Create preclinical databases..."
python -m src.PCDB.build_pcdb