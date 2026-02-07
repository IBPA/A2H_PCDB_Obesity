#!/bin/bash
set -e

echo "Step 1/4: Running disease external DB mapping..."
python -m src.PCDB.disease_external_db_mapping

echo "Step 2/4: Running drug external DB mapping..."
python -m src.PCDB.drug_external_db_mapping

echo "Step 3/4: Running animal model external DB mapping..."
python -m src.PCDB.animal_model_external_db_mapping

echo "Step 4/4: Integrating mapping results..."
python -m src.PCDB.integrate_mapping_results

echo "All steps completed."
