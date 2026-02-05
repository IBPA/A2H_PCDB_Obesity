#!/bin/bash
# Example: run_all.sh

INPUT_CSV=$1 
OUTPUT_DIR=$2 #Output Directory of PMC Multimedia
OUTCOME_MEASURE_INPUT=$3 
FILTER_CSV_NAME=$4 #CSV from filtering step
ZIP_FNAME=$5 #Zip File Name to be written to 

echo "$INPUT_FILE"
set -e 

echo "Running Article Multimedia Extraction" 
python3 scrape_figures.py "$INPUT_CSV" "$OUTPUT_DIR" 

echo "Filtering for Outcome Measure on Multimedia..." 
python3 filter.py "$OUTPUT_DIR" "$OUTCOME_MEASURE_INPUT" "$FILTER_CSV_NAME" 

echo "Compressing filtered figures into ZIP file..." 
python3 compress.py temp "$FILTER_CSV_NAME" "$ZIP_FNAME" 
