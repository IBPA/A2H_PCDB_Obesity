  # Navigate to the R_LMER directory
cd src/obesity_LMER

DATA_FILE="../../data/obesity/obesity_a2h.csv"

# Run mice analysis
Rscript mice_weight_duration_analysis.R "$DATA_FILE" > mice_weight_duration_output.txt 2>&1

# Run rats analysis
Rscript rats_weight_duration_analysis.R "$DATA_FILE" > rats_weight_duration_output.txt 2>&1

# Run species comparison analysis
Rscript species_comparison_analysis.R "$DATA_FILE" > species_comparison_output.txt 2>&1