################################################################################
# MICE: Weight and Duration Effects Analysis
# Analyzes weight, duration, and weight×duration effects for liraglutide and
# semaglutide in mouse DIO models
################################################################################

library(dplyr)
library(readr)
library(lme4)
library(lmerTest)
library(performance)  # For VIF calculation
library(ggplot2)
library(gridExtra)

## =============================================================================
## Load and Prepare Data
## =============================================================================

cat("\n")
cat("================================================================================\n")
cat("MICE: WEIGHT AND DURATION EFFECTS ANALYSIS\n")
cat("================================================================================\n\n")

# Get data file path from command line argument, or use default
args <- commandArgs(trailingOnly = TRUE)
data_file <- if (length(args) >= 1) args[1] else "../../obesity_a2h.csv"

cat(sprintf("Reading data from: %s\n\n", data_file))
df <- read_csv(data_file)

df <- df %>%
  rename(
    NCT_Number = `NCT Number`,
    pre_dose_mgkg = `preclinical_dosage_amount_value(mg/kg)`,
    pre_dur_days = `preclinical_dosage_duration(days)`,
    pre_wt_treat_g = `preclinical_animal_weight_before_treatment(grams)`,
    pre_age_treat_d = `preclinical_animal_age_before_treatment(days)`,
    pre_sex = preclinical_animal_sex,
    pre_strain = preclinical_animal_strain,
    clin_dose_mg = `clinical_dosage_amount_value(mg)`,
    clin_dur_days = `clinical_dosage_duration(days)`,
    y = translation_outcome
  ) %>%
  mutate(
    pre_wt_treat_g_missing = as.integer(is.na(pre_wt_treat_g)),
    pre_age_treat_d_missing = as.integer(is.na(pre_age_treat_d)),
    pre_wt_treat_g = ifelse(is.na(pre_wt_treat_g), median(pre_wt_treat_g, na.rm=TRUE), pre_wt_treat_g),
    pre_age_treat_d = ifelse(is.na(pre_age_treat_d), median(pre_age_treat_d, na.rm=TRUE), pre_age_treat_d),
    preclinical_animal_species = factor(preclinical_animal_species),
    pre_sex = factor(pre_sex),
    pre_strain = factor(pre_strain),
    intervention = factor(intervention),
    pmcid = factor(pmcid),
    NCT_Number = factor(NCT_Number),

    # Classify disease model
    model_type = case_when(
      grepl("DIO|diet|high-fat|HFD", preclinical_disease_model, ignore.case = TRUE) ~ "DIO",
      grepl("ob/ob|db/db|genetic|leptin", preclinical_disease_model, ignore.case = TRUE) ~ "Genetic",
      TRUE ~ "Other"
    )
  )

# Numeric columns for standardization
num_cols <- c("pre_dose_mgkg", "pre_dur_days", "pre_wt_treat_g", "pre_age_treat_d",
              "clin_dose_mg", "clin_dur_days", "clinical_sample_size")

# Helper function to format p-values with scientific notation
format_pvalue <- function(p) {
  if (is.na(p)) {
    return("NA")
  } else if (p < 0.001) {
    return(sprintf("%.2e", p))
  } else {
    return(sprintf("%.4f", p))
  }
}

# Helper function to add significance stars
get_sig_stars <- function(p) {
  if (is.na(p)) {
    return("")
  } else if (p < 0.001) {
    return("***")
  } else if (p < 0.01) {
    return("**")
  } else if (p < 0.05) {
    return("*")
  } else {
    return("NS")
  }
}

# Helper function to print VIF values
print_vif <- function(model, model_name = "Model") {
  cat("\nVariance Inflation Factors (VIF) for", model_name, ":\n")
  cat("---------------------------------------------\n")
  tryCatch({
    vif_result <- check_collinearity(model)
    # Print the result directly with its formatting
    print(vif_result)
    cat("\nInterpretation: VIF < 5 (low), 5-10 (moderate), >10 (severe multicollinearity)\n")
  }, error = function(e) {
    cat("  VIF calculation not available for this model\n")
  })
  cat("\n")
}

## =============================================================================
## SECTION 1: LIRAGLUTIDE MICE (DIO ONLY)
## =============================================================================

cat("\n")
cat("================================================================================\n")
cat("SECTION 1: LIRAGLUTIDE MICE (DIO MODELS)\n")
cat("================================================================================\n\n")

df_lira_mice_dio <- df %>%
  filter(intervention == "liraglutide",
         preclinical_animal_species == "mice",
         model_type == "DIO",
         !is.na(y),
         !is.na(pre_wt_treat_g),
         !is.na(pre_dur_days)) %>%
  droplevels()

cat(sprintf("Sample size: %d observations from %d studies\n\n", 
            nrow(df_lira_mice_dio), n_distinct(df_lira_mice_dio$pmcid)))

# Standardize predictors
df_lira_mice_dio[paste0(num_cols, "_z")] <- scale(df_lira_mice_dio[num_cols])

## -----------------------------------------------------------------------------
## 1.1: Main Effects (Duration + Weight)
## -----------------------------------------------------------------------------

cat("1.1: MAIN EFFECTS MODEL\n")
cat("-----------------------\n")

# Build formula based on available factor levels
formula_parts <- c("y ~ pre_dur_days_z", "pre_wt_treat_g_z")
if (n_distinct(df_lira_mice_dio$pre_sex, na.rm = TRUE) > 1) {
  formula_parts <- c(formula_parts, "pre_sex")
  cat("Including sex as fixed effect\n")
} else {
  cat("Sex excluded (only 1 level)\n")
}
if (n_distinct(df_lira_mice_dio$pre_strain, na.rm = TRUE) > 1) {
  formula_parts <- c(formula_parts, "pre_strain")
  cat("Including strain as fixed effect\n")
} else {
  cat("Strain excluded (only 1 level)\n")
}
if (n_distinct(df_lira_mice_dio$preclinical_administration_route, na.rm = TRUE) > 1) {
  formula_parts <- c(formula_parts, "preclinical_administration_route")
  cat("Including administration route as fixed effect\n")
} else {
  cat("Administration route excluded (only 1 level)\n")
}
formula_parts <- c(formula_parts,
                  "pre_age_treat_d_z", "pre_age_treat_d_missing",
                  "pre_wt_treat_g_missing", "pre_dose_mgkg_z",
                  "clin_dose_mg_z", "clin_dur_days_z", "clinical_sample_size_z",
                  "(1 | NCT_Number)")
formula_str <- paste(formula_parts, collapse = " + ")

model_lira_main <- lmer(
  as.formula(formula_str),
  data = df_lira_mice_dio,
  REML = TRUE,
  control = lmerControl(optimizer = "bobyqa")
)

coef_lira_main <- summary(model_lira_main)$coefficients
ci_lira_main <- confint(model_lira_main, method = "Wald", level = 0.95)

cat(sprintf("Duration effect:  β = %7.3f, SE = %.3f, p = %s %s\n",
            coef_lira_main["pre_dur_days_z", "Estimate"],
            coef_lira_main["pre_dur_days_z", "Std. Error"],
            format_pvalue(coef_lira_main["pre_dur_days_z", "Pr(>|t|)"]),
            get_sig_stars(coef_lira_main["pre_dur_days_z", "Pr(>|t|)"])))
cat(sprintf("                  95%% CI: [%.3f, %.3f]\n",
            ci_lira_main["pre_dur_days_z", 1],
            ci_lira_main["pre_dur_days_z", 2]))

cat(sprintf("Weight effect:    β = %7.3f, SE = %.3f, p = %s %s\n",
            coef_lira_main["pre_wt_treat_g_z", "Estimate"],
            coef_lira_main["pre_wt_treat_g_z", "Std. Error"],
            format_pvalue(coef_lira_main["pre_wt_treat_g_z", "Pr(>|t|)"]),
            get_sig_stars(coef_lira_main["pre_wt_treat_g_z", "Pr(>|t|)"])))
cat(sprintf("                  95%% CI: [%.3f, %.3f]\n",
            ci_lira_main["pre_wt_treat_g_z", 1],
            ci_lira_main["pre_wt_treat_g_z", 2]))

# VIF details suppressed for brevity

## -----------------------------------------------------------------------------
## 1.2: Weight × Duration Interaction
## -----------------------------------------------------------------------------

cat("1.2: WEIGHT × DURATION INTERACTION\n")
cat("----------------------------------\n")

# Build formula based on available factor levels
formula_parts <- c("y ~ pre_dur_days_z * pre_wt_treat_g_z")
if (n_distinct(df_lira_mice_dio$pre_sex, na.rm = TRUE) > 1) {
  formula_parts <- c(formula_parts, "pre_sex")
}
if (n_distinct(df_lira_mice_dio$pre_strain, na.rm = TRUE) > 1) {
  formula_parts <- c(formula_parts, "pre_strain")
}
if (n_distinct(df_lira_mice_dio$preclinical_administration_route, na.rm = TRUE) > 1) {
  formula_parts <- c(formula_parts, "preclinical_administration_route")
}
formula_parts <- c(formula_parts,
                  "pre_age_treat_d_z", "pre_age_treat_d_missing",
                  "pre_wt_treat_g_missing", "pre_dose_mgkg_z",
                  "clin_dose_mg_z", "clin_dur_days_z", "clinical_sample_size_z",
                  "(1 | NCT_Number)")
formula_str <- paste(formula_parts, collapse = " + ")

model_lira_interaction <- lmer(
  as.formula(formula_str),
  data = df_lira_mice_dio,
  REML = TRUE,
  control = lmerControl(optimizer = "bobyqa")
)

coef_lira_int <- summary(model_lira_interaction)$coefficients
ci_lira_int <- confint(model_lira_interaction, method = "Wald", level = 0.95)

cat(sprintf("Duration main:    β = %7.3f, SE = %.3f, p = %s\n",
            coef_lira_int["pre_dur_days_z", "Estimate"],
            coef_lira_int["pre_dur_days_z", "Std. Error"],
            format_pvalue(coef_lira_int["pre_dur_days_z", "Pr(>|t|)"])))

cat(sprintf("Weight main:      β = %7.3f, SE = %.3f, p = %s\n",
            coef_lira_int["pre_wt_treat_g_z", "Estimate"],
            coef_lira_int["pre_wt_treat_g_z", "Std. Error"],
            format_pvalue(coef_lira_int["pre_wt_treat_g_z", "Pr(>|t|)"])))

cat(sprintf("Interaction:      β = %7.3f, SE = %.3f, p = %s %s\n",
            coef_lira_int["pre_dur_days_z:pre_wt_treat_g_z", "Estimate"],
            coef_lira_int["pre_dur_days_z:pre_wt_treat_g_z", "Std. Error"],
            format_pvalue(coef_lira_int["pre_dur_days_z:pre_wt_treat_g_z", "Pr(>|t|)"]),
            get_sig_stars(coef_lira_int["pre_dur_days_z:pre_wt_treat_g_z", "Pr(>|t|)"])))
cat(sprintf("                  95%% CI: [%.3f, %.3f]\n\n",
            ci_lira_int["pre_dur_days_z:pre_wt_treat_g_z", 1],
            ci_lira_int["pre_dur_days_z:pre_wt_treat_g_z", 2]))

if (coef_lira_int["pre_dur_days_z:pre_wt_treat_g_z", "Pr(>|t|)"] < 0.05) {
  cat("→ SIGNIFICANT interaction: Weight effect varies by duration\n\n")
} else {
  cat("→ No significant interaction: Weight and duration effects are additive\n\n")
}

## =============================================================================
## SECTION 2: SEMAGLUTIDE MICE (DIO ONLY)
## =============================================================================

cat("\n")
cat("================================================================================\n")
cat("SECTION 2: SEMAGLUTIDE MICE (DIO MODELS)\n")
cat("================================================================================\n\n")

df_sema_mice_dio <- df %>%
  filter(intervention == "semaglutide",
         preclinical_animal_species == "mice",
         model_type == "DIO",
         !is.na(y),
         !is.na(pre_wt_treat_g),
         !is.na(pre_dur_days)) %>%
  droplevels()

cat(sprintf("Sample size: %d observations from %d studies\n\n", 
            nrow(df_sema_mice_dio), n_distinct(df_sema_mice_dio$pmcid)))

# Standardize predictors
df_sema_mice_dio[paste0(num_cols, "_z")] <- scale(df_sema_mice_dio[num_cols])

## -----------------------------------------------------------------------------
## 2.1: Main Effects (Duration + Weight)
## -----------------------------------------------------------------------------

cat("2.1: MAIN EFFECTS MODEL\n")
cat("-----------------------\n")

# Build formula based on available factor levels
# Exclude route (highest adjusted VIF = 11.09, not target variable)
# Keep strain to check if VIF improves after removing route
formula_parts <- c("y ~ pre_dur_days_z", "pre_wt_treat_g_z")
if (n_distinct(df_sema_mice_dio$pre_sex, na.rm = TRUE) > 1) {
  formula_parts <- c(formula_parts, "pre_sex")
  cat("Including sex as fixed effect\n")
} else {
  cat("Sex excluded (only 1 level)\n")
}
if (n_distinct(df_sema_mice_dio$pre_strain, na.rm = TRUE) > 1) {
  formula_parts <- c(formula_parts, "pre_strain")
  cat("Including strain as fixed effect\n")
} else {
  cat("Strain excluded (only 1 level)\n")
}
cat("Administration route excluded (adjusted VIF = 11.09, not target variable)\n")
formula_parts <- c(formula_parts,
                  "pre_age_treat_d_z", "pre_age_treat_d_missing",
                  "pre_wt_treat_g_missing", "pre_dose_mgkg_z",
                  "clin_dose_mg_z", "clin_dur_days_z", "clinical_sample_size_z",
                  "(1 | NCT_Number)")
formula_str <- paste(formula_parts, collapse = " + ")

model_sema_main <- lmer(
  as.formula(formula_str),
  data = df_sema_mice_dio,
  REML = TRUE,
  control = lmerControl(optimizer = "bobyqa")
)

coef_sema_main <- summary(model_sema_main)$coefficients
ci_sema_main <- confint(model_sema_main, method = "Wald", level = 0.95)

cat(sprintf("Duration effect:  β = %7.3f, SE = %.3f, p = %s %s\n",
            coef_sema_main["pre_dur_days_z", "Estimate"],
            coef_sema_main["pre_dur_days_z", "Std. Error"],
            format_pvalue(coef_sema_main["pre_dur_days_z", "Pr(>|t|)"]),
            get_sig_stars(coef_sema_main["pre_dur_days_z", "Pr(>|t|)"])))
cat(sprintf("                  95%% CI: [%.3f, %.3f]\n",
            ci_sema_main["pre_dur_days_z", 1],
            ci_sema_main["pre_dur_days_z", 2]))

cat(sprintf("Weight effect:    β = %7.3f, SE = %.3f, p = %s %s\n",
            coef_sema_main["pre_wt_treat_g_z", "Estimate"],
            coef_sema_main["pre_wt_treat_g_z", "Std. Error"],
            format_pvalue(coef_sema_main["pre_wt_treat_g_z", "Pr(>|t|)"]),
            get_sig_stars(coef_sema_main["pre_wt_treat_g_z", "Pr(>|t|)"])))
cat(sprintf("                  95%% CI: [%.3f, %.3f]\n",
            ci_sema_main["pre_wt_treat_g_z", 1],
            ci_sema_main["pre_wt_treat_g_z", 2]))

# VIF details suppressed for brevity

## -----------------------------------------------------------------------------
## 2.2: Weight × Duration Interaction
## -----------------------------------------------------------------------------

cat("2.2: WEIGHT × DURATION INTERACTION\n")
cat("----------------------------------\n")

# Build formula based on available factor levels
# Exclude route (highest adjusted VIF = 11.09, not target variable)
formula_parts <- c("y ~ pre_dur_days_z * pre_wt_treat_g_z")
if (n_distinct(df_sema_mice_dio$pre_sex, na.rm = TRUE) > 1) {
  formula_parts <- c(formula_parts, "pre_sex")
}
if (n_distinct(df_sema_mice_dio$pre_strain, na.rm = TRUE) > 1) {
  formula_parts <- c(formula_parts, "pre_strain")
}
formula_parts <- c(formula_parts,
                  "pre_age_treat_d_z", "pre_age_treat_d_missing",
                  "pre_wt_treat_g_missing", "pre_dose_mgkg_z",
                  "clin_dose_mg_z", "clin_dur_days_z", "clinical_sample_size_z",
                  "(1 | NCT_Number)")
formula_str <- paste(formula_parts, collapse = " + ")

model_sema_interaction <- lmer(
  as.formula(formula_str),
  data = df_sema_mice_dio,
  REML = TRUE,
  control = lmerControl(optimizer = "bobyqa")
)

coef_sema_int <- summary(model_sema_interaction)$coefficients
ci_sema_int <- confint(model_sema_interaction, method = "Wald", level = 0.95)

cat(sprintf("Duration main:    β = %7.3f, SE = %.3f, p = %s\n",
            coef_sema_int["pre_dur_days_z", "Estimate"],
            coef_sema_int["pre_dur_days_z", "Std. Error"],
            format_pvalue(coef_sema_int["pre_dur_days_z", "Pr(>|t|)"])))

cat(sprintf("Weight main:      β = %7.3f, SE = %.3f, p = %s\n",
            coef_sema_int["pre_wt_treat_g_z", "Estimate"],
            coef_sema_int["pre_wt_treat_g_z", "Std. Error"],
            format_pvalue(coef_sema_int["pre_wt_treat_g_z", "Pr(>|t|)"])))

cat(sprintf("Interaction:      β = %7.3f, SE = %.3f, p = %s %s\n",
            coef_sema_int["pre_dur_days_z:pre_wt_treat_g_z", "Estimate"],
            coef_sema_int["pre_dur_days_z:pre_wt_treat_g_z", "Std. Error"],
            format_pvalue(coef_sema_int["pre_dur_days_z:pre_wt_treat_g_z", "Pr(>|t|)"]),
            get_sig_stars(coef_sema_int["pre_dur_days_z:pre_wt_treat_g_z", "Pr(>|t|)"])))
cat(sprintf("                  95%% CI: [%.3f, %.3f]\n\n",
            ci_sema_int["pre_dur_days_z:pre_wt_treat_g_z", 1],
            ci_sema_int["pre_dur_days_z:pre_wt_treat_g_z", 2]))

if (coef_sema_int["pre_dur_days_z:pre_wt_treat_g_z", "Pr(>|t|)"] < 0.05) {
  cat("→ SIGNIFICANT interaction: Weight effect varies by duration\n\n")
} else {
  cat("→ No significant interaction: Weight and duration effects are additive\n\n")
}

## =============================================================================
## SECTION 3: CROSS-DRUG COMPARISON (DRUG × WEIGHT × DURATION)
## =============================================================================

cat("\n")
cat("================================================================================\n")
cat("SECTION 3: CROSS-DRUG COMPARISON (MICE DIO MODELS)\n")
cat("================================================================================\n\n")

# Combine datasets
df_both_mice_dio <- rbind(
  df_lira_mice_dio %>% mutate(drug = "liraglutide"),
  df_sema_mice_dio %>% mutate(drug = "semaglutide")
)

cat(sprintf("Combined sample: %d observations from %d studies\n", 
            nrow(df_both_mice_dio), n_distinct(df_both_mice_dio$pmcid)))
cat(sprintf("  Liraglutide: %d obs from %d studies\n", 
            nrow(df_lira_mice_dio), n_distinct(df_lira_mice_dio$pmcid)))
cat(sprintf("  Semaglutide: %d obs from %d studies\n\n", 
            nrow(df_sema_mice_dio), n_distinct(df_sema_mice_dio$pmcid)))

# Re-standardize on combined dataset
df_both_mice_dio[paste0(num_cols, "_z")] <- scale(df_both_mice_dio[num_cols])

## -----------------------------------------------------------------------------
## 3.1: Test Drug × Weight Interaction
## -----------------------------------------------------------------------------

cat("3.1: DRUG × WEIGHT INTERACTION\n")
cat("------------------------------\n")

# Build formula based on available factor levels
formula_parts <- c("y ~ drug * pre_wt_treat_g_z", "pre_dur_days_z")
if (n_distinct(df_both_mice_dio$pre_sex, na.rm = TRUE) > 1) {
  formula_parts <- c(formula_parts, "pre_sex")
}
if (n_distinct(df_both_mice_dio$pre_strain, na.rm = TRUE) > 1) {
  formula_parts <- c(formula_parts, "pre_strain")
}
formula_parts <- c(formula_parts,
                  "pre_age_treat_d_z", "pre_age_treat_d_missing",
                  "pre_wt_treat_g_missing", "pre_dose_mgkg_z",
                  "clin_dose_mg_z", "clin_dur_days_z", "clinical_sample_size_z",
                  "(1 | NCT_Number)")
formula_str <- paste(formula_parts, collapse = " + ")

model_drug_x_weight <- lmer(
  as.formula(formula_str),
  data = df_both_mice_dio,
  REML = TRUE,
  control = lmerControl(optimizer = "bobyqa")
)

coef_drug_x_wt <- summary(model_drug_x_weight)$coefficients

cat(sprintf("Drug × Weight:    β = %7.3f, SE = %.3f, p = %s %s\n\n",
            coef_drug_x_wt["drugsemaglutide:pre_wt_treat_g_z", "Estimate"],
            coef_drug_x_wt["drugsemaglutide:pre_wt_treat_g_z", "Std. Error"],
            format_pvalue(coef_drug_x_wt["drugsemaglutide:pre_wt_treat_g_z", "Pr(>|t|)"]),
            get_sig_stars(coef_drug_x_wt["drugsemaglutide:pre_wt_treat_g_z", "Pr(>|t|)"])))

if (coef_drug_x_wt["drugsemaglutide:pre_wt_treat_g_z", "Pr(>|t|)"] < 0.05) {
  cat("→ Weight effects DIFFER significantly between drugs\n")
  cat("→ Drug-specific mechanisms, limited generalizability\n\n")
} else {
  cat("→ Weight effects DO NOT differ significantly between drugs\n")
  cat("→ Consistent pattern, semaglutide null result likely reflects insufficient power\n\n")
}

## -----------------------------------------------------------------------------
## 3.2: Test Drug × Duration Interaction
## -----------------------------------------------------------------------------

cat("3.2: DRUG × DURATION INTERACTION\n")
cat("--------------------------------\n")

# Build formula based on available factor levels
formula_parts <- c("y ~ drug * pre_dur_days_z", "pre_wt_treat_g_z")
if (n_distinct(df_both_mice_dio$pre_sex, na.rm = TRUE) > 1) {
  formula_parts <- c(formula_parts, "pre_sex")
}
if (n_distinct(df_both_mice_dio$pre_strain, na.rm = TRUE) > 1) {
  formula_parts <- c(formula_parts, "pre_strain")
}
formula_parts <- c(formula_parts,
                  "pre_age_treat_d_z", "pre_age_treat_d_missing",
                  "pre_wt_treat_g_missing", "pre_dose_mgkg_z",
                  "clin_dose_mg_z", "clin_dur_days_z", "clinical_sample_size_z",
                  "(1 | NCT_Number)")
formula_str <- paste(formula_parts, collapse = " + ")

model_drug_x_duration <- lmer(
  as.formula(formula_str),
  data = df_both_mice_dio,
  REML = TRUE,
  control = lmerControl(optimizer = "bobyqa")
)

coef_drug_x_dur <- summary(model_drug_x_duration)$coefficients

cat(sprintf("Drug × Duration:  β = %7.3f, SE = %.3f, p = %s %s\n\n",
            coef_drug_x_dur["drugsemaglutide:pre_dur_days_z", "Estimate"],
            coef_drug_x_dur["drugsemaglutide:pre_dur_days_z", "Std. Error"],
            format_pvalue(coef_drug_x_dur["drugsemaglutide:pre_dur_days_z", "Pr(>|t|)"]),
            get_sig_stars(coef_drug_x_dur["drugsemaglutide:pre_dur_days_z", "Pr(>|t|)"])))

if (coef_drug_x_dur["drugsemaglutide:pre_dur_days_z", "Pr(>|t|)"] < 0.05) {
  cat("→ Duration effects DIFFER significantly between drugs\n\n")
} else {
  cat("→ Duration effects DO NOT differ significantly between drugs\n\n")
}


cat("3.3: DRUG × DURATION × WEIGHT INTERACTION\n")
cat("-----------------------------------------\n")

formula_parts <- c("y ~ drug * pre_dur_days_z * pre_wt_treat_g_z")
if (n_distinct(df_both_mice_dio$pre_sex, na.rm = TRUE) > 1) {
  formula_parts <- c(formula_parts, "pre_sex")
}
if (n_distinct(df_both_mice_dio$pre_strain, na.rm = TRUE) > 1) {
  formula_parts <- c(formula_parts, "pre_strain")
}
formula_parts <- c(formula_parts,
                  "pre_age_treat_d_z", "pre_age_treat_d_missing",
                  "pre_wt_treat_g_missing", "pre_dose_mgkg_z",
                  "clin_dose_mg_z", "clin_dur_days_z", "clinical_sample_size_z",
                  "(1 | NCT_Number)")
formula_str <- paste(formula_parts, collapse = " + ")

model_drug_x_dur_x_wt <- lmer(
  as.formula(formula_str),
  data = df_both_mice_dio,
  REML = TRUE,
  control = lmerControl(optimizer = "bobyqa")
)

coef_3way <- summary(model_drug_x_dur_x_wt)$coefficients

# Name of the 3-way term depends on factor coding; with drug having level "semaglutide"
term_3way <- "drugsemaglutide:pre_dur_days_z:pre_wt_treat_g_z"

if (!term_3way %in% rownames(coef_3way)) {
  cat("WARNING: 3-way term not found. Available interaction terms:\n")
  print(grep(":", rownames(coef_3way), value = TRUE))
} else {
  cat(sprintf("Drug × Duration × Weight:  β = %7.3f, SE = %.3f, p = %s %s\n\n",
              coef_3way[term_3way, "Estimate"],
              coef_3way[term_3way, "Std. Error"],
              format_pvalue(coef_3way[term_3way, "Pr(>|t|)"]),
              get_sig_stars(coef_3way[term_3way, "Pr(>|t|)"])))
  
  if (coef_3way[term_3way, "Pr(>|t|)"] < 0.05) {
    cat("→ The duration×weight interaction differs significantly between drugs\n")
    cat("  (i.e., weight modifies the duration effect differently for semaglutide vs liraglutide)\n\n")
  } else {
    cat("→ No evidence that the duration×weight interaction differs between drugs\n")
    cat("  (i.e., the weight-dependent duration effect is broadly similar across drugs)\n\n")
  }
}

# Optional: also test whether adding the 3-way improves fit (use ML for LRT)
formula_parts <- c("y ~ drug + pre_dur_days_z + pre_wt_treat_g_z",
                  "drug:pre_dur_days_z", "drug:pre_wt_treat_g_z", "pre_dur_days_z:pre_wt_treat_g_z")
if (n_distinct(df_both_mice_dio$pre_sex, na.rm = TRUE) > 1) {
  formula_parts <- c(formula_parts, "pre_sex")
}
if (n_distinct(df_both_mice_dio$pre_strain, na.rm = TRUE) > 1) {
  formula_parts <- c(formula_parts, "pre_strain")
}
formula_parts <- c(formula_parts,
                  "pre_age_treat_d_z", "pre_age_treat_d_missing",
                  "pre_wt_treat_g_missing", "pre_dose_mgkg_z",
                  "clin_dose_mg_z", "clin_dur_days_z", "clinical_sample_size_z",
                  "(1 | NCT_Number)")
formula_str <- paste(formula_parts, collapse = " + ")

model_2way_only <- lmer(
  as.formula(formula_str),
  data = df_both_mice_dio,
  REML = FALSE,
  control = lmerControl(optimizer = "bobyqa")
)

model_3way_ML <- update(model_drug_x_dur_x_wt, REML = FALSE)

# Likelihood-ratio test details suppressed for brevity

## =============================================================================
## SECTION 4: SUMMARY TABLE
## =============================================================================

cat("\n")
cat("================================================================================\n")
cat("SECTION 4: SUMMARY TABLE\n")
cat("================================================================================\n\n")

summary_mice <- data.frame(
  Drug = c("Liraglutide", "Semaglutide", "Drug difference"),
  N_obs = c(
    nrow(df_lira_mice_dio),
    nrow(df_sema_mice_dio),
    nrow(df_both_mice_dio)
  ),
  N_studies = c(
    n_distinct(df_lira_mice_dio$pmcid),
    n_distinct(df_sema_mice_dio$pmcid),
    n_distinct(df_both_mice_dio$pmcid)
  ),
  Duration_beta = c(
    coef_lira_main["pre_dur_days_z", "Estimate"],
    coef_sema_main["pre_dur_days_z", "Estimate"],
    coef_drug_x_dur["drugsemaglutide:pre_dur_days_z", "Estimate"]
  ),
  Duration_p = c(
    coef_lira_main["pre_dur_days_z", "Pr(>|t|)"],
    coef_sema_main["pre_dur_days_z", "Pr(>|t|)"],
    coef_drug_x_dur["drugsemaglutide:pre_dur_days_z", "Pr(>|t|)"]
  ),
  Weight_beta = c(
    coef_lira_main["pre_wt_treat_g_z", "Estimate"],
    coef_sema_main["pre_wt_treat_g_z", "Estimate"],
    coef_drug_x_wt["drugsemaglutide:pre_wt_treat_g_z", "Estimate"]
  ),
  Weight_p = c(
    coef_lira_main["pre_wt_treat_g_z", "Pr(>|t|)"],
    coef_sema_main["pre_wt_treat_g_z", "Pr(>|t|)"],
    coef_drug_x_wt["drugsemaglutide:pre_wt_treat_g_z", "Pr(>|t|)"]
  ),
  Interaction_beta = c(
    coef_lira_int["pre_dur_days_z:pre_wt_treat_g_z", "Estimate"],
    coef_sema_int["pre_dur_days_z:pre_wt_treat_g_z", "Estimate"],
    NA
  ),
  Interaction_p = c(
    coef_lira_int["pre_dur_days_z:pre_wt_treat_g_z", "Pr(>|t|)"],
    coef_sema_int["pre_dur_days_z:pre_wt_treat_g_z", "Pr(>|t|)"],
    NA
  )
)

print(summary_mice, digits = 3, row.names = FALSE)

cat("\n")
cat("Significance: *** p<0.001, ** p<0.01, * p<0.05, NS p>=0.05\n")
cat("Positive β for duration/weight indicates worse translation (larger gap)\n\n")

# Export results
write.csv(summary_mice, "mice_weight_duration_summary.csv", row.names = FALSE)
cat("✓ Results exported to: mice_weight_duration_summary.csv\n\n")

## =============================================================================
## SECTION 5: INTERPRETATION
## =============================================================================

cat("================================================================================\n")
cat("SECTION 5: INTERPRETATION\n")
cat("================================================================================\n\n")

cat("See summary table above for complete results.\n")
cat("Positive β indicates worse translation (larger animal-to-human gap).\n")
cat("*** p<0.001, ** p<0.01, * p<0.05, NS p>=0.05\n\n")

cat("================================================================================\n")
cat("MICE ANALYSIS COMPLETE\n")
cat("================================================================================\n")
