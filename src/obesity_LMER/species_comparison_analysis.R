################################################################################
# SPECIES COMPARISON ANALYSIS: Mice vs Rats
# Tests whether using mice or rats models leads to different translation outcomes
# for liraglutide and semaglutide in DIO models
################################################################################

library(dplyr)
library(readr)
library(lme4)
library(lmerTest)
library(performance)  # For VIF calculation

## =============================================================================
## Load and Prepare Data
## =============================================================================

cat("\n")
cat("================================================================================\n")
cat("SPECIES COMPARISON ANALYSIS: MICE vs RATS\n")
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
    pre_species = preclinical_animal_species,
    clin_dose_mg = `clinical_dosage_amount_value(mg)`,
    clin_dur_days = `clinical_dosage_duration(days)`,
    y = translation_outcome
  ) %>%
  mutate(
    pre_wt_treat_g_missing = as.integer(is.na(pre_wt_treat_g)),
    pre_age_treat_d_missing = as.integer(is.na(pre_age_treat_d)),
    pre_wt_treat_g = ifelse(is.na(pre_wt_treat_g), median(pre_wt_treat_g, na.rm=TRUE), pre_wt_treat_g),
    pre_age_treat_d = ifelse(is.na(pre_age_treat_d), median(pre_age_treat_d, na.rm=TRUE), pre_age_treat_d),
    pre_species = factor(pre_species),
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
num_cols <- c("pre_dose_mgkg", "pre_dur_days", "pre_age_treat_d",
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
## SECTION 1: LIRAGLUTIDE - MICE vs RATS COMPARISON
## =============================================================================

cat("\n")
cat("================================================================================\n")
cat("SECTION 1: LIRAGLUTIDE - MICE vs RATS COMPARISON (DIO MODELS)\n")
cat("================================================================================\n\n")

df_lira_dio <- df %>%
  filter(intervention == "liraglutide",
         pre_species %in% c("mice", "rats"),
         model_type == "DIO",
         !is.na(y),
         !is.na(pre_dur_days)) %>%
  droplevels()

cat(sprintf("Total sample: %d observations from %d studies\n",
            nrow(df_lira_dio), n_distinct(df_lira_dio$pmcid)))
cat(sprintf("  Mice: %d observations from %d studies\n",
            sum(df_lira_dio$pre_species == "mice"),
            n_distinct(df_lira_dio$pmcid[df_lira_dio$pre_species == "mice"])))
cat(sprintf("  Rats: %d observations from %d studies\n\n",
            sum(df_lira_dio$pre_species == "rats"),
            n_distinct(df_lira_dio$pmcid[df_lira_dio$pre_species == "rats"])))

# Standardize predictors
df_lira_dio[paste0(num_cols, "_z")] <- scale(df_lira_dio[num_cols])

## -----------------------------------------------------------------------------
## 1.1: Main Species Effect
## -----------------------------------------------------------------------------

cat("MAIN SPECIES EFFECT\n")
cat("-------------------\n")

# Build formula - exclude strain and weight to avoid confounding with species
# (strain is species-specific; weight is 96% correlated with species)
# Include sex conditionally (not species-specific, but may be imbalanced)
formula_parts <- c("y ~ pre_species", "pre_dur_days_z", "preclinical_administration_route",
                   "pre_age_treat_d_z", "pre_age_treat_d_missing", "pre_dose_mgkg_z",
                   "clin_dose_mg_z", "clin_dur_days_z", "clinical_sample_size_z")
if (n_distinct(df_lira_dio$pre_sex, na.rm = TRUE) > 1) {
  formula_parts <- c(formula_parts, "pre_sex")
  cat("Including sex as fixed effect\n")
} else {
  cat("Sex excluded (only 1 level)\n")
}
formula_parts <- c(formula_parts, "(1 | NCT_Number)")
formula_lira_species <- paste(formula_parts, collapse = " + ")

model_lira_species <- tryCatch({
  lmer(
    as.formula(formula_lira_species),
    data = df_lira_dio,
    REML = TRUE,
    control = lmerControl(optimizer = "bobyqa")
  )
}, error = function(e) {
  cat("Model failed:", e$message, "\n")
  NULL
}, warning = function(w) {
  NULL
})

if (!is.null(model_lira_species)) {
  coef_lira_species <- summary(model_lira_species)$coefficients

  # Calculate 95% confidence intervals
  cat("Calculating 95% confidence intervals...\n")
  ci_lira_species <- confint(model_lira_species, method = "Wald", level = 0.95)

  if ("pre_speciesmice" %in% rownames(coef_lira_species)) {
    ci_lower <- ci_lira_species["pre_speciesmice", "2.5 %"]
    ci_upper <- ci_lira_species["pre_speciesmice", "97.5 %"]

    cat(sprintf("Species effect (mice vs rats): β = %7.3f, SE = %.3f, p = %s %s\n",
                coef_lira_species["pre_speciesmice", "Estimate"],
                coef_lira_species["pre_speciesmice", "Std. Error"],
                format_pvalue(coef_lira_species["pre_speciesmice", "Pr(>|t|)"]),
                get_sig_stars(coef_lira_species["pre_speciesmice", "Pr(>|t|)"])))
    cat(sprintf("                                95%% CI: [%.3f, %.3f]\n\n", ci_lower, ci_upper))

    if (coef_lira_species["pre_speciesmice", "Pr(>|t|)"] >= 0.05) {
      cat("→ NO SIGNIFICANT DIFFERENCE between mice and rats models\n")
      cat("→ Species choice does not significantly affect translation outcome\n\n")
    } else {
      cat("→ SIGNIFICANT DIFFERENCE found between mice and rats\n")
      cat("→ Species choice matters for translation prediction\n\n")
    }
  } else if ("pre_speciesrats" %in% rownames(coef_lira_species)) {
    ci_lower <- ci_lira_species["pre_speciesrats", "2.5 %"]
    ci_upper <- ci_lira_species["pre_speciesrats", "97.5 %"]

    cat(sprintf("Species effect (rats vs mice): β = %7.3f, SE = %.3f, p = %s %s\n",
                coef_lira_species["pre_speciesrats", "Estimate"],
                coef_lira_species["pre_speciesrats", "Std. Error"],
                format_pvalue(coef_lira_species["pre_speciesrats", "Pr(>|t|)"]),
                get_sig_stars(coef_lira_species["pre_speciesrats", "Pr(>|t|)"])))
    cat(sprintf("                               95%% CI: [%.3f, %.3f]\n\n", ci_lower, ci_upper))

    if (coef_lira_species["pre_speciesrats", "Pr(>|t|)"] >= 0.05) {
      cat("→ NO SIGNIFICANT DIFFERENCE between mice and rats models\n")
      cat("→ Species choice does not significantly affect translation outcome\n\n")
    } else {
      cat("→ SIGNIFICANT DIFFERENCE found between mice and rats\n")
      cat("→ Species choice matters for translation prediction\n\n")
    }
  }

  # VIF details suppressed for brevity
} else {
  cat("Model failed to converge\n\n")
  ci_lira_species <- NULL
}

## =============================================================================
## SECTION 2: SEMAGLUTIDE - MICE vs RATS COMPARISON
## =============================================================================

cat("\n")
cat("================================================================================\n")
cat("SECTION 2: SEMAGLUTIDE - MICE vs RATS COMPARISON (DIO MODELS)\n")
cat("================================================================================\n\n")

df_sema_dio <- df %>%
  filter(intervention == "semaglutide",
         pre_species %in% c("mice", "rats"),
         model_type == "DIO",
         !is.na(y),
         !is.na(pre_dur_days)) %>%
  droplevels()

cat(sprintf("Total sample: %d observations from %d studies\n",
            nrow(df_sema_dio), n_distinct(df_sema_dio$pmcid)))
cat(sprintf("  Mice: %d observations from %d studies\n",
            sum(df_sema_dio$pre_species == "mice"),
            n_distinct(df_sema_dio$pmcid[df_sema_dio$pre_species == "mice"])))
cat(sprintf("  Rats: %d observations from %d studies\n\n",
            sum(df_sema_dio$pre_species == "rats"),
            n_distinct(df_sema_dio$pmcid[df_sema_dio$pre_species == "rats"])))

if (nrow(df_sema_dio) < 30 || n_distinct(df_sema_dio$pre_species) < 2) {
  cat("⚠ INSUFFICIENT DATA for semaglutide species comparison\n")
  cat("  Need data from both mice and rats with adequate sample sizes\n\n")
  coef_sema_species <- NULL
} else {
  # Standardize predictors
  df_sema_dio[paste0(num_cols, "_z")] <- scale(df_sema_dio[num_cols])

  ## ---------------------------------------------------------------------------
  ## Main Species Effect
  ## ---------------------------------------------------------------------------

  cat("MAIN SPECIES EFFECT\n")
  cat("-------------------\n")

  # Build formula - exclude strain and weight to avoid confounding with species
  # Include sex conditionally (not species-specific, but may be imbalanced)
  formula_parts <- c("y ~ pre_species", "pre_dur_days_z", "preclinical_administration_route",
                     "pre_age_treat_d_z", "pre_age_treat_d_missing", "pre_dose_mgkg_z",
                     "clin_dose_mg_z", "clin_dur_days_z", "clinical_sample_size_z")
  if (n_distinct(df_sema_dio$pre_sex, na.rm = TRUE) > 1) {
    formula_parts <- c(formula_parts, "pre_sex")
    cat("Including sex as fixed effect\n")
  } else {
    cat("Sex excluded (only 1 level)\n")
  }
  formula_parts <- c(formula_parts, "(1 | NCT_Number)")
  formula_sema_species <- paste(formula_parts, collapse = " + ")

  model_sema_species <- tryCatch({
    lmer(
      as.formula(formula_sema_species),
      data = df_sema_dio,
      REML = TRUE,
      control = lmerControl(optimizer = "bobyqa")
    )
  }, error = function(e) {
    cat("Model failed:", e$message, "\n")
    NULL
  }, warning = function(w) {
    NULL
  })

  if (!is.null(model_sema_species)) {
    coef_sema_species <- summary(model_sema_species)$coefficients
    cat("Calculating 95% confidence intervals...\n")
    ci_sema_species <- confint(model_sema_species, method = "Wald", level = 0.95)

    if ("pre_speciesmice" %in% rownames(coef_sema_species)) {
      ci_lower <- ci_sema_species["pre_speciesmice", "2.5 %"]
      ci_upper <- ci_sema_species["pre_speciesmice", "97.5 %"]

      cat(sprintf("Species effect (mice vs rats): β = %7.3f, SE = %.3f, p = %s %s\n",
                  coef_sema_species["pre_speciesmice", "Estimate"],
                  coef_sema_species["pre_speciesmice", "Std. Error"],
                  format_pvalue(coef_sema_species["pre_speciesmice", "Pr(>|t|)"]),
                  get_sig_stars(coef_sema_species["pre_speciesmice", "Pr(>|t|)"])))
      cat(sprintf("                                95%% CI: [%.3f, %.3f]\n\n", ci_lower, ci_upper))

      # Check for unreliable estimate due to insufficient study diversity
      beta_val <- abs(coef_sema_species["pre_speciesmice", "Estimate"])
      n_rat_studies <- n_distinct(df_sema_dio$pmcid[df_sema_dio$pre_species == "rats"])

      if (beta_val > 100 || n_rat_studies == 1) {
        cat("⚠⚠⚠ WARNING: UNRELIABLE ESTIMATE ⚠⚠⚠\n")
        cat(sprintf("  Rat data from only %d study (need ≥3-5 for valid species comparison)\n", n_rat_studies))
        cat("  Species effect is CONFOUNDED with study-specific effects\n")
        cat(sprintf("  β = %.1f reflects 'this one rat study vs all mouse studies'\n", coef_sema_species["pre_speciesmice", "Estimate"]))
        cat("  NOT a true biological species difference\n")
        cat("  DO NOT INTERPRET as generalizable species effect\n\n")
      }

      if (coef_sema_species["pre_speciesmice", "Pr(>|t|)"] >= 0.05) {
        cat("→ NO SIGNIFICANT DIFFERENCE between mice and rats models\n")
        cat("→ Species choice does not significantly affect translation outcome\n\n")
      } else {
        cat("→ SIGNIFICANT DIFFERENCE found between mice and rats\n")
        cat("→ Species choice matters for translation prediction\n\n")
      }
    } else if ("pre_speciesrats" %in% rownames(coef_sema_species)) {
      ci_lower <- ci_sema_species["pre_speciesrats", "2.5 %"]
      ci_upper <- ci_sema_species["pre_speciesrats", "97.5 %"]

      cat(sprintf("Species effect (rats vs mice): β = %7.3f, SE = %.3f, p = %s %s\n",
                  coef_sema_species["pre_speciesrats", "Estimate"],
                  coef_sema_species["pre_speciesrats", "Std. Error"],
                  format_pvalue(coef_sema_species["pre_speciesrats", "Pr(>|t|)"]),
                  get_sig_stars(coef_sema_species["pre_speciesrats", "Pr(>|t|)"])))
      cat(sprintf("                               95%% CI: [%.3f, %.3f]\n\n", ci_lower, ci_upper))

      # Check for unreliable estimate due to insufficient study diversity
      beta_val <- abs(coef_sema_species["pre_speciesrats", "Estimate"])
      n_rat_studies <- n_distinct(df_sema_dio$pmcid[df_sema_dio$pre_species == "rats"])

      if (beta_val > 100 || n_rat_studies == 1) {
        cat("⚠⚠⚠ WARNING: UNRELIABLE ESTIMATE ⚠⚠⚠\n")
        cat(sprintf("  Rat data from only %d study (need ≥3-5 for valid species comparison)\n", n_rat_studies))
        cat("  Species effect is CONFOUNDED with study-specific effects\n")
        cat(sprintf("  β = %.1f reflects 'this one rat study vs all mouse studies'\n", coef_sema_species["pre_speciesrats", "Estimate"]))
        cat("  NOT a true biological species difference\n")
        cat("  DO NOT INTERPRET as generalizable species effect\n\n")
      }

      if (coef_sema_species["pre_speciesrats", "Pr(>|t|)"] >= 0.05) {
        cat("→ NO SIGNIFICANT DIFFERENCE between mice and rats models\n")
        cat("→ Species choice does not significantly affect translation outcome\n\n")
      } else {
        cat("→ SIGNIFICANT DIFFERENCE found between mice and rats\n")
        cat("→ Species choice matters for translation prediction\n\n")
      }
    }
  } else {
    cat("Model failed to converge\n\n")
    coef_sema_species <- NULL
    ci_sema_species <- NULL
  }

  # VIF details suppressed for brevity
}

## =============================================================================
## SECTION 3: SUMMARY TABLE
## =============================================================================

cat("\n")
cat("================================================================================\n")
cat("SECTION 3: SUMMARY TABLE\n")
cat("================================================================================\n\n")

# Create summary table
summary_species <- data.frame(
  Drug = character(),
  N_total = integer(),
  N_mice = integer(),
  N_rats = integer(),
  Species_main_effect_beta = numeric(),
  Species_main_effect_CI_lower = numeric(),
  Species_main_effect_CI_upper = numeric(),
  Species_main_effect_p = numeric(),
  stringsAsFactors = FALSE
)

# Add liraglutide results
if (!is.null(model_lira_species)) {
  species_term <- if ("pre_speciesmice" %in% rownames(coef_lira_species)) {
    "pre_speciesmice"
  } else {
    "pre_speciesrats"
  }

  summary_species <- rbind(summary_species, data.frame(
    Drug = "Liraglutide",
    N_total = nrow(df_lira_dio),
    N_mice = sum(df_lira_dio$pre_species == "mice"),
    N_rats = sum(df_lira_dio$pre_species == "rats"),
    Species_main_effect_beta = coef_lira_species[species_term, "Estimate"],
    Species_main_effect_CI_lower = ci_lira_species[species_term, "2.5 %"],
    Species_main_effect_CI_upper = ci_lira_species[species_term, "97.5 %"],
    Species_main_effect_p = coef_lira_species[species_term, "Pr(>|t|)"]
  ))
}

# Add semaglutide results
if (!is.null(coef_sema_species)) {
  species_term <- if ("pre_speciesmice" %in% rownames(coef_sema_species)) {
    "pre_speciesmice"
  } else {
    "pre_speciesrats"
  }

  summary_species <- rbind(summary_species, data.frame(
    Drug = "Semaglutide",
    N_total = nrow(df_sema_dio),
    N_mice = sum(df_sema_dio$pre_species == "mice"),
    N_rats = sum(df_sema_dio$pre_species == "rats"),
    Species_main_effect_beta = coef_sema_species[species_term, "Estimate"],
    Species_main_effect_CI_lower = ci_sema_species[species_term, "2.5 %"],
    Species_main_effect_CI_upper = ci_sema_species[species_term, "97.5 %"],
    Species_main_effect_p = coef_sema_species[species_term, "Pr(>|t|)"]
  ))
}

if (nrow(summary_species) > 0) {
  print(summary_species, digits = 3, row.names = FALSE)

  cat("\n")
  cat("Significance: *** p<0.001, ** p<0.01, * p<0.05, NS p>=0.05\n")
  cat("Species main effect β: difference in translation outcome between species\n")
  cat("95% CI: Wald confidence interval for species effect\n")
  cat("p>=0.05 indicates NO significant difference between mice and rats\n")
  cat("If CI includes 0, the species effect is not statistically significant\n\n")

  # Export results
  write.csv(summary_species, "species_comparison_summary.csv", row.names = FALSE)
  cat("✓ Results exported to: species_comparison_summary.csv\n\n")
}

## =============================================================================
## SECTION 4: INTERPRETATION
## =============================================================================

cat("================================================================================\n")
cat("SECTION 4: INTERPRETATION\n")
cat("================================================================================\n\n")

cat("See summary table above for complete results.\n")
cat("p>=0.05: No significant difference between mice and rats models.\n")
cat("CI including 0: Species effect not statistically significant.\n")
cat("*** p<0.001, ** p<0.01, * p<0.05, NS p>=0.05\n\n")

cat("================================================================================\n")
cat("SPECIES COMPARISON ANALYSIS COMPLETE\n")
cat("================================================================================\n")
