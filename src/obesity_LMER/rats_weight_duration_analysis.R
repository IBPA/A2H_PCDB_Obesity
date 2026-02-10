################################################################################
# RATS: Weight and Duration Effects Analysis
# Analyzes weight, duration, and weight×duration effects for liraglutide and
# semaglutide in rat DIO models
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
cat("RATS: WEIGHT AND DURATION EFFECTS ANALYSIS\n")
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
## SECTION 1: LIRAGLUTIDE RATS (DIO ONLY)
## =============================================================================

cat("\n")
cat("================================================================================\n")
cat("SECTION 1: LIRAGLUTIDE RATS (DIO MODELS)\n")
cat("================================================================================\n\n")

df_lira_rats_dio <- df %>%
  filter(intervention == "liraglutide",
         preclinical_animal_species == "rats",
         model_type == "DIO",
         !is.na(y),
         !is.na(pre_wt_treat_g),
         !is.na(pre_dur_days)) %>%
  droplevels()

cat(sprintf("Sample size: %d observations from %d studies\n", 
            nrow(df_lira_rats_dio), n_distinct(df_lira_rats_dio$pmcid)))

if (nrow(df_lira_rats_dio) < 50) {
  cat("\n⚠ WARNING: Small sample size may limit statistical power\n")
  cat("  Results should be interpreted with caution\n\n")
} else {
  cat("\n")
}

# Standardize predictors
df_lira_rats_dio[paste0(num_cols, "_z")] <- scale(df_lira_rats_dio[num_cols])

## -----------------------------------------------------------------------------
## 1.1: Main Effects (Duration + Weight)
## -----------------------------------------------------------------------------

if (nrow(df_lira_rats_dio) >= 50) {
  
  cat("1.1: MAIN EFFECTS MODEL\n")
  cat("-----------------------\n")

  # Build formula based on available factor levels
  # Exclude sex (adjusted VIF = 18.46, highest multicollinearity)
  formula_parts <- c("y ~ pre_dur_days_z", "pre_wt_treat_g_z")
  cat("Sex excluded (adjusted VIF = 18.46, highest multicollinearity)\n")
  if (n_distinct(df_lira_rats_dio$pre_strain, na.rm = TRUE) > 1) {
    formula_parts <- c(formula_parts, "pre_strain")
    cat("Including strain as fixed effect\n")
  } else {
    cat("Strain excluded (only 1 level)\n")
  }
  if (n_distinct(df_lira_rats_dio$preclinical_administration_route, na.rm = TRUE) > 1) {
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

  model_lira_main <- tryCatch({
    lmer(
      as.formula(formula_str),
      data = df_lira_rats_dio,
      REML = TRUE,
      control = lmerControl(optimizer = "bobyqa")
    )
  }, error = function(e) {
    NULL
  }, warning = function(w) {
    NULL
  })
  
  if (!is.null(model_lira_main)) {
    coef_lira_main <- summary(model_lira_main)$coefficients
    ci_lira_main <- confint(model_lira_main, method = "Wald", level = 0.95)

    if ("pre_dur_days_z" %in% rownames(coef_lira_main)) {
      cat(sprintf("Duration effect:  β = %7.3f, SE = %.3f, p = %s %s\n",
                  coef_lira_main["pre_dur_days_z", "Estimate"],
                  coef_lira_main["pre_dur_days_z", "Std. Error"],
                  format_pvalue(coef_lira_main["pre_dur_days_z", "Pr(>|t|)"]),
                  get_sig_stars(coef_lira_main["pre_dur_days_z", "Pr(>|t|)"])))
      cat(sprintf("                  95%% CI: [%.3f, %.3f]\n",
                  ci_lira_main["pre_dur_days_z", 1],
                  ci_lira_main["pre_dur_days_z", 2]))
    } else {
      cat("Duration coefficient dropped (rank deficiency)\n")
    }

    if ("pre_wt_treat_g_z" %in% rownames(coef_lira_main)) {
      cat(sprintf("Weight effect:    β = %7.3f, SE = %.3f, p = %s %s\n",
                  coef_lira_main["pre_wt_treat_g_z", "Estimate"],
                  coef_lira_main["pre_wt_treat_g_z", "Std. Error"],
                  format_pvalue(coef_lira_main["pre_wt_treat_g_z", "Pr(>|t|)"]),
                  get_sig_stars(coef_lira_main["pre_wt_treat_g_z", "Pr(>|t|)"])))
      cat(sprintf("                  95%% CI: [%.3f, %.3f]\n",
                  ci_lira_main["pre_wt_treat_g_z", 1],
                  ci_lira_main["pre_wt_treat_g_z", 2]))

    # VIF details suppressed for brevity
    } else {
      cat("Weight coefficient dropped (rank deficiency)\n\n")
    }
  } else {
    cat("Model failed to converge - insufficient data\n\n")
    coef_lira_main <- NULL
  }
  
  ## ---------------------------------------------------------------------------
  ## 1.2: Weight × Duration Interaction
  ## ---------------------------------------------------------------------------
  
  cat("1.2: WEIGHT × DURATION INTERACTION\n")
  cat("----------------------------------\n")

  # Build formula with interaction
  # Exclude sex (adjusted VIF = 18.46)
  formula_parts <- c("y ~ pre_dur_days_z * pre_wt_treat_g_z")
  if (n_distinct(df_lira_rats_dio$pre_strain, na.rm = TRUE) > 1) {
    formula_parts <- c(formula_parts, "pre_strain")
  }
  if (n_distinct(df_lira_rats_dio$preclinical_administration_route, na.rm = TRUE) > 1) {
    formula_parts <- c(formula_parts, "preclinical_administration_route")
  }
  formula_parts <- c(formula_parts,
                    "pre_age_treat_d_z", "pre_age_treat_d_missing",
                    "pre_wt_treat_g_missing", "pre_dose_mgkg_z",
                    "clin_dose_mg_z", "clin_dur_days_z", "clinical_sample_size_z",
                    "(1 | NCT_Number)")
  formula_str <- paste(formula_parts, collapse = " + ")

  model_lira_interaction <- tryCatch({
    lmer(
      as.formula(formula_str),
      data = df_lira_rats_dio,
      REML = TRUE,
      control = lmerControl(optimizer = "bobyqa")
    )
  }, error = function(e) {
    NULL
  }, warning = function(w) {
    NULL
  })
  
  if (!is.null(model_lira_interaction)) {
    coef_lira_int <- summary(model_lira_interaction)$coefficients
    ci_lira_int <- confint(model_lira_interaction, method = "Wald", level = 0.95)

    if ("pre_dur_days_z:pre_wt_treat_g_z" %in% rownames(coef_lira_int)) {
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
    } else {
      cat("Interaction term dropped (rank deficiency or convergence issue)\n\n")
      coef_lira_int <- NULL
    }
  } else {
    cat("Model failed to converge - insufficient data for interaction test\n\n")
    coef_lira_int <- NULL
  }
  
} else {
  cat("Insufficient data for liraglutide rat analysis (need ≥50 observations)\n\n")
  coef_lira_main <- NULL
  coef_lira_int <- NULL
}

## =============================================================================
## SECTION 2: SEMAGLUTIDE RATS (DIO ONLY)
## =============================================================================

cat("\n")
cat("================================================================================\n")
cat("SECTION 2: SEMAGLUTIDE RATS (DIO MODELS)\n")
cat("================================================================================\n\n")

df_sema_rats_dio <- df %>%
  filter(intervention == "semaglutide",
         preclinical_animal_species == "rats",
         model_type == "DIO",
         !is.na(y),
         !is.na(pre_wt_treat_g),
         !is.na(pre_dur_days)) %>%
  droplevels()

cat(sprintf("Sample size: %d observations from %d studies\n\n", 
            nrow(df_sema_rats_dio), n_distinct(df_sema_rats_dio$pmcid)))

if (nrow(df_sema_rats_dio) < 10) {
  cat("⚠ INSUFFICIENT DATA for semaglutide rat analysis\n")
  cat("  Need at least 10 observations\n")
  cat("  Semaglutide rats excluded from analysis\n\n")
  coef_sema_main <- NULL
  coef_sema_int <- NULL
} else {

  if (nrow(df_sema_rats_dio) < 30) {
    cat("⚠ WARNING: Very small sample size (N=%d)\n", nrow(df_sema_rats_dio))
    cat("  Results should be interpreted with extreme caution\n\n")
  }
  
  # Standardize predictors
  df_sema_rats_dio[paste0(num_cols, "_z")] <- scale(df_sema_rats_dio[num_cols])
  
  ## ---------------------------------------------------------------------------
  ## 2.1: Main Effects (Duration + Weight)
  ## ---------------------------------------------------------------------------
  
  cat("2.1: MAIN EFFECTS MODEL\n")
  cat("-----------------------\n")

  # Check which predictors have variation
  n_dur <- n_distinct(df_sema_rats_dio$pre_dur_days, na.rm = TRUE)
  n_wt <- n_distinct(df_sema_rats_dio$pre_wt_treat_g, na.rm = TRUE)
  n_sex <- n_distinct(df_sema_rats_dio$pre_sex, na.rm = TRUE)
  n_strain <- n_distinct(df_sema_rats_dio$pre_strain, na.rm = TRUE)

  # Report what's being excluded/included
  if (n_sex <= 1) cat("Sex excluded (only 1 level)\n")
  if (n_strain <= 1) cat("Strain excluded (only 1 level)\n")

  if (n_dur == 1 && n_wt == 1) {
    cat("⚠ CRITICAL: Duration and weight are CONSTANT (no variation)\n")
    cat("  Cannot estimate duration or weight effects\n")
    cat("  All 19 rats have identical preclinical parameters\n")
    cat("  Only clinical variables vary across observations\n\n")
    cat("Fitting model with CLINICAL PREDICTORS ONLY:\n")

    model_sema_main <- tryCatch({
      lm(y ~ clin_dose_mg_z + clin_dur_days_z + clinical_sample_size_z,
         data = df_sema_rats_dio)
    }, error = function(e) {
      NULL
    })

    if (!is.null(model_sema_main)) {
      coef_sema_main <- summary(model_sema_main)$coefficients

      cat("Model successfully fit with clinical predictors only\n")
      cat("Clinical dose:     β = ", sprintf("%7.3f", coef_sema_main["clin_dose_mg_z", "Estimate"]), "\n", sep="")
      cat("Clinical duration: β = ", sprintf("%7.3f", coef_sema_main["clin_dur_days_z", "Estimate"]), "\n", sep="")
      cat("Clinical N:        β = ", sprintf("%7.3f", coef_sema_main["clinical_sample_size_z", "Estimate"]), "\n\n", sep="")

      cat("⚠ NOTE: Duration and weight effects CANNOT be estimated\n")
      cat("  (all preclinical parameters are constant)\n\n")

      # Set to special values to indicate "cannot estimate"
      coef_sema_main <- data.frame(
        Estimate = c(NA, NA),
        `Std. Error` = c(NA, NA),
        `Pr(>|t|)` = c(NA, NA),
        row.names = c("pre_dur_days_z", "pre_wt_treat_g_z"),
        check.names = FALSE
      )
    } else {
      cat("Model failed\n\n")
      coef_sema_main <- NULL
    }
  } else {
    # Original logic for when there is variation
    cat(sprintf("Duration has %d unique values, Weight has %d unique values\n", n_dur, n_wt))

    # Build formula dynamically based on variation
    formula_parts <- c("y ~")
    if (n_dur > 1) formula_parts <- c(formula_parts, "pre_dur_days_z")
    if (n_wt > 1) formula_parts <- c(formula_parts, "pre_wt_treat_g_z")
    if (n_sex > 1) {
      formula_parts <- c(formula_parts, "pre_sex")
      cat("Including sex as fixed effect\n")
    }
    if (n_strain > 1) {
      formula_parts <- c(formula_parts, "pre_strain")
      cat("Including strain as fixed effect\n")
    }
    formula_parts <- c(formula_parts,
                      "pre_age_treat_d_z", "pre_age_treat_d_missing",
                      "pre_wt_treat_g_missing", "pre_dose_mgkg_z",
                      "clin_dose_mg_z", "clin_dur_days_z", "clinical_sample_size_z")
    formula_str <- paste(formula_parts, collapse = " + ")

    model_sema_main <- tryCatch({
      lm(as.formula(formula_str), data = df_sema_rats_dio)
    }, error = function(e) {
      NULL
    })

    if (!is.null(model_sema_main)) {
      coef_sema_main <- summary(model_sema_main)$coefficients

      if ("pre_dur_days_z" %in% rownames(coef_sema_main)) {
        cat(sprintf("Duration effect:  β = %7.3f, SE = %.3f, p = %s %s\n",
                    coef_sema_main["pre_dur_days_z", "Estimate"],
                    coef_sema_main["pre_dur_days_z", "Std. Error"],
                    format_pvalue(coef_sema_main["pre_dur_days_z", "Pr(>|t|)"]),
                    get_sig_stars(coef_sema_main["pre_dur_days_z", "Pr(>|t|)"])))
      } else {
        cat("Duration: No variation (constant)\n")
      }

      if ("pre_wt_treat_g_z" %in% rownames(coef_sema_main)) {
        cat(sprintf("Weight effect:    β = %7.3f, SE = %.3f, p = %s %s\n\n",
                    coef_sema_main["pre_wt_treat_g_z", "Estimate"],
                    coef_sema_main["pre_wt_treat_g_z", "Std. Error"],
                    format_pvalue(coef_sema_main["pre_wt_treat_g_z", "Pr(>|t|)"]),
                    get_sig_stars(coef_sema_main["pre_wt_treat_g_z", "Pr(>|t|)"])))
      } else {
        cat("Weight: No variation (constant)\n\n")
      }
    } else {
      cat("Model failed\n\n")
      coef_sema_main <- NULL
    }
  }
  
  ## ---------------------------------------------------------------------------
  ## 2.2: Weight × Duration Interaction
  ## ---------------------------------------------------------------------------
  
  cat("2.2: WEIGHT × DURATION INTERACTION\n")
  cat("----------------------------------\n")
  cat("Skipped - insufficient data for interaction test\n\n")
  coef_sema_int <- NULL
}

## =============================================================================
## SECTION 3: CROSS-DRUG COMPARISON (IF BOTH DRUGS HAVE DATA)
## =============================================================================

if (!is.null(coef_lira_main) && !is.null(coef_sema_main)) {
  
  cat("\n")
  cat("================================================================================\n")
  cat("SECTION 3: CROSS-DRUG COMPARISON (RATS DIO MODELS)\n")
  cat("================================================================================\n\n")
  
  # Combine datasets
  df_both_rats_dio <- rbind(
    df_lira_rats_dio %>% mutate(drug = "liraglutide"),
    df_sema_rats_dio %>% mutate(drug = "semaglutide")
  )
  
  cat(sprintf("Combined sample: %d observations from %d studies\n", 
              nrow(df_both_rats_dio), n_distinct(df_both_rats_dio$pmcid)))
  cat(sprintf("  Liraglutide: %d obs from %d studies\n", 
              nrow(df_lira_rats_dio), n_distinct(df_lira_rats_dio$pmcid)))
  cat(sprintf("  Semaglutide: %d obs from %d studies\n\n", 
              nrow(df_sema_rats_dio), n_distinct(df_sema_rats_dio$pmcid)))
  
  # Re-standardize on combined dataset
  df_both_rats_dio[paste0(num_cols, "_z")] <- scale(df_both_rats_dio[num_cols])
  
  ## ---------------------------------------------------------------------------
  ## 3.1: Test Drug × Weight Interaction
  ## ---------------------------------------------------------------------------
  
  cat("3.1: DRUG × WEIGHT INTERACTION\n")
  cat("------------------------------\n")

  # Build formula (consistent with mice analysis)
  formula_parts <- c("y ~ drug * pre_wt_treat_g_z", "pre_dur_days_z")
  if (n_distinct(df_both_rats_dio$pre_sex, na.rm = TRUE) > 1) {
    formula_parts <- c(formula_parts, "pre_sex")
  }
  if (n_distinct(df_both_rats_dio$pre_strain, na.rm = TRUE) > 1) {
    formula_parts <- c(formula_parts, "pre_strain")
  }
  if (n_distinct(df_both_rats_dio$preclinical_administration_route, na.rm = TRUE) > 1) {
    formula_parts <- c(formula_parts, "preclinical_administration_route")
  }
  formula_parts <- c(formula_parts,
                    "pre_age_treat_d_z", "pre_age_treat_d_missing",
                    "pre_wt_treat_g_missing", "pre_dose_mgkg_z",
                    "clin_dose_mg_z", "clin_dur_days_z", "clinical_sample_size_z",
                    "(1 | NCT_Number)")
  formula_str <- paste(formula_parts, collapse = " + ")

  model_drug_x_weight <- tryCatch({
    lmer(
      as.formula(formula_str),
      data = df_both_rats_dio,
      REML = TRUE,
      control = lmerControl(optimizer = "bobyqa")
    )
  }, error = function(e) {
    NULL
  })
  
  if (!is.null(model_drug_x_weight)) {
    coef_drug_x_wt <- summary(model_drug_x_weight)$coefficients

    # Check if interaction term exists
    if ("drugsemaglutide:pre_wt_treat_g_z" %in% rownames(coef_drug_x_wt)) {
      cat(sprintf("Drug × Weight:    β = %7.3f, SE = %.3f, p = %s %s\n\n",
                  coef_drug_x_wt["drugsemaglutide:pre_wt_treat_g_z", "Estimate"],
                  coef_drug_x_wt["drugsemaglutide:pre_wt_treat_g_z", "Std. Error"],
                  format_pvalue(coef_drug_x_wt["drugsemaglutide:pre_wt_treat_g_z", "Pr(>|t|)"]),
                  get_sig_stars(coef_drug_x_wt["drugsemaglutide:pre_wt_treat_g_z", "Pr(>|t|)"])))

      if (coef_drug_x_wt["drugsemaglutide:pre_wt_treat_g_z", "Pr(>|t|)"] < 0.05) {
        cat("→ Weight effects DIFFER significantly between drugs\n\n")
      } else {
        cat("→ Weight effects DO NOT differ significantly between drugs\n\n")
      }
    } else {
      cat("Interaction term dropped (semaglutide weight has no variation)\n\n")
      coef_drug_x_wt <- NULL
    }
  } else {
    cat("Model failed to converge - insufficient data\n\n")
    coef_drug_x_wt <- NULL
  }
  
  ## ---------------------------------------------------------------------------
  ## 3.2: Test Drug × Duration Interaction
  ## ---------------------------------------------------------------------------
  
  cat("3.2: DRUG × DURATION INTERACTION\n")
  cat("--------------------------------\n")

  # Build formula (consistent with mice analysis)
  formula_parts <- c("y ~ drug * pre_dur_days_z", "pre_wt_treat_g_z")
  if (n_distinct(df_both_rats_dio$pre_sex, na.rm = TRUE) > 1) {
    formula_parts <- c(formula_parts, "pre_sex")
  }
  if (n_distinct(df_both_rats_dio$pre_strain, na.rm = TRUE) > 1) {
    formula_parts <- c(formula_parts, "pre_strain")
  }
  if (n_distinct(df_both_rats_dio$preclinical_administration_route, na.rm = TRUE) > 1) {
    formula_parts <- c(formula_parts, "preclinical_administration_route")
  }
  formula_parts <- c(formula_parts,
                    "pre_age_treat_d_z", "pre_age_treat_d_missing",
                    "pre_wt_treat_g_missing", "pre_dose_mgkg_z",
                    "clin_dose_mg_z", "clin_dur_days_z", "clinical_sample_size_z",
                    "(1 | NCT_Number)")
  formula_str <- paste(formula_parts, collapse = " + ")

  model_drug_x_duration <- tryCatch({
    lmer(
      as.formula(formula_str),
      data = df_both_rats_dio,
      REML = TRUE,
      control = lmerControl(optimizer = "bobyqa")
    )
  }, error = function(e) {
    NULL
  })
  
  if (!is.null(model_drug_x_duration)) {
    coef_drug_x_dur <- summary(model_drug_x_duration)$coefficients

    # Check if interaction term exists
    if ("drugsemaglutide:pre_dur_days_z" %in% rownames(coef_drug_x_dur)) {
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
    } else {
      cat("Interaction term dropped (semaglutide duration has no variation)\n\n")
      coef_drug_x_dur <- NULL
    }
  } else {
    cat("Model failed to converge - insufficient data\n\n")
    coef_drug_x_dur <- NULL
  }
  
} else {
  cat("\n")
  cat("================================================================================\n")
  cat("SECTION 3: CROSS-DRUG COMPARISON\n")
  cat("================================================================================\n\n")
  cat("Skipped - insufficient data for both drugs\n\n")
  coef_drug_x_wt <- NULL
  coef_drug_x_dur <- NULL
}

## =============================================================================
## SECTION 4: SUMMARY TABLE
## =============================================================================

cat("\n")
cat("================================================================================\n")
cat("SECTION 4: SUMMARY TABLE\n")
cat("================================================================================\n\n")

if (!is.null(coef_lira_main)) {
  
  summary_rats <- data.frame(
    Drug = character(),
    N_obs = integer(),
    N_studies = integer(),
    Duration_beta = numeric(),
    Duration_p = numeric(),
    Weight_beta = numeric(),
    Weight_p = numeric(),
    Interaction_beta = numeric(),
    Interaction_p = numeric(),
    stringsAsFactors = FALSE
  )
  
  # Liraglutide
  if ("pre_dur_days_z" %in% rownames(coef_lira_main) && 
      "pre_wt_treat_g_z" %in% rownames(coef_lira_main)) {
    summary_rats <- rbind(summary_rats, data.frame(
      Drug = "Liraglutide",
      N_obs = nrow(df_lira_rats_dio),
      N_studies = n_distinct(df_lira_rats_dio$pmcid),
      Duration_beta = coef_lira_main["pre_dur_days_z", "Estimate"],
      Duration_p = coef_lira_main["pre_dur_days_z", "Pr(>|t|)"],
      Weight_beta = coef_lira_main["pre_wt_treat_g_z", "Estimate"],
      Weight_p = coef_lira_main["pre_wt_treat_g_z", "Pr(>|t|)"],
      Interaction_beta = if (!is.null(coef_lira_int) && 
                            "pre_dur_days_z:pre_wt_treat_g_z" %in% rownames(coef_lira_int)) {
        coef_lira_int["pre_dur_days_z:pre_wt_treat_g_z", "Estimate"]
      } else { NA },
      Interaction_p = if (!is.null(coef_lira_int) && 
                          "pre_dur_days_z:pre_wt_treat_g_z" %in% rownames(coef_lira_int)) {
        coef_lira_int["pre_dur_days_z:pre_wt_treat_g_z", "Pr(>|t|)"]
      } else { NA }
    ))
  }
  
  # Semaglutide
  if (!is.null(coef_sema_main)) {
    summary_rats <- rbind(summary_rats, data.frame(
      Drug = "Semaglutide",
      N_obs = nrow(df_sema_rats_dio),
      N_studies = n_distinct(df_sema_rats_dio$pmcid),
      Duration_beta = coef_sema_main["pre_dur_days_z", "Estimate"],
      Duration_p = coef_sema_main["pre_dur_days_z", "Pr(>|t|)"],
      Weight_beta = coef_sema_main["pre_wt_treat_g_z", "Estimate"],
      Weight_p = coef_sema_main["pre_wt_treat_g_z", "Pr(>|t|)"],
      Interaction_beta = NA,
      Interaction_p = NA
    ))
  }
  
  # Cross-drug comparison
  if (!is.null(coef_drug_x_wt) && !is.null(coef_drug_x_dur)) {
    summary_rats <- rbind(summary_rats, data.frame(
      Drug = "Drug difference",
      N_obs = nrow(df_both_rats_dio),
      N_studies = n_distinct(df_both_rats_dio$pmcid),
      Duration_beta = coef_drug_x_dur["drugsemaglutide:pre_dur_days_z", "Estimate"],
      Duration_p = coef_drug_x_dur["drugsemaglutide:pre_dur_days_z", "Pr(>|t|)"],
      Weight_beta = coef_drug_x_wt["drugsemaglutide:pre_wt_treat_g_z", "Estimate"],
      Weight_p = coef_drug_x_wt["drugsemaglutide:pre_wt_treat_g_z", "Pr(>|t|)"],
      Interaction_beta = NA,
      Interaction_p = NA
    ))
  }
  
  print(summary_rats, digits = 3, row.names = FALSE)
  
  cat("\n")
  cat("Significance: *** p<0.001, ** p<0.01, * p<0.05, NS p>=0.05\n")
  cat("Positive β for duration/weight indicates worse translation (larger gap)\n")
  cat("NA indicates insufficient data or model convergence issues\n\n")
  
  # Export results
  write.csv(summary_rats, "rats_weight_duration_summary.csv", row.names = FALSE)
  cat("✓ Results exported to: rats_weight_duration_summary.csv\n\n")
  
} else {
  cat("No results available - insufficient data for rat analysis\n\n")
}

## =============================================================================
## SECTION 5: INTERPRETATION
## =============================================================================

cat("================================================================================\n")
cat("SECTION 5: INTERPRETATION\n")
cat("================================================================================\n\n")

cat("See summary table above for complete results.\n")
cat("Note: Rat data are limited. Non-significant findings may reflect insufficient power.\n")
cat("Positive β indicates worse translation (larger animal-to-human gap).\n")
cat("*** p<0.001, ** p<0.01, * p<0.05, NS p>=0.05\n\n")

cat("================================================================================\n")
cat("RATS ANALYSIS COMPLETE\n")
cat("================================================================================\n")
