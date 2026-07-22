################################################################################
# Combined LMER Diagnostics: Residuals vs Fitted + Normal Q-Q
#
# Models:
#   1. Lira DIO species         (mice vs rats)
#   2. Lira DIO duration, mice
#   3. Lira DIO duration, rats
#   4. Sema DIO duration, mice
#
# Output: lmer_combined_diagnostics.pdf  (2 rows × 4 columns)
#
# Usage:  Rscript lmer_combined_diagnostics.R [data_file]
#   default data_file: ../../data/obesity/obesity_a2h_dataset.csv
################################################################################

suppressPackageStartupMessages({
  library(dplyr)
  library(lme4)
})

## ---------------------------------------------------------------------------
## 0. Data
## ---------------------------------------------------------------------------

args      <- commandArgs(trailingOnly = TRUE)
data_file <- if (length(args) >= 1) args[1] else "../../data/obesity/obesity_a2h_dataset.csv"
cat(sprintf("Reading data from: %s\n\n", data_file))

df_raw <- read.csv(data_file, check.names = FALSE)

df_raw <- df_raw %>%
  rename(
    NCT_Number      = `NCT Number`,
    pre_dose_mgkg   = `preclinical_dosage_amount_value(mg/kg)`,
    pre_dur_days    = `preclinical_dosage_duration(days)`,
    pre_wt_treat_g  = `preclinical_animal_weight_before_treatment(grams)`,
    pre_age_treat_d = `preclinical_animal_age_before_treatment(days)`,
    pre_sex         = preclinical_animal_sex,
    pre_strain      = preclinical_animal_strain,
    pre_species     = preclinical_animal_species,
    pre_dose_freq   = preclinical_dosage_frequency,
    pre_n           = preclinical_animal_subject_size,
    clin_dose_mg    = `clinical_dosage_amount_value(mg)`,
    clin_dur_days   = `clinical_dosage_duration(days)`,
    clin_age_groups = clinical_age_groups,
    clin_phases     = clinical_phases,
    y               = translation_outcome
  ) %>%
  mutate(
    pre_age_treat_d_missing = as.integer(is.na(pre_age_treat_d)),
    pre_age_treat_d  = ifelse(is.na(pre_age_treat_d),
                              median(pre_age_treat_d, na.rm = TRUE), pre_age_treat_d),
    pre_species      = factor(pre_species),
    pre_sex          = factor(pre_sex),
    pre_dose_freq    = factor(pre_dose_freq),
    clin_age_groups  = factor(clin_age_groups),
    clin_phases      = factor(clin_phases),
    intervention     = factor(intervention),
    NCT_Number       = factor(NCT_Number),
    pmcid            = factor(pmcid),
    model_type = case_when(
      grepl("DIO|diet|high-fat|HFD",           preclinical_disease_model, ignore.case = TRUE) ~ "DIO",
      grepl("ob/ob|db/db|genetic|leptin",       preclinical_disease_model, ignore.case = TRUE) ~ "Genetic",
      TRUE ~ "Other"
    )
  )

## ---------------------------------------------------------------------------
## Helper: standardise columns
## ---------------------------------------------------------------------------

safe_scale <- function(x) {
  s <- sd(x, na.rm = TRUE)
  if (is.na(s) || s == 0) return(x - mean(x, na.rm = TRUE))
  as.numeric(scale(x))
}

add_scaled_cols <- function(df, global_cols, drug_cols) {
  df[paste0(global_cols, "_z")] <- scale(df[global_cols])
  df <- df %>% mutate(across(all_of(drug_cols), safe_scale, .names = "{.col}_z"))
  df
}

fit_lmer <- function(fml, data) {
  for (opt in c("bobyqa", "nlminbwrap", "Nelder_Mead")) {
    m <- tryCatch(
      suppressWarnings(lmer(as.formula(fml), data = data, REML = TRUE,
                            control = lmerControl(optimizer = opt))),
      error = function(e) NULL
    )
    if (!is.null(m)) return(m)
  }
  stop(paste("All optimisers failed for formula:", fml))
}

## ---------------------------------------------------------------------------
## Model 1: Lira DIO species (mice vs rats)
## ---------------------------------------------------------------------------

global_cols1 <- c("pre_age_treat_d", "clinical_sample_size", "pre_n")
drug_cols1   <- c("pre_dose_mgkg_log", "pre_dur_days", "clin_dose_mg", "clin_dur_days")

df1 <- df_raw %>%
  filter(
    tolower(as.character(intervention)) == "liraglutide",
    pre_species %in% c("mice", "rats"),
    model_type == "DIO",
    !is.na(y), !is.na(pre_dur_days),
    !preclinical_arm_id %in% c(144L, 170L, 263L)
  ) %>%
  mutate(pre_dose_mgkg_log = log(pre_dose_mgkg)) %>%
  droplevels()

df1 <- add_scaled_cols(df1, global_cols1, drug_cols1)

fml1 <- paste(
  "y ~ pre_species + pre_dur_days_z + preclinical_administration_route +",
  "pre_dose_freq + pre_age_treat_d_z + pre_age_treat_d_missing +",
  "pre_dose_mgkg_log_z + pre_n_z + clin_dose_mg_z + clin_dur_days_z +",
  "clinical_sample_size_z + clin_age_groups + clin_phases + pre_sex +",
  "(1 | pmcid) + (1 | NCT_Number)"
)

cat("Fitting model 1: Lira DIO species...\n")
m1 <- fit_lmer(fml1, df1)

## ---------------------------------------------------------------------------
## Model 2: Lira DIO duration, mice
## ---------------------------------------------------------------------------

global_cols2 <- c("pre_age_treat_d", "clinical_sample_size", "pre_n")
drug_cols2   <- c("pre_dose_mgkg_log", "clin_dose_mg", "clin_dur_days")

df2 <- df_raw %>%
  filter(
    tolower(as.character(intervention)) == "liraglutide",
    pre_species == "mice",
    model_type == "DIO",
    !is.na(y), !is.na(pre_dur_days),
    !preclinical_arm_id %in% c(144L, 263L)
  ) %>%
  mutate(pre_dose_mgkg_log = log(pre_dose_mgkg)) %>%
  droplevels()

df2 <- add_scaled_cols(df2, global_cols2, drug_cols2)

fml2 <- paste(
  "y ~ pre_dur_days + pre_dose_freq + clin_age_groups + clin_phases + pre_sex +",
  "pre_age_treat_d_z + pre_age_treat_d_missing + pre_dose_mgkg_log_z +",
  "pre_n_z + clin_dose_mg_z + clin_dur_days_z + clinical_sample_size_z +",
  "(1 | pmcid) + (1 | NCT_Number)"
)

cat("Fitting model 2: Lira DIO duration (mice)...\n")
m2 <- fit_lmer(fml2, df2)

## ---------------------------------------------------------------------------
## Model 3: Lira DIO duration, rats
## ---------------------------------------------------------------------------

global_cols3 <- c("pre_age_treat_d", "clinical_sample_size", "pre_n")
drug_cols3   <- c("pre_dose_mgkg_log", "clin_dose_mg", "clin_dur_days")

df3 <- df_raw %>%
  filter(
    tolower(as.character(intervention)) == "liraglutide",
    pre_species == "rats",
    model_type == "DIO",
    !is.na(y), !is.na(pre_dur_days),
    !preclinical_arm_id %in% c(170L)
  ) %>%
  mutate(pre_dose_mgkg_log = log(pre_dose_mgkg)) %>%
  droplevels()

df3 <- add_scaled_cols(df3, global_cols3, drug_cols3)

fml3 <- paste(
  "y ~ pre_dur_days + preclinical_administration_route + clin_age_groups + clin_phases +",
  "pre_age_treat_d_z + pre_dose_mgkg_log_z + pre_n_z +",
  "clin_dose_mg_z + clin_dur_days_z + clinical_sample_size_z +",
  "(1 | pmcid) + (1 | NCT_Number)"
)

cat("Fitting model 3: Lira DIO duration (rats)...\n")
m3 <- fit_lmer(fml3, df3)

## ---------------------------------------------------------------------------
## Model 4: Sema DIO duration, mice
## ---------------------------------------------------------------------------

global_cols4 <- c("pre_age_treat_d", "clinical_sample_size", "pre_n")
drug_cols4   <- c("pre_dose_mgkg_log", "clin_dur_days")

df4 <- df_raw %>%
  filter(
    tolower(as.character(intervention)) == "semaglutide",
    pre_species == "mice",
    model_type == "DIO",
    !is.na(y), !is.na(pre_dur_days),
    !preclinical_arm_id %in% c(22L)
  ) %>%
  mutate(pre_dose_mgkg_log = log(pre_dose_mgkg)) %>%
  droplevels()

df4 <- add_scaled_cols(df4, global_cols4, drug_cols4)

fml4 <- paste(
  "y ~ pre_dur_days + preclinical_administration_route + pre_dose_freq +",
  "clin_age_groups + clin_phases + pre_age_treat_d_z + pre_age_treat_d_missing +",
  "pre_dose_mgkg_log_z + pre_n_z + clin_dur_days_z + clinical_sample_size_z +",
  "(1 | pmcid) + (1 | NCT_Number)"
)

cat("Fitting model 4: Sema DIO duration (mice)...\n")
m4 <- fit_lmer(fml4, df4)

## ---------------------------------------------------------------------------
## Combined diagnostic figure
## ---------------------------------------------------------------------------

models <- list(m1, m2, m3, m4)
labels <- c(
  "Lira DIO: species\n(mice vs rats)",
  "Lira DIO: duration\n(mice)",
  "Lira DIO: duration\n(rats)",
  "Sema DIO: duration\n(mice)"
)

out_file <- "lmer_combined_diagnostics.pdf"
pdf(out_file, width = 7, height = 11)
par(mfrow = c(4, 2), mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0))

for (i in seq_along(models)) {
  rv <- resid(models[[i]])
  fv <- fitted(models[[i]])

  ## Residuals vs Fitted
  plot(fv, rv,
       xlab = "Fitted values", ylab = "Residuals",
       main = paste0(labels[i], "\nResiduals vs Fitted"),
       pch = 16, cex = 0.6, col = "steelblue",
       cex.main = 0.85)
  abline(h = 0, col = "red", lty = 2)

  ## Normal Q-Q
  qqnorm(rv, main = paste0(labels[i], "\nNormal Q-Q"),
         pch = 16, cex = 0.6, col = "steelblue",
         cex.main = 0.85)
  qqline(rv, col = "red", lty = 2)
}

mtext("LMER model residual diagnostics", outer = TRUE, cex = 1, font = 2)
dev.off()

cat(sprintf("\nDiagnostic figure saved: %s\n", out_file))
