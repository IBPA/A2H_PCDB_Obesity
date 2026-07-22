################################################################################
# LIRAGLUTIDE DIO SPECIES ANALYSIS: LMER — Mice vs Rats
#
# Estimand: does species (mice vs rats) predict the translational gap in
# liraglutide DIO studies?
#
# Unit of analysis: individual preclinical-clinical arm pairs (row level).
#
# Statistical method:
#   Full model random effects: (1 | pmcid) + (1 | NCT_Number)
#   Full model fixed effects:  pre_species + pre_dur_days_z +
#     preclinical_administration_route + pre_dose_freq + pre_age_treat_d_z +
#     pre_age_treat_d_missing + pre_dose_mgkg_log_z + pre_n_z +
#     clin_dose_mg_z + clin_dur_days_z + clinical_sample_size_z +
#     clin_age_groups + clin_phases (+ pre_sex if >1 level)
#
#   All covariates with VIF < 5 are retained in the final model without
#   automated elimination. Candidate covariates are pre-screened for
#   near-collinearity with pre_species: >50% of observations in species-exclusive
#   factor levels (categorical) or |r| > 0.7 (numeric) triggers exclusion to
#   prevent suppression bias. VIF is computed on the fixed-effects structure
#   via an equivalent lm() and reported alongside results.
#
#   Random effects (1 | pmcid) + (1 | NCT_Number) are always retained.
#   Pre_species is the estimand of interest.
#
#   Model residuals checked for homoscedasticity and normality.
#
#   R² (marginal and conditional) estimated with MuMIn (Bartón, 2015;
#   Nakagawa et al., 2017).
#
# Note: three publications containing genetic-background animals misclassified
# as DIO are excluded (same as lira_dio_species_lmer_analysis.R):
#   PMC8663785 (Ldlr-/-, mice), PMC5348377 (KKAy, mice),
#   PMC5966539 (Goto-Kakizaki, rats)
################################################################################

library(dplyr)
library(lme4)
library(lmerTest)
library(lmtest)
library(MuMIn)

## =============================================================================
## Load and Prepare Data
## =============================================================================

cat("================================================================================\n")
cat("LIRAGLUTIDE DIO SPECIES ANALYSIS: LMER\n")
cat("================================================================================\n\n")

args <- commandArgs(trailingOnly = TRUE)
data_file <- if (length(args) >= 1) args[1] else "../../data/obesity/obesity_a2h_dataset.csv"
cat(sprintf("Reading data from: %s\n\n", data_file))

df <- read.csv(data_file, check.names = FALSE)

df <- df %>%
  rename(
    NCT_Number       = `NCT Number`,
    pre_dose_mgkg    = `preclinical_dosage_amount_value(mg/kg)`,
    pre_dur_days     = `preclinical_dosage_duration(days)`,
    pre_wt_treat_g   = `preclinical_animal_weight_before_treatment(grams)`,
    pre_age_treat_d  = `preclinical_animal_age_before_treatment(days)`,
    pre_sex          = preclinical_animal_sex,
    pre_strain       = preclinical_animal_strain,
    pre_species      = preclinical_animal_species,
    pre_dose_freq    = preclinical_dosage_frequency,
    pre_n            = preclinical_animal_subject_size,
    clin_dose_mg     = `clinical_dosage_amount_value(mg)`,
    clin_dur_days    = `clinical_dosage_duration(days)`,
    clin_age_groups  = clinical_age_groups,
    clin_phases      = clinical_phases,
    y                = translation_outcome
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
      grepl("ob/ob|db/db|genetic|leptin", preclinical_disease_model, ignore.case = TRUE) ~ "Genetic",
      grepl("DIO|diet|high-fat|HFD", preclinical_disease_model, ignore.case = TRUE) ~ "DIO",
      TRUE ~ "Other"
    )
  )

## =============================================================================
## Filter: Liraglutide, DIO model, mice and rats only
## =============================================================================

drug_cols   <- c("pre_dose_mgkg_log", "pre_dur_days", "clin_dose_mg", "clin_dur_days")
global_cols <- c("pre_age_treat_d", "clinical_sample_size", "pre_n")

df_lira <- df %>%
  filter(
    tolower(as.character(intervention)) == "liraglutide",
    pre_species %in% c("mice", "rats"),
    model_type == "DIO",
    !is.na(y),
    !is.na(pre_dur_days)
  ) %>%
  droplevels()

## Exclude specific arms with genetic-background animals misclassified as DIO:
excluded_arm_ids <- c(144L, 170L, 263L)
df_lira <- df_lira %>%
  filter(!preclinical_arm_id %in% excluded_arm_ids) %>%
  mutate(pre_dose_mgkg_log = log(pre_dose_mgkg)) %>%
  droplevels()
cat(sprintf("NOTE: %d arms excluded — genetic strains misclassified as DIO:\n",
            length(excluded_arm_ids)))
cat("  arm 144 PMC5348377 (KKAy, mice), arm 170 PMC5966539 (Goto-Kakizaki, rats),\n")
cat("  arm 263 PMC8663785 (Ldlr-/-, mice)\n\n")

cat(sprintf("Liraglutide DIO dataset: %d observations, %d publications, %d clinical trials\n",
            nrow(df_lira), n_distinct(df_lira$pmcid), n_distinct(df_lira$NCT_Number)))
cat(sprintf("  Mice: %d obs from %d publications\n",
            sum(df_lira$pre_species == "mice"),
            n_distinct(df_lira$pmcid[df_lira$pre_species == "mice"])))
cat(sprintf("  Rats: %d obs from %d publications\n\n",
            sum(df_lira$pre_species == "rats"),
            n_distinct(df_lira$pmcid[df_lira$pre_species == "rats"])))

## Species-study confounding check
n_mixed <- df_lira %>%
  group_by(pmcid) %>%
  summarise(n_sp = n_distinct(pre_species), .groups = "drop") %>%
  summarise(mixed = sum(n_sp > 1)) %>%
  pull(mixed)
cat(sprintf("NOTE: %d / %d publications use both species\n",
            n_mixed, n_distinct(df_lira$pmcid)))
cat("  (1 | pmcid) RE will partially absorb between-study variance;\n")
cat("  within-study species signal is preserved where both species appear.\n\n")

## Standardize numeric predictors
df_lira[paste0(global_cols, "_z")] <- scale(df_lira[global_cols])

safe_scale <- function(x) {
  s <- sd(x, na.rm = TRUE)
  if (is.na(s) || s == 0) return(x - mean(x, na.rm = TRUE))
  as.numeric(scale(x))
}
df_lira <- df_lira %>%
  mutate(across(all_of(drug_cols), safe_scale, .names = "{.col}_z"))

## =============================================================================
## Helper: fit LMER with optimizer fallback
## =============================================================================

fit_lmer_reml <- function(fml, data = df_lira, opt = "bobyqa") {
  m <- tryCatch(
    suppressWarnings(lmer(as.formula(fml), data = data, REML = TRUE,
                          control = lmerControl(optimizer = opt))),
    error = function(e) NULL
  )
  list(model = m, optimizer = if (!is.null(m)) opt else NA)
}

fit_lmer_ml <- function(fml, data = df_lira, opt) {
  tryCatch(
    suppressWarnings(lmer(as.formula(fml), data = data, REML = FALSE,
                          control = lmerControl(optimizer = opt))),
    error = function(e) NULL
  )
}

## =============================================================================
## Build Full Model Formula
## =============================================================================

re_terms  <- c("(1 | pmcid)", "(1 | NCT_Number)")
fe_terms  <- c(
  "pre_dur_days_z",
  "preclinical_administration_route",
  "pre_dose_freq",
  "pre_age_treat_d_z",
  "pre_age_treat_d_missing",
  "pre_dose_mgkg_log_z",
  "pre_n_z",
  "clin_dose_mg_z",
  "clin_dur_days_z",
  "clinical_sample_size_z",
  "clin_age_groups",
  "clin_phases"
)
if (n_distinct(df_lira$pre_sex, na.rm = TRUE) > 1) {
  fe_terms <- c(fe_terms, "pre_sex")
  cat("Including sex as fixed effect\n")
} else {
  cat("Sex excluded (only 1 level)\n")
}

make_formula <- function(fe, re) {
  paste(c("y ~ pre_species", fe, re), collapse = " + ")
}

## =============================================================================
## PRE-SCREENING: Exclude covariates near-collinear with pre_species
##
## Covariates strongly associated with the estimand act as suppressor variables,
## inflating the species coefficient. Threshold: Cramér's V > 0.7 (categorical)
## or |r| > 0.7 (numeric, point-biserial against species as 0/1).
## =============================================================================

cat("================================================================================\n")
cat("PRE-SCREENING: Collinearity with pre_species\n")
cat("================================================================================\n\n")
cat("Categorical: excluded if >50% of observations fall in species-exclusive levels\n")
cat("             (levels present in only one species; threshold = 0.50).\n")
cat("Numeric:     excluded if |r| > 0.7 (point-biserial vs species as 0/1).\n\n")

species_bin <- as.integer(df_lira$pre_species == "rats")

screen_results <- lapply(fe_terms, function(term) {
  col_name <- sub("_z$", "", term)
  if (!col_name %in% names(df_lira)) {
    return(data.frame(term = term, col = col_name, type = "unknown",
                      criterion = NA_character_, value = NA_character_,
                      exclude = FALSE, stringsAsFactors = FALSE))
  }
  vals <- df_lira[[col_name]]
  if (is.factor(vals) || is.character(vals)) {
    tbl       <- table(droplevels(factor(vals)), df_lira$pre_species)
    ## Species-exclusive levels: present in only one species (zero cell)
    excl_lvls <- rownames(tbl)[apply(tbl, 1, function(r) any(r == 0))]
    n_excl    <- sum(vals %in% excl_lvls, na.rm = TRUE)
    prop_excl <- n_excl / length(vals)
    exclude   <- prop_excl > 0.5
    data.frame(term = term, col = col_name, type = "categorical",
               criterion = ">50% obs in exclusive levels",
               value = sprintf("%.1f%% (%s)", 100 * prop_excl,
                               if (length(excl_lvls) > 0) paste(excl_lvls, collapse = "; ") else "none"),
               exclude = exclude, stringsAsFactors = FALSE)
  } else {
    r <- abs(cor(as.numeric(vals), species_bin, use = "complete.obs"))
    data.frame(term = term, col = col_name, type = "numeric",
               criterion = "|r| > 0.7",
               value = sprintf("%.3f", r),
               exclude = !is.na(r) && r > 0.7, stringsAsFactors = FALSE)
  }
})

screen_df <- do.call(rbind, screen_results)
cat("Screening results:\n\n")
print(screen_df[, c("term", "type", "criterion", "value", "exclude")], row.names = FALSE)
cat("\n")

excluded_by_screen <- screen_df$term[!is.na(screen_df$exclude) & screen_df$exclude]
if (length(excluded_by_screen) > 0) {
  cat(sprintf("Excluding (near-collinear with pre_species): %s\n\n",
              paste(excluded_by_screen, collapse = ", ")))
  fe_terms <- fe_terms[!fe_terms %in% excluded_by_screen]
} else {
  cat("No covariates flagged — all proceed to backward selection.\n\n")
}

full_formula <- make_formula(fe_terms, re_terms)
cat(sprintf("Full model formula:\n  %s\n\n", full_formula))

## =============================================================================
## Helper Functions
## =============================================================================

format_p <- function(p) {
  if (is.na(p)) "NA" else if (p < 0.001) sprintf("%.2e", p) else sprintf("%.4f", p)
}
sig_stars <- function(p) {
  if (is.na(p)) "" else if (p < 0.001) "***" else if (p < 0.01) "**" else
    if (p < 0.05) "*" else if (p < 0.1) "." else "NS"
}

## =============================================================================
## STEP 1: FIT FINAL MODEL (REML = TRUE)
## =============================================================================

cat("================================================================================\n")
cat("STEP 1: FIT FINAL MODEL (REML = TRUE)\n")
cat("================================================================================\n\n")
cat("All covariates retained (no backward elimination; VIF < 5 for all terms).\n\n")
cat(sprintf("Final formula:\n  %s\n\n", full_formula))

res_final   <- fit_lmer_reml(full_formula)
model_final <- res_final$model
opt_final   <- res_final$optimizer

if (is.null(model_final)) stop("Final model failed with all optimizers.")
cat(sprintf("Model fitted with optimizer: %s\n\n", opt_final))

## =============================================================================
## STEP 2: VARIANCE INFLATION FACTORS (VIF)
## =============================================================================

cat("================================================================================\n")
cat("STEP 2: VARIANCE INFLATION FACTORS (VIF)\n")
cat("================================================================================\n\n")
cat("Computed via car::vif() on equivalent lm() (fixed effects only).\n")
cat("GVIF^(1/(2*Df)) is comparable across all terms; VIF equiv = [GVIF^(1/(2*Df))]^2.\n")
cat("Threshold: VIF equiv > 5 (high), > 10 (severe).\n\n")

fml_lm_vif <- as.formula(gsub(" \\+ \\(1 \\| [^)]+\\)", "", full_formula))
vif_df <- tryCatch({
  m_lm_vif <- lm(fml_lm_vif, data = df_lira)
  v <- car::vif(m_lm_vif)
  df_v <- as.data.frame(v)
  df_v$VIF_equiv <- round(df_v[, ncol(df_v)]^2, 3)
  colnames(df_v)[ncol(df_v) - 1] <- "GVIF_1_2Df"
  round(df_v, 3)
}, error = function(e) {
  cat(sprintf("  VIF computation failed: %s\n\n", e$message)); NULL
})
if (!is.null(vif_df)) {
  print(vif_df)
  max_vif <- max(vif_df$VIF_equiv, na.rm = TRUE)
  cat(sprintf("\n  Maximum VIF equiv: %.3f", max_vif))
  if (max_vif < 5) cat(" — no multicollinearity concern.\n\n") else
    cat(" — WARNING: high VIF detected.\n\n")
}

cat("FIXED EFFECTS SUMMARY\n")
cat("---------------------\n")
coefs_final <- summary(model_final)$coefficients
print(round(coefs_final, 4))
cat("\n")

## Variance components
vc_final  <- as.data.frame(VarCorr(model_final))
total_var <- sum(vc_final$vcov)
vc_final$pct <- round(100 * vc_final$vcov / total_var, 1)
cat("VARIANCE COMPONENTS\n")
cat("-------------------\n")
print(vc_final[, c("grp", "vcov", "sdcor", "pct")])
cat("\n")

icc_pmcid <- if ("pmcid" %in% vc_final$grp)
  vc_final$vcov[vc_final$grp == "pmcid"] / total_var else NA
icc_nct   <- if ("NCT_Number" %in% vc_final$grp)
  vc_final$vcov[vc_final$grp == "NCT_Number"] / total_var else NA
if (!is.na(icc_pmcid)) cat(sprintf("  ICC(pmcid)      = %.3f\n", icc_pmcid))
if (!is.na(icc_nct))   cat(sprintf("  ICC(NCT_Number) = %.3f\n", icc_nct))
cat("\n")

## =============================================================================
## STEP 3: SPECIES EFFECT — Wald t-test and LRT
## =============================================================================

cat("================================================================================\n")
cat("STEP 3: SPECIES EFFECT — WALD t-TEST AND LRT\n")
cat("================================================================================\n\n")

ci_final <- confint(model_final, method = "Wald", level = 0.95)

species_term <- grep("^pre_species", rownames(coefs_final), value = TRUE)
beta <- NA; se <- NA; pval <- NA; ci_l <- NA; ci_u <- NA

if (length(species_term) > 0) {
  st   <- species_term[1]
  beta <- coefs_final[st, "Estimate"]
  se   <- coefs_final[st, "Std. Error"]
  pval <- coefs_final[st, "Pr(>|t|)"]
  ci_l <- ci_final[st, "2.5 %"]
  ci_u <- ci_final[st, "97.5 %"]

  cat("SPECIES EFFECT (primary result)\n")
  cat("--------------------------------\n")
  cat(sprintf("  Term: %s\n", st))
  cat(sprintf("  β = %.3f,  SE = %.3f\n", beta, se))
  cat(sprintf("  95%% CI (Wald): [%.3f, %.3f]\n", ci_l, ci_u))
  cat(sprintf("  Wald p = %s %s\n", format_p(pval), sig_stars(pval)))
}

## LRT for species (REML = FALSE comparison)
cat("\nLRT: SPECIES FIXED EFFECT\n")
cat("--------------------------\n")

fml_full_ml_sp <- make_formula(fe_terms, re_terms)   # includes pre_species
fml_null_ml_sp <- sub("y ~ pre_species \\+ ", "y ~ ", fml_full_ml_sp)
if (fml_null_ml_sp == fml_full_ml_sp) {
  ## pre_species was the only fixed effect
  fml_null_ml_sp <- paste(c("y ~ 1", re_terms), collapse = " + ")
}

lrt_p <- NA; lrt_chi <- NA; lrt_df <- NA

m_full_ml_sp <- fit_lmer_ml(fml_full_ml_sp, opt = opt_final)
m_null_ml_sp <- fit_lmer_ml(fml_null_ml_sp, opt = opt_final)

if (!is.null(m_full_ml_sp) && !is.null(m_null_ml_sp)) {
  ll_full <- as.numeric(logLik(m_full_ml_sp))
  ll_null <- as.numeric(logLik(m_null_ml_sp))
  ll_diff <- ll_full - ll_null
  lrt     <- anova(m_null_ml_sp, m_full_ml_sp)
  lrt_chi <- if (length(lrt$Chisq) >= 2) lrt$Chisq[2] else NA
  lrt_df  <- if (length(lrt$Df)    >= 2) lrt$Df[2]    else NA
  lrt_p   <- if (length(lrt$`Pr(>Chisq)`) >= 2) lrt$`Pr(>Chisq)`[2] else NA
  if (!is.na(ll_diff) && ll_diff < 0) {
    cat(sprintf("  logLik(full) = %.3f,  logLik(null) = %.3f,  diff = %.3f\n",
                ll_full, ll_null, ll_diff))
    cat("  WARNING: null model has higher ML log-likelihood — LRT is degenerate.\n\n")
    lrt_chi <- NA; lrt_df <- NA; lrt_p <- NA
  } else {
    cat(sprintf("  χ²(%s) = %.3f,  p = %s %s\n\n",
                ifelse(is.na(lrt_df), "?", as.character(lrt_df)),
                ifelse(is.na(lrt_chi), 0, lrt_chi),
                format_p(lrt_p), sig_stars(lrt_p)))
  }
} else {
  cat("  LRT could not be computed.\n\n")
}

if (length(species_term) > 0) {
  ref_p <- if (!is.na(lrt_p)) lrt_p else pval
  cat(sprintf("  Wald p = %s %s\n", format_p(pval), sig_stars(pval)))
  cat(sprintf("  LRT   p = %s %s  (preferred)\n\n", format_p(lrt_p), sig_stars(lrt_p)))
  if (!is.na(ref_p) && ref_p >= 0.05) {
    cat("-> NO SIGNIFICANT DIFFERENCE between mice and rats (Liraglutide DIO)\n")
    cat("-> Species choice does not significantly affect translation outcome\n\n")
  } else {
    cat("-> SIGNIFICANT DIFFERENCE found between mice and rats (Liraglutide DIO)\n")
    cat("-> Species choice matters for translation prediction\n\n")
  }
}

## =============================================================================
## STEP 4: RESIDUAL DIAGNOSTICS
## =============================================================================

cat("================================================================================\n")
cat("STEP 4: RESIDUAL DIAGNOSTICS\n")
cat("================================================================================\n\n")

resid_val  <- resid(model_final)
fitted_val <- fitted(model_final)

## --- Normality: Shapiro-Wilk (n ≤ 5000) ---
cat("Normality of residuals (Shapiro-Wilk test):\n")
if (length(resid_val) <= 5000) {
  sw <- shapiro.test(resid_val)
  cat(sprintf("  W = %.4f,  p = %s\n", sw$statistic, format_p(sw$p.value)))
  if (sw$p.value > 0.05) {
    cat("  -> Residuals consistent with normality (p > 0.05)\n\n")
  } else {
    cat("  -> Residuals may deviate from normality (p ≤ 0.05) — interpret cautiously\n\n")
  }
} else {
  cat("  n > 5000: skipping Shapiro-Wilk; inspect Q-Q plot.\n\n")
}

## --- Homoscedasticity: Breusch-Pagan (residuals ~ fitted) ---
cat("Homoscedasticity of residuals (Breusch-Pagan test via lmtest):\n")
diag_df <- data.frame(resid = resid_val, fitted = fitted_val)
bp <- tryCatch(
  lmtest::bptest(lm(resid^2 ~ fitted, data = diag_df)),
  error = function(e) NULL
)
if (!is.null(bp)) {
  cat(sprintf("  BP = %.4f,  df = %d,  p = %s\n",
              bp$statistic, bp$parameter, format_p(bp$p.value)))
  if (bp$p.value > 0.05) {
    cat("  -> No evidence of heteroscedasticity (p > 0.05)\n\n")
  } else {
    cat("  -> Possible heteroscedasticity (p ≤ 0.05) — inspect residual plot\n\n")
  }
} else {
  cat("  Breusch-Pagan test failed.\n\n")
}

## --- Summary statistics of residuals ---
cat("Residual summary:\n")
print(summary(resid_val))
cat(sprintf("  SD of residuals: %.4f\n\n", sd(resid_val)))

## =============================================================================
## STEP 5: R² (MARGINAL AND CONDITIONAL) — MuMIn::r.squaredGLMM
## =============================================================================

cat("================================================================================\n")
cat("STEP 5: R² — MARGINAL (fixed effects) AND CONDITIONAL (fixed + random)\n")
cat("================================================================================\n\n")
cat("Using MuMIn::r.squaredGLMM() (Nakagawa et al., 2017; Bartón, 2015).\n\n")

r2_vals <- tryCatch(
  MuMIn::r.squaredGLMM(model_final),
  error = function(e) {
    cat(sprintf("  r.squaredGLMM() failed: %s\n", e$message)); NULL
  }
)
if (!is.null(r2_vals)) {
  r2_m <- r2_vals[1, "R2m"]
  r2_c <- r2_vals[1, "R2c"]
  cat(sprintf("  Marginal  R²LMM(m) = %.4f  (variance explained by fixed effects)\n", r2_m))
  cat(sprintf("  Conditional R²LMM(c) = %.4f  (fixed + random effects)\n\n", r2_c))
} else {
  r2_m <- NA; r2_c <- NA
}

## =============================================================================
## Save Results
## =============================================================================

cat("================================================================================\n")
cat("RESULTS SUMMARY\n")
cat("================================================================================\n\n")

if (length(species_term) > 0) {
  result_row <- data.frame(
    Drug              = "Liraglutide",
    Model             = "DIO",
    Final_formula     = full_formula,
    Selected_RE       = paste(re_terms, collapse = " + "),  # always (1|pmcid) + (1|NCT_Number)
    Selected_FE       = paste(c("pre_species", fe_terms), collapse = " + "),
    N_obs             = nrow(df_lira),
    N_studies         = n_distinct(df_lira$pmcid),
    N_trials          = n_distinct(df_lira$NCT_Number),
    N_mice_obs        = sum(df_lira$pre_species == "mice"),
    N_rats_obs        = sum(df_lira$pre_species == "rats"),
    ICC_pmcid         = round(icc_pmcid, 4),
    ICC_NCT           = round(icc_nct, 4),
    Species_term      = st,
    Beta              = round(beta, 4),
    SE                = round(se, 4),
    CI_L_95           = round(ci_l, 4),
    CI_U_95           = round(ci_u, 4),
    Wald_p            = pval,
    LRT_chi2          = round(lrt_chi, 4),
    LRT_df            = lrt_df,
    LRT_p             = lrt_p,
    R2_marginal       = round(r2_m, 4),
    R2_conditional    = round(r2_c, 4),
    stringsAsFactors  = FALSE
  )
  write.csv(result_row, "lira_dio_species_lmer_robust_summary.csv", row.names = FALSE)
  cat("Results saved to: lira_dio_species_lmer_robust_summary.csv\n\n")
  print(result_row)
}

## =============================================================================
## DOSAGE AND AGE CONFOUNDING CHECK
## =============================================================================

cat("================================================================================\n")
cat("DOSAGE AND AGE CONFOUNDING CHECK\n")
cat("================================================================================\n\n")

sex_part <- if ("pre_sex" %in% fe_terms) "pre_sex" else character(0)

## ---------------------------------------------------------------------------
## 1. Progressive β stability
## ---------------------------------------------------------------------------

cat("1. PROGRESSIVE β STABILITY (species β as covariates are added)\n\n")

formula_m1 <- paste(c("y ~ pre_species",
                       "preclinical_administration_route",
                       sex_part, re_terms),
                    collapse = " + ")
formula_m2 <- paste(c("y ~ pre_species",
                       "preclinical_administration_route",
                       "pre_dose_mgkg_log_z",
                       sex_part, re_terms),
                    collapse = " + ")
formula_m3 <- paste(c("y ~ pre_species",
                       "preclinical_administration_route",
                       "pre_dose_mgkg_log_z",
                       "pre_age_treat_d_z", "pre_age_treat_d_missing",
                       sex_part, re_terms),
                    collapse = " + ")

m1 <- fit_lmer_reml(formula_m1)$model
m2 <- fit_lmer_reml(formula_m2)$model
m3 <- fit_lmer_reml(formula_m3)$model

extract_sp_row <- function(fit, label) {
  if (is.null(fit))
    return(data.frame(Model = label, Beta = NA, SE = NA, Wald_p = NA,
                      stringsAsFactors = FALSE))
  cf <- summary(fit)$coefficients
  s  <- grep("^pre_speciesrats", rownames(cf), value = TRUE)
  if (length(s) == 0)
    return(data.frame(Model = label, Beta = NA, SE = NA, Wald_p = NA,
                      stringsAsFactors = FALSE))
  s <- s[1]
  data.frame(
    Model  = label,
    Beta   = round(cf[s, "Estimate"],   3),
    SE     = round(cf[s, "Std. Error"], 3),
    Wald_p = round(cf[s, "Pr(>|t|)"],  4),
    stringsAsFactors = FALSE
  )
}

stab <- rbind(
  extract_sp_row(m1,           "Species only (+ route, sex)"),
  extract_sp_row(m2,           "+ log(dose)"),
  extract_sp_row(m3,           "+ log(dose) + age"),
  extract_sp_row(model_final,  "Final selected model")
)
print(stab, row.names = FALSE)
cat("\n")

if (!is.na(stab$Beta[1]) && nrow(stab) >= 4 &&
    !is.na(stab$Beta[2]) && !is.na(stab$Beta[3]) && !is.na(stab$Beta[4])) {
  pct_dose_age <- abs((stab$Beta[3] - stab$Beta[1]) / stab$Beta[1]) * 100
  pct_dur_clin <- abs((stab$Beta[4] - stab$Beta[3]) / stab$Beta[1]) * 100
  cat(sprintf("  β shift from adding dose + age:           %.1f%% (%+.3f → %+.3f)\n",
              pct_dose_age, stab$Beta[1], stab$Beta[3]))
  cat(sprintf("  β shift from adding further covariates:   %.1f%% (%+.3f → %+.3f)\n",
              pct_dur_clin, stab$Beta[3], stab$Beta[4]))
  cat("\n")
  if (pct_dose_age < 30) {
    cat("-> Dose and age explain <30% of the species β — not the primary drivers.\n\n")
  } else {
    cat("-> Dose and age account for substantial β change — potential confounding.\n\n")
  }
}

## ---------------------------------------------------------------------------
## 2. Interaction LRTs
## ---------------------------------------------------------------------------

cat("2. INTERACTION TESTS: does log(dose) or age moderate the species effect?\n\n")

lrt_interaction <- function(int_term, label) {
  fml_base <- full_formula
  fml_int  <- paste(full_formula, "+", int_term)
  m_base   <- fit_lmer_ml(fml_base, opt = opt_final)
  m_int    <- fit_lmer_ml(fml_int,  opt = opt_final)
  if (is.null(m_base) || is.null(m_int)) {
    cat(sprintf("  %-38s  LRT failed\n", label)); return(invisible(NULL))
  }
  lrt <- anova(m_base, m_int)
  chi <- lrt$Chisq[2]; df_ <- lrt$Df[2]; p <- lrt$`Pr(>Chisq)`[2]
  cat(sprintf("  %-38s  χ²(%d) = %.3f,  p = %s %s\n",
              label, df_, chi, format_p(p), sig_stars(p)))
}

lrt_interaction("pre_species:pre_dose_mgkg_log_z", "Species × log(dose)")
lrt_interaction("pre_species:pre_age_treat_d_z",   "Species × age")
lrt_interaction("pre_species:pre_dur_days_z",       "Species × duration")
cat("\n")
cat("  Non-significant interactions (p > 0.05) indicate that dosage, age,\n")
cat("  and duration do not moderate the species effect.\n\n")

## =============================================================================
## SENSITIVITY: LEAVE-ONE-PUBLICATION-OUT (LOPO)
## =============================================================================

cat("================================================================================\n")
cat("SENSITIVITY: LEAVE-ONE-PUBLICATION-OUT (LOPO)\n")
cat("================================================================================\n\n")
cat(sprintf("Refitting final selected model leaving out each of the %d publications.\n\n",
            n_distinct(df_lira$pmcid)))

pubs <- levels(df_lira$pmcid)

lopo_results <- lapply(pubs, function(pub) {
  df_loo <- df_lira %>% filter(pmcid != pub) %>% droplevels()

  lopo_na_row <- function() data.frame(
    dropped_pub = pub,
    dropped_sp  = as.character(df_lira$pre_species[df_lira$pmcid == pub][1]),
    N_obs       = nrow(df_loo),
    Beta = NA, SE = NA, Wald_p = NA, LRT_p = NA,
    stringsAsFactors = FALSE
  )

  if (n_distinct(df_loo$pre_species) < 2) return(lopo_na_row())

  m <- fit_lmer_reml(full_formula, data = df_loo)$model
  if (is.null(m)) return(lopo_na_row())

  cf   <- summary(m)$coefficients
  st_l <- grep("^pre_species", rownames(cf), value = TRUE)
  if (length(st_l) == 0) return(lopo_na_row())

  b_l  <- cf[st_l[1], "Estimate"]
  se_l <- cf[st_l[1], "Std. Error"]
  p_l  <- cf[st_l[1], "Pr(>|t|)"]

  fml_null_l <- sub("y ~ pre_species \\+ ", "y ~ ", full_formula)
  if (fml_null_l == full_formula)
    fml_null_l <- paste(c("y ~ 1", re_terms), collapse = " + ")

  lrt_p_l <- tryCatch({
    m_full_l <- suppressWarnings(update(m, REML = FALSE))
    m_null_l <- suppressWarnings(lmer(as.formula(fml_null_l), data = df_loo,
                                      REML = FALSE,
                                      control = lmerControl(optimizer = opt_final)))
    lrt_l    <- anova(m_null_l, m_full_l)
    lrt_l$`Pr(>Chisq)`[2]
  }, error = function(e) NA)

  data.frame(
    dropped_pub = pub,
    dropped_sp  = as.character(df_lira$pre_species[df_lira$pmcid == pub][1]),
    N_obs       = nrow(df_loo),
    Beta        = round(b_l,  3),
    SE          = round(se_l, 3),
    Wald_p      = round(p_l,  4),
    LRT_p       = round(lrt_p_l, 4),
    stringsAsFactors = FALSE
  )
})

lopo_df <- do.call(rbind, lopo_results)
lopo_df$sig_LRT  <- ifelse(is.na(lopo_df$LRT_p),  "—",
                           ifelse(lopo_df$LRT_p  < 0.05, "*", "NS"))
lopo_df$sig_Wald <- ifelse(is.na(lopo_df$Wald_p), "—",
                           ifelse(lopo_df$Wald_p < 0.05, "*", "NS"))

cat("Leave-one-out results (sorted by LRT p):\n\n")
print(lopo_df[order(lopo_df$LRT_p, na.last = TRUE), ], row.names = FALSE)
cat("\n")

beta_lopo <- lopo_df$Beta[!is.na(lopo_df$Beta)]

cat("LOPO SUMMARY — β DISTRIBUTION\n")
cat("------------------------------\n")
cat(sprintf("  Final-model β       = %+.4f\n", beta))
cat(sprintf("  LOPO models (valid) = %d / %d\n\n", length(beta_lopo), nrow(lopo_df)))
if (length(beta_lopo) > 0) {
  cat(sprintf("  Mean    = %+.4f\n",   mean(beta_lopo)))
  cat(sprintf("  Median  = %+.4f\n",   median(beta_lopo)))
  cat(sprintf("  SD      =  %.4f\n",   sd(beta_lopo)))
  cat(sprintf("  Q25–Q75 = [%+.4f, %+.4f]\n",
              quantile(beta_lopo, 0.25), quantile(beta_lopo, 0.75)))
  cat(sprintf("  Range   = [%+.4f, %+.4f]\n", min(beta_lopo), max(beta_lopo)))
  cat(sprintf("  N negative (β < 0) = %d / %d\n\n",
              sum(beta_lopo < 0), length(beta_lopo)))
}

lopo_df$beta_shift <- abs(lopo_df$Beta - beta)
most_inf <- lopo_df[which.max(lopo_df$beta_shift), ]
cat(sprintf("  Most influential publication: %s (%s)\n",
            most_inf$dropped_pub, most_inf$dropped_sp))
cat(sprintf("  β without it: %.3f (shift = %+.3f from final-model β = %.3f)\n\n",
            most_inf$Beta, most_inf$Beta - beta, beta))

cat("================================================================================\n")
cat("LIRAGLUTIDE DIO SPECIES LMER ROBUST ANALYSIS COMPLETE\n")
cat("================================================================================\n")
