################################################################################
# LIRAGLUTIDE DIO BASELINE WEIGHT ANALYSIS: LMER — MICE ONLY
#
# Estimand: does preclinical baseline body weight (before treatment) predict the
# translational gap in liraglutide DIO studies, restricted to mouse models?
#
# Unit of analysis: individual preclinical-clinical arm pairs (row level).
#
# Statistical method:
#   Full model random effects: (1 | pmcid) + (1 | NCT_Number)
#   Full model fixed effects:  pre_wt_treat_g + preclinical_administration_route +
#     pre_dose_freq + pre_age_treat_d_z + pre_age_treat_d_missing +
#     pre_dose_mgkg_log_z + pre_dur_days_z + pre_n_z + clin_dose_mg_z +
#     clin_dur_days_z + clinical_sample_size_z + clin_age_groups + clin_phases
#     (+ pre_sex if >1 level)
#
#   Observations with missing pre_wt_treat_g are excluded (estimand cannot be
#   imputed). Missing pre_age_treat_d is median-imputed with a missingness
#   indicator covariate.
#
#   Before selection, candidate covariates are screened for near-collinearity
#   with pre_wt_treat_g: η (correlation ratio from one-way ANOVA) > 0.7
#   (categorical) or |r| > 0.7 (numeric) triggers exclusion to prevent
#   suppression bias.
#
#   VIF-based elimination: iteratively removes the covariate with the highest
#   VIF_equiv (> 10) until all retained terms have VIF_equiv ≤ 10.
#   Random effects (1 | pmcid) + (1 | NCT_Number) are always retained.
#   pre_wt_treat_g is the estimand of interest and is always retained.
#
#   Model residuals checked for homoscedasticity and normality.
#
#   R² (marginal and conditional) estimated with MuMIn (Bartón, 2015;
#   Nakagawa et al., 2017).
#
# Note: two publications containing genetic-background animals misclassified
# as DIO are excluded (same as lira_dio_duration_mice_lmer_robust.R):
#   PMC5348377 (KKAy, mice, arm 144), PMC8663785 (Ldlr-/-, mice, arm 263)
################################################################################

library(dplyr)
library(lme4)
library(lmerTest)
library(lmtest)
library(MuMIn)
library(clubSandwich)

## =============================================================================
## Load and Prepare Data
## =============================================================================

cat("================================================================================\n")
cat("LIRAGLUTIDE DIO BASELINE WEIGHT ANALYSIS: LMER — MICE ONLY\n")
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
      grepl("DIO|diet|high-fat|HFD", preclinical_disease_model, ignore.case = TRUE) ~ "DIO",
      grepl("ob/ob|db/db|genetic|leptin", preclinical_disease_model, ignore.case = TRUE) ~ "Genetic",
      TRUE ~ "Other"
    )
  )

## =============================================================================
## Filter: Liraglutide, DIO model, mice only
## =============================================================================

## pre_dur_days is a covariate here (standardized); clin_dose_mg, clin_dur_days
## are additional drug-level covariates.
drug_cols   <- c("pre_dose_mgkg_log", "pre_dur_days", "clin_dose_mg", "clin_dur_days")
global_cols <- c("pre_age_treat_d", "clinical_sample_size", "pre_n")

df_lira <- df %>%
  filter(
    tolower(as.character(intervention)) == "liraglutide",
    pre_species == "mice",
    model_type == "DIO",
    !is.na(y),
    !is.na(pre_wt_treat_g)          ## estimand must be observed
  ) %>%
  droplevels()

## Exclude specific arms with genetic-background animals misclassified as DIO
excluded_arm_ids <- c(144L, 263L)
df_lira <- df_lira %>%
  filter(!preclinical_arm_id %in% excluded_arm_ids) %>%
  mutate(pre_dose_mgkg_log = log(pre_dose_mgkg)) %>%
  droplevels()
cat(sprintf("NOTE: %d arms excluded — genetic strains misclassified as DIO:\n",
            length(excluded_arm_ids)))
cat("  arm 144 PMC5348377 (KKAy, mice), arm 263 PMC8663785 (Ldlr-/-, mice)\n\n")

cat(sprintf("Liraglutide DIO mice dataset: %d observations, %d publications, %d clinical trials\n\n",
            nrow(df_lira), n_distinct(df_lira$pmcid), n_distinct(df_lira$NCT_Number)))

## Baseline weight descriptives
wt_desc <- df_lira %>%
  distinct(preclinical_arm_id, .keep_all = TRUE) %>%
  summarise(
    n_arms = n(),
    mean   = round(mean(pre_wt_treat_g), 1),
    sd     = round(sd(pre_wt_treat_g), 1),
    median = median(pre_wt_treat_g),
    q25    = quantile(pre_wt_treat_g, 0.25),
    q75    = quantile(pre_wt_treat_g, 0.75),
    min    = min(pre_wt_treat_g),
    max    = max(pre_wt_treat_g)
  )
cat("Baseline weight (g) — unique preclinical arms:\n")
print(as.data.frame(wt_desc))
cat("\n")

## Standardize numeric predictors (covariates only; pre_wt_treat_g kept in raw grams)
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

fit_lmer_reml <- function(fml, data = df_lira, opt = NULL) {
  optimizers <- if (!is.null(opt)) opt else
    c("bobyqa", "nlminbwrap", "Nelder_Mead", "nloptwrap")
  for (o in optimizers) {
    m <- tryCatch(
      suppressWarnings(lmer(as.formula(fml), data = data, REML = TRUE,
                            control = lmerControl(optimizer = o))),
      error = function(e) NULL
    )
    if (!is.null(m)) return(list(model = m, optimizer = o))
  }
  list(model = NULL, optimizer = NA)
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

re_terms <- c("(1 | pmcid)", "(1 | NCT_Number)")

## Candidate categorical terms — only include if >1 level in this filtered dataset
cat_candidates <- list(
  preclinical_administration_route = "preclinical_administration_route",
  pre_dose_freq                    = "pre_dose_freq",
  clin_age_groups                  = "clin_age_groups",
  clin_phases                      = "clin_phases",
  pre_sex                          = "pre_sex"
)
included_cats <- character(0)
for (col in names(cat_candidates)) {
  nlevs <- n_distinct(df_lira[[col]], na.rm = TRUE)
  if (nlevs > 1) {
    included_cats <- c(included_cats, cat_candidates[[col]])
    cat(sprintf("Including %s as fixed effect (%d levels)\n", col, nlevs))
  } else {
    cat(sprintf("%s excluded (only 1 level: %s)\n", col,
                as.character(unique(df_lira[[col]])[1])))
  }
}
cat("\n")

fe_terms <- c(
  included_cats,
  "pre_age_treat_d_z",
  "pre_age_treat_d_missing",
  "pre_dose_mgkg_log_z",
  "pre_dur_days_z",
  "pre_n_z",
  "clin_dose_mg_z",
  "clin_dur_days_z",
  "clinical_sample_size_z"
)

make_formula <- function(fe, re) {
  paste(c("y ~ pre_wt_treat_g", fe, re), collapse = " + ")
}

## =============================================================================
## PRE-SCREENING: Exclude covariates near-collinear with pre_wt_treat_g
##
## Covariates strongly associated with the estimand act as suppressor variables,
## inflating the baseline weight coefficient. Threshold: η (correlation ratio
## from one-way ANOVA of pre_wt_treat_g ~ covariate) > 0.7 (categorical) or
## |r| > 0.7 (numeric, Pearson with pre_wt_treat_g).
## =============================================================================

cat("================================================================================\n")
cat("PRE-SCREENING: Collinearity with pre_wt_treat_g\n")
cat("================================================================================\n\n")
cat("Categorical: excluded if η (correlation ratio from ANOVA) > 0.7.\n")
cat("Numeric:     excluded if |r| > 0.7 (Pearson with pre_wt_treat_g).\n\n")

screen_results <- lapply(fe_terms, function(term) {
  col_name <- sub("_z$", "", term)
  if (!col_name %in% names(df_lira)) {
    return(data.frame(term = term, col = col_name, type = "unknown",
                      criterion = NA_character_, value = NA_character_,
                      exclude = FALSE, stringsAsFactors = FALSE))
  }
  vals <- df_lira[[col_name]]
  if (is.factor(vals) || is.character(vals)) {
    fit_aov <- tryCatch(
      aov(df_lira$pre_wt_treat_g ~ droplevels(factor(vals))),
      error = function(e) NULL
    )
    if (is.null(fit_aov)) {
      return(data.frame(term = term, col = col_name, type = "categorical",
                        criterion = "η > 0.7", value = "ANOVA failed",
                        exclude = FALSE, stringsAsFactors = FALSE))
    }
    ss  <- summary(fit_aov)[[1]]$`Sum Sq`
    eta <- if (sum(ss) > 0) sqrt(ss[1] / sum(ss)) else 0
    data.frame(term = term, col = col_name, type = "categorical",
               criterion = "η > 0.7",
               value = sprintf("η = %.3f", eta),
               exclude = eta > 0.7, stringsAsFactors = FALSE)
  } else {
    r <- abs(cor(as.numeric(vals), df_lira$pre_wt_treat_g, use = "complete.obs"))
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
  cat(sprintf("Excluding (near-collinear with pre_wt_treat_g): %s\n\n",
              paste(excluded_by_screen, collapse = ", ")))
  fe_terms <- fe_terms[!fe_terms %in% excluded_by_screen]
} else {
  cat("No covariates flagged — all retained in model.\n\n")
}

## Alias check: iteratively detect and remove perfectly collinear covariates.
n_alias_removed <- 0L
repeat {
  fml_alias_check <- as.formula(
    paste("y ~ pre_wt_treat_g +", paste(fe_terms, collapse = " + "))
  )
  m_alias_check <- tryCatch(lm(fml_alias_check, data = df_lira), error = function(e) NULL)
  if (is.null(m_alias_check)) break
  ali <- alias(m_alias_check)$Complete
  if (is.null(ali) || nrow(ali) == 0) break
  aliased_cols <- rownames(ali)
  aliased_vars <- unique(unlist(lapply(aliased_cols, function(col) {
    fe_terms[sapply(fe_terms, function(v) col == v || startsWith(col, v))]
  })))
  if (length(aliased_vars) == 0) break
  cat(sprintf(
    "NOTE: Aliased (perfectly collinear) term removed: %s\n",
    paste(aliased_vars, collapse = ", ")
  ))
  fe_terms <- fe_terms[!fe_terms %in% aliased_vars]
  n_alias_removed <- n_alias_removed + length(aliased_vars)
}
if (n_alias_removed > 0) cat("\n")

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
## VIF-BASED ELIMINATION (threshold: VIF_equiv > 10)
## Iteratively remove the covariate with highest VIF_equiv (protecting
## pre_wt_treat_g). Computed on equivalent lm() for the fixed-effects structure.
## =============================================================================

VIF_THRESHOLD <- 10L

get_vif_df <- function(fe) {
  fml <- as.formula(paste("y ~ pre_wt_treat_g +", paste(fe, collapse = " + ")))
  m   <- tryCatch(lm(fml, data = df_lira), error = function(e) NULL)
  if (is.null(m)) return(NULL)
  v   <- tryCatch(car::vif(m), error = function(e) NULL)
  if (is.null(v)) return(NULL)
  if (is.vector(v)) {
    data.frame(VIF_equiv = round(v, 3), row.names = names(v))
  } else {
    df_v <- as.data.frame(v)
    colnames(df_v)[3] <- "GVIF_1_2Df"
    df_v$VIF_equiv <- round(df_v$GVIF_1_2Df^2, 3)
    df_v
  }
}

cat("================================================================================\n")
cat("VIF-BASED ELIMINATION (threshold: VIF_equiv > 10)\n")
cat("================================================================================\n\n")

vif_fe_terms <- fe_terms
vif_removed  <- character(0)

repeat {
  df_v <- get_vif_df(vif_fe_terms)
  if (is.null(df_v)) break
  cov_rows <- setdiff(rownames(df_v), "pre_wt_treat_g")
  if (length(cov_rows) == 0) break
  max_row <- cov_rows[which.max(df_v[cov_rows, "VIF_equiv"])]
  max_val <- df_v[max_row, "VIF_equiv"]
  if (max_val <= VIF_THRESHOLD) break
  cat(sprintf("  VIF removal: %-35s  VIF_equiv = %.1f\n", max_row, max_val))
  vif_removed  <- c(vif_removed, max_row)
  vif_fe_terms <- vif_fe_terms[vif_fe_terms != max_row]
}

if (length(vif_removed) > 0) {
  cat(sprintf("\n  %d term(s) removed (VIF_equiv > %d): %s\n\n",
              length(vif_removed), VIF_THRESHOLD,
              paste(vif_removed, collapse = ", ")))
} else {
  cat("  No terms removed — all VIF_equiv ≤ 10.\n\n")
}

selected_fe   <- vif_fe_terms
final_formula <- make_formula(selected_fe, re_terms)
cat(sprintf("Final formula after VIF elimination:\n  %s\n\n", final_formula))

## =============================================================================
## STEP 1: FIT FINAL MODEL (REML = TRUE)
## =============================================================================

cat("================================================================================\n")
cat("STEP 1: FIT FINAL MODEL (REML = TRUE)\n")
cat("================================================================================\n\n")
cat(sprintf("Covariates retained after VIF elimination (VIF_equiv threshold = %d).\n\n",
            VIF_THRESHOLD))

res_final   <- fit_lmer_reml(final_formula)
model_final <- res_final$model
opt_final   <- res_final$optimizer

if (is.null(model_final)) stop("Final model failed with all optimizers.")
cat(sprintf("Model fitted with optimizer: %s\n", opt_final))
if (isSingular(model_final)) {
  cat("WARNING: singular fit — one or more random-effect variances estimated at zero.\n")
  cat("  Interpret random-effect estimates and ICC values cautiously.\n\n")
} else {
  cat("Random-effect structure: no singular fit detected.\n\n")
}

## =============================================================================
## STEP 2: VARIANCE INFLATION FACTORS (VIF)
## =============================================================================

cat("================================================================================\n")
cat("STEP 2: VARIANCE INFLATION FACTORS (VIF)\n")
cat("================================================================================\n\n")
cat("Computed via car::vif() on equivalent lm() (fixed effects only).\n")
cat("GVIF^(1/(2*Df)) is comparable across all terms; VIF equiv = [GVIF^(1/(2*Df))]^2.\n")
cat("Threshold: VIF equiv > 5 (high), > 10 (severe).\n\n")

fml_lm_vif <- as.formula(sub(" \\+ \\(1 \\| [^)]+\\)", "",
                              gsub(" \\+ \\(1 \\| [^)]+\\)", "", final_formula)))
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
## STEP 3: BASELINE WEIGHT EFFECT — Wald t-test and LRT
## =============================================================================

cat("================================================================================\n")
cat("STEP 3: BASELINE WEIGHT EFFECT — WALD t-TEST AND LRT\n")
cat("================================================================================\n\n")

ci_final <- confint(model_final, method = "Wald", level = 0.95)
wt_term  <- "pre_wt_treat_g"
beta <- NA; se <- NA; pval <- NA; ci_l <- NA; ci_u <- NA

if (wt_term %in% rownames(coefs_final)) {
  beta <- coefs_final[wt_term, "Estimate"]
  se   <- coefs_final[wt_term, "Std. Error"]
  pval <- coefs_final[wt_term, "Pr(>|t|)"]
  ci_l <- ci_final[wt_term, "2.5 %"]
  ci_u <- ci_final[wt_term, "97.5 %"]

  cat("BASELINE WEIGHT EFFECT (primary result)\n")
  cat("----------------------------------------\n")
  cat(sprintf("  Term: %s\n", wt_term))
  cat(sprintf("  β = %.5f per gram,  SE = %.5f\n", beta, se))
  cat(sprintf("  95%% CI (Wald): [%.5f, %.5f]\n", ci_l, ci_u))
  cat(sprintf("  Wald p = %s %s\n", format_p(pval), sig_stars(pval)))
}

## LRT for baseline weight (REML = FALSE comparison)
cat("\nLRT: BASELINE WEIGHT FIXED EFFECT\n")
cat("-----------------------------------\n")

fml_full_ml_wt <- make_formula(selected_fe, re_terms)
fml_null_ml_wt <- sub("y ~ pre_wt_treat_g \\+ ", "y ~ ", fml_full_ml_wt)
if (fml_null_ml_wt == fml_full_ml_wt) {
  fml_null_ml_wt <- paste(c("y ~ 1", re_terms), collapse = " + ")
}

lrt_p <- NA; lrt_chi <- NA; lrt_df <- NA

m_full_ml_wt <- fit_lmer_ml(fml_full_ml_wt, opt = opt_final)
m_null_ml_wt <- fit_lmer_ml(fml_null_ml_wt, opt = opt_final)

if (!is.null(m_full_ml_wt) && !is.null(m_null_ml_wt)) {
  ll_full <- as.numeric(logLik(m_full_ml_wt))
  ll_null <- as.numeric(logLik(m_null_ml_wt))
  ll_diff <- ll_full - ll_null
  lrt     <- anova(m_null_ml_wt, m_full_ml_wt)
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

if (wt_term %in% rownames(coefs_final)) {
  ref_p <- if (!is.na(lrt_p)) lrt_p else pval
  cat(sprintf("  Wald p = %s %s\n", format_p(pval), sig_stars(pval)))
  cat(sprintf("  LRT   p = %s %s  (preferred)\n\n", format_p(lrt_p), sig_stars(lrt_p)))
  if (!is.na(ref_p) && ref_p >= 0.05) {
    cat("-> NO SIGNIFICANT baseline weight effect (Liraglutide DIO — Mice)\n\n")
  } else {
    cat("-> SIGNIFICANT baseline weight effect: heavier animals at baseline are associated\n")
    cat("   with a larger translational gap.\n\n")
  }
}

## =============================================================================
## STEP 4: CLUSTER-ROBUST STANDARD ERRORS (CRSE)
##
## CR2 (bias-reduced linearization) applied separately for each clustering level:
##   4a. Cluster on pmcid (publication)   — G = n_distinct(pmcid)
##   4b. Cluster on NCT_Number (trial)    — G = n_distinct(NCT_Number)
##
## clubSandwich requires nested/single-level RE structure, so each block refits
## with the relevant single RE and clusters on that grouping variable.
## The LRT from Step 3 is unchanged — CRSE only affects SE/CI/p for Wald tests.
## =============================================================================

cat("================================================================================\n")
cat("STEP 4: CLUSTER-ROBUST STANDARD ERRORS (clubSandwich CR2)\n")
cat("================================================================================\n\n")
cat("NOTE: clubSandwich requires nested/single-level RE structure.\n")
cat("  Each block refits with the relevant single RE before applying CR2.\n\n")

## Helper: extract baseline weight CRSE row from coef_test output
crse_wt_row <- function(ct, beta_ref) {
  if (is.null(ct) || !wt_term %in% rownames(ct))
    return(list(se = NA, df = NA, p = NA, ci_l = NA, ci_u = NA))
  se_v <- ct[wt_term, "SE"]
  df_v <- ct[wt_term, "df_Satt"]
  p_v  <- ct[wt_term, "p_Satt"]
  list(
    se   = se_v,
    df   = df_v,
    p    = p_v,
    ci_l = beta_ref - qt(0.975, df_v) * se_v,
    ci_u = beta_ref + qt(0.975, df_v) * se_v
  )
}

## --------------------------------------------------------------------------
## 4a. Cluster on pmcid
## --------------------------------------------------------------------------

G_pmcid <- n_distinct(df_lira$pmcid)
cat(sprintf("--- 4a. Cluster on pmcid (G = %d publications) ---\n\n", G_pmcid))

fml_re_pmcid   <- sub(" \\+ \\(1 \\| NCT_Number\\)", "", final_formula)
model_re_pmcid <- tryCatch(
  suppressWarnings(lmer(as.formula(fml_re_pmcid), data = df_lira, REML = TRUE,
                        control = lmerControl(optimizer = opt_final))),
  error = function(e) { cat("  Model (pmcid RE) failed:", e$message, "\n"); NULL }
)

crse_pmcid_cr2 <- NULL; crse_pmcid_cr1 <- NULL
beta_pmcid <- NA
if (!is.null(model_re_pmcid)) {
  beta_pmcid <- summary(model_re_pmcid)$coefficients[wt_term, "Estimate"]
  cat(sprintf("  β (pmcid-RE model): %.6f  (crossed-RE: %.6f, diff = %+.6f)\n\n",
              beta_pmcid, beta, beta_pmcid - beta))

  vc2 <- tryCatch(vcovCR(model_re_pmcid, cluster = df_lira$pmcid, type = "CR2"),
                  error = function(e) { cat("  CR2 (pmcid) failed:", e$message, "\n"); NULL })
  vc1 <- tryCatch(vcovCR(model_re_pmcid, cluster = df_lira$pmcid, type = "CR1"),
                  error = function(e) NULL)
  if (!is.null(vc2)) {
    crse_pmcid_cr2 <- coef_test(model_re_pmcid, vcov = vc2, test = "Satterthwaite")
    cat("  Fixed effects — CR2 (pmcid):\n")
    print(crse_pmcid_cr2)
    cat("\n")
  }
  if (!is.null(vc1))
    crse_pmcid_cr1 <- coef_test(model_re_pmcid, vcov = vc1, test = "Satterthwaite")
}

pmcid_cr2 <- crse_wt_row(crse_pmcid_cr2, beta_pmcid)
pmcid_cr1 <- crse_wt_row(crse_pmcid_cr1, beta_pmcid)

## --------------------------------------------------------------------------
## 4b. Cluster on NCT_Number
## --------------------------------------------------------------------------

G_nct <- n_distinct(df_lira$NCT_Number)
cat(sprintf("--- 4b. Cluster on NCT_Number (G = %d trials) ---\n\n", G_nct))

fml_re_nct   <- sub("\\(1 \\| pmcid\\) \\+ ", "", final_formula)
model_re_nct <- tryCatch(
  suppressWarnings(lmer(as.formula(fml_re_nct), data = df_lira, REML = TRUE,
                        control = lmerControl(optimizer = opt_final))),
  error = function(e) { cat("  Model (NCT RE) failed:", e$message, "\n"); NULL }
)

crse_nct_cr2 <- NULL; crse_nct_cr1 <- NULL
beta_nct <- NA
if (!is.null(model_re_nct)) {
  beta_nct <- summary(model_re_nct)$coefficients[wt_term, "Estimate"]
  cat(sprintf("  β (NCT-RE model): %.6f  (crossed-RE: %.6f, diff = %+.6f)\n\n",
              beta_nct, beta, beta_nct - beta))

  vc2 <- tryCatch(vcovCR(model_re_nct, cluster = df_lira$NCT_Number, type = "CR2"),
                  error = function(e) { cat("  CR2 (NCT) failed:", e$message, "\n"); NULL })
  vc1 <- tryCatch(vcovCR(model_re_nct, cluster = df_lira$NCT_Number, type = "CR1"),
                  error = function(e) NULL)
  if (!is.null(vc2)) {
    crse_nct_cr2 <- coef_test(model_re_nct, vcov = vc2, test = "Satterthwaite")
    cat("  Fixed effects — CR2 (NCT_Number):\n")
    print(crse_nct_cr2)
    cat("\n")
  }
  if (!is.null(vc1))
    crse_nct_cr1 <- coef_test(model_re_nct, vcov = vc1, test = "Satterthwaite")
}

nct_cr2 <- crse_wt_row(crse_nct_cr2, beta_nct)
nct_cr1 <- crse_wt_row(crse_nct_cr1, beta_nct)

## --------------------------------------------------------------------------
## Unified comparison table — baseline weight term only
## --------------------------------------------------------------------------

cat("BASELINE WEIGHT TERM — UNIFIED SE COMPARISON\n")
cat("----------------------------------------------\n")

reml_se_v  <- coefs_final[wt_term, "Std. Error"]
reml_p_v   <- coefs_final[wt_term, "Pr(>|t|)"]
reml_ci_l  <- ci_final[wt_term, "2.5 %"]
reml_ci_u  <- ci_final[wt_term, "97.5 %"]

cat(sprintf("  β = %.6f  (per raw gram of preclinical baseline weight)\n\n", beta))
cat(sprintf("  %-36s  SE = %.6f,  95%% CI [%.6f, %.6f],  p = %s %s\n",
            "LMER crossed-RE (Satterthwaite):", reml_se_v, reml_ci_l, reml_ci_u,
            format_p(reml_p_v), sig_stars(reml_p_v)))

fmt_crse_row <- function(label, r) {
  if (is.na(r$se)) {
    cat(sprintf("  %-36s  (not estimable)\n", label))
  } else {
    cat(sprintf("  %-36s  SE = %.6f,  95%% CI [%.6f, %.6f],  p = %s %s  (df = %.1f)\n",
                label, r$se, r$ci_l, r$ci_u,
                format_p(r$p), sig_stars(r$p), r$df))
  }
}

fmt_crse_row(sprintf("CRSE CR2 (pmcid,       G = %2d):", G_pmcid), pmcid_cr2)
fmt_crse_row(sprintf("CRSE CR1 (pmcid,       G = %2d):", G_pmcid), pmcid_cr1)
fmt_crse_row(sprintf("CRSE CR2 (NCT_Number,  G = %2d):", G_nct),   nct_cr2)
fmt_crse_row(sprintf("CRSE CR1 (NCT_Number,  G = %2d):", G_nct),   nct_cr1)
cat("\n")

for (lbl_r in list(
    list(label = "CR2(pmcid) / REML",      se = pmcid_cr2$se),
    list(label = "CR2(NCT_Number) / REML", se = nct_cr2$se)
  )) {
  if (!is.na(lbl_r$se)) {
    ratio <- lbl_r$se / reml_se_v
    cat(sprintf("  SE ratio %-28s = %.2f\n", lbl_r$label, ratio))
  }
}
cat("\n")

## Save CRSE values for CSV output
cr2_se_v  <- pmcid_cr2$se;  cr2_df_v  <- pmcid_cr2$df
cr2_p_v   <- pmcid_cr2$p;   cr2_ci_l  <- pmcid_cr2$ci_l;  cr2_ci_u  <- pmcid_cr2$ci_u
nct_cr2_se_v <- nct_cr2$se; nct_cr2_df_v <- nct_cr2$df
nct_cr2_p_v  <- nct_cr2$p;  nct_cr2_ci_l <- nct_cr2$ci_l; nct_cr2_ci_u <- nct_cr2$ci_u

## =============================================================================
## STEP 5: RESIDUAL DIAGNOSTICS
## =============================================================================

cat("================================================================================\n")
cat("STEP 5: RESIDUAL DIAGNOSTICS\n")
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

## --- Diagnostic plots ---
plot_file <- "lira_dio_baselinewt_mice_lmer_robust_diagnostics.pdf"
pdf(plot_file, width = 8, height = 4)
par(mfrow = c(1, 2))
plot(fitted_val, resid_val,
     xlab = "Fitted values", ylab = "Residuals",
     main = "Residuals vs Fitted",
     pch = 16, cex = 0.6, col = "steelblue")
abline(h = 0, col = "red", lty = 2)
qqnorm(resid_val, main = "Normal Q-Q Plot",
       pch = 16, cex = 0.6, col = "steelblue")
qqline(resid_val, col = "red", lty = 2)
dev.off()
cat(sprintf("  Diagnostic plots saved: %s\n\n", plot_file))

## =============================================================================
## STEP 6: R² (MARGINAL AND CONDITIONAL) — MuMIn::r.squaredGLMM
## =============================================================================

cat("================================================================================\n")
cat("STEP 6: R² — MARGINAL (fixed effects) AND CONDITIONAL (fixed + random)\n")
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

if (wt_term %in% rownames(coefs_final)) {
  result_row <- data.frame(
    Drug              = "Liraglutide",
    Model             = "DIO",
    Species           = "mice",
    Final_formula     = final_formula,
    Selected_RE       = paste(re_terms, collapse = " + "),
    Selected_FE       = paste(c("pre_wt_treat_g", selected_fe), collapse = " + "),
    N_obs             = nrow(df_lira),
    N_studies         = n_distinct(df_lira$pmcid),
    N_trials          = n_distinct(df_lira$NCT_Number),
    ICC_pmcid         = round(icc_pmcid, 4),
    ICC_NCT           = round(icc_nct, 4),
    Weight_term       = wt_term,
    Beta_per_gram     = round(beta, 6),
    REML_SE           = round(se, 6),
    REML_CI_L         = round(ci_l, 6),
    REML_CI_U         = round(ci_u, 6),
    REML_Wald_p       = pval,
    LRT_chi2          = round(lrt_chi, 4),
    LRT_df            = lrt_df,
    LRT_p             = lrt_p,
    CR2_pmcid_SE      = round(cr2_se_v,     6),
    CR2_pmcid_df      = round(cr2_df_v,     2),
    CR2_pmcid_CI_L    = round(cr2_ci_l,     6),
    CR2_pmcid_CI_U    = round(cr2_ci_u,     6),
    CR2_pmcid_p       = cr2_p_v,
    CR2_pmcid_ratio   = round(cr2_se_v / se, 3),
    CR2_NCT_SE        = round(nct_cr2_se_v,  6),
    CR2_NCT_df        = round(nct_cr2_df_v,  2),
    CR2_NCT_CI_L      = round(nct_cr2_ci_l,  6),
    CR2_NCT_CI_U      = round(nct_cr2_ci_u,  6),
    CR2_NCT_p         = nct_cr2_p_v,
    CR2_NCT_ratio     = round(nct_cr2_se_v / se, 3),
    R2_marginal       = round(r2_m, 4),
    R2_conditional    = round(r2_c, 4),
    stringsAsFactors  = FALSE
  )
  write.csv(result_row, "lira_dio_baselinewt_mice_lmer_robust_summary.csv", row.names = FALSE)
  cat("Results saved to: lira_dio_baselinewt_mice_lmer_robust_summary.csv\n\n")
  print(result_row)
}

## =============================================================================
## DOSAGE AND DURATION CONFOUNDING CHECK
## =============================================================================

cat("================================================================================\n")
cat("DOSAGE AND DURATION CONFOUNDING CHECK\n")
cat("================================================================================\n\n")

sex_part   <- if ("pre_sex" %in% selected_fe) "pre_sex" else character(0)
route_part <- if ("preclinical_administration_route" %in% selected_fe)
               "preclinical_administration_route" else character(0)

fit_lmer_safe <- function(fml) {
  tryCatch(suppressWarnings(
    lmer(as.formula(fml), data = df_lira, REML = TRUE,
         control = lmerControl(optimizer = opt_final))),
    error = function(e) NULL)
}

## ---------------------------------------------------------------------------
## 1. Progressive β stability
## ---------------------------------------------------------------------------

cat("1. PROGRESSIVE β STABILITY (baseline weight β as covariates are added)\n\n")

formula_m1 <- paste(c("y ~ pre_wt_treat_g",
                       route_part, sex_part, re_terms),
                    collapse = " + ")
formula_m2 <- paste(c("y ~ pre_wt_treat_g",
                       route_part, "pre_dose_mgkg_log_z",
                       sex_part, re_terms),
                    collapse = " + ")
formula_m3 <- paste(c("y ~ pre_wt_treat_g",
                       route_part, "pre_dose_mgkg_log_z",
                       "pre_dur_days_z",
                       sex_part, re_terms),
                    collapse = " + ")
formula_m4 <- paste(c("y ~ pre_wt_treat_g",
                       route_part, "pre_dose_mgkg_log_z",
                       "pre_dur_days_z",
                       "pre_age_treat_d_z", "pre_age_treat_d_missing",
                       sex_part, re_terms),
                    collapse = " + ")

m1 <- fit_lmer_safe(formula_m1)
m2 <- fit_lmer_safe(formula_m2)
m3 <- fit_lmer_safe(formula_m3)
m4 <- fit_lmer_safe(formula_m4)

extract_wt_row <- function(fit, label) {
  if (is.null(fit))
    return(data.frame(Model = label, Beta = NA, SE = NA, Wald_p = NA,
                      stringsAsFactors = FALSE))
  cf <- summary(fit)$coefficients
  if (!wt_term %in% rownames(cf))
    return(data.frame(Model = label, Beta = NA, SE = NA, Wald_p = NA,
                      stringsAsFactors = FALSE))
  data.frame(
    Model  = label,
    Beta   = round(cf[wt_term, "Estimate"],   6),
    SE     = round(cf[wt_term, "Std. Error"], 6),
    Wald_p = round(cf[wt_term, "Pr(>|t|)"],  4),
    stringsAsFactors = FALSE
  )
}

stab <- rbind(
  extract_wt_row(m1,          "Weight only (+ route, sex)"),
  extract_wt_row(m2,          "+ log(dose)"),
  extract_wt_row(m3,          "+ log(dose) + duration"),
  extract_wt_row(m4,          "+ log(dose) + duration + age"),
  extract_wt_row(model_final, "Final selected model")
)
print(stab, row.names = FALSE)
cat("\n")

if (!is.na(stab$Beta[1]) && nrow(stab) >= 5 &&
    !is.na(stab$Beta[4]) && !is.na(stab$Beta[5]) && stab$Beta[1] != 0) {
  pct_dose_dur_age <- abs((stab$Beta[4] - stab$Beta[1]) / stab$Beta[1]) * 100
  pct_final        <- abs((stab$Beta[5] - stab$Beta[4]) / stab$Beta[1]) * 100
  cat(sprintf("  β shift from adding dose + duration + age: %.1f%% (%+.6f → %+.6f)\n",
              pct_dose_dur_age, stab$Beta[1], stab$Beta[4]))
  cat(sprintf("  β shift from adding further covariates:    %.1f%% (%+.6f → %+.6f)\n",
              pct_final, stab$Beta[4], stab$Beta[5]))
  cat("\n")
  if (pct_dose_dur_age < 10) {
    cat("-> Baseline weight effect ROBUST to dose, duration, and age adjustment (<10% change).\n\n")
  } else if (pct_dose_dur_age < 30) {
    cat("-> Baseline weight effect MODESTLY sensitive to adjustment (10–30% change).\n\n")
  } else {
    cat("-> Baseline weight effect shows SUBSTANTIAL sensitivity to adjustment (>30% change).\n\n")
  }
}

## ---------------------------------------------------------------------------
## 2. Interaction LRTs
## ---------------------------------------------------------------------------

cat("2. INTERACTION TESTS: does log(dose), duration, or age moderate the weight effect?\n\n")

fit_lmer_ml2 <- function(fml) {
  tryCatch(suppressWarnings(
    lmer(as.formula(fml), data = df_lira, REML = FALSE,
         control = lmerControl(optimizer = opt_final))),
    error = function(e) NULL)
}

lrt_interaction <- function(int_term, label) {
  fml_base <- final_formula
  fml_int  <- paste(final_formula, "+", int_term)
  m_base   <- fit_lmer_ml2(fml_base)
  m_int    <- fit_lmer_ml2(fml_int)
  if (is.null(m_base) || is.null(m_int)) {
    cat(sprintf("  %-40s  LRT failed\n", label)); return(invisible(NULL))
  }
  lrt <- anova(m_base, m_int)
  chi <- lrt$Chisq[2]; df_ <- lrt$Df[2]; p <- lrt$`Pr(>Chisq)`[2]
  cat(sprintf("  %-40s  χ²(%d) = %.3f,  p = %s %s\n",
              label, df_, chi, format_p(p), sig_stars(p)))
}

lrt_interaction("pre_wt_treat_g:pre_dose_mgkg_log_z", "Weight × log(dose)")
lrt_interaction("pre_wt_treat_g:pre_dur_days_z",       "Weight × duration")
lrt_interaction("pre_wt_treat_g:pre_age_treat_d_z",    "Weight × age")
lrt_interaction("pre_wt_treat_g:clin_dur_days_z",      "Weight × clinical duration")
cat("\n")
cat("  Non-significant interactions (p > 0.05) indicate that dosage, duration,\n")
cat("  age, and clinical duration do not moderate the baseline weight effect.\n\n")

## =============================================================================
## SENSITIVITY: LEAVE-ONE-PUBLICATION-OUT (LOPO)
## =============================================================================

cat("================================================================================\n")
cat("SENSITIVITY: LEAVE-ONE-PUBLICATION-OUT (LOPO)\n")
cat("================================================================================\n\n")
cat(sprintf("Refitting final selected model leaving out each of the %d publications.\n\n",
            n_distinct(df_lira$pmcid)))

pubs <- levels(df_lira$pmcid)

fml_null_lopo <- sub("y ~ pre_wt_treat_g \\+ ", "y ~ ", final_formula)
if (fml_null_lopo == final_formula)
  fml_null_lopo <- paste(c("y ~ 1", re_terms), collapse = " + ")

lopo_results <- lapply(pubs, function(pub) {
  df_loo <- df_lira %>% filter(pmcid != pub) %>% droplevels()

  m <- tryCatch(
    suppressWarnings(lmer(as.formula(final_formula), data = df_loo, REML = TRUE,
                          control = lmerControl(optimizer = opt_final))),
    error = function(e) NULL
  )
  if (is.null(m))
    return(data.frame(dropped_pub = pub, N_obs = nrow(df_loo),
                      Beta = NA, SE = NA, Wald_p = NA, LRT_p = NA,
                      stringsAsFactors = FALSE))

  cf <- summary(m)$coefficients
  if (!wt_term %in% rownames(cf))
    return(data.frame(dropped_pub = pub, N_obs = nrow(df_loo),
                      Beta = NA, SE = NA, Wald_p = NA, LRT_p = NA,
                      stringsAsFactors = FALSE))

  b_l  <- cf[wt_term, "Estimate"]
  se_l <- cf[wt_term, "Std. Error"]
  p_l  <- cf[wt_term, "Pr(>|t|)"]

  lrt_p_l <- tryCatch({
    m_full_l <- suppressWarnings(update(m, REML = FALSE))
    m_null_l <- suppressWarnings(lmer(as.formula(fml_null_lopo), data = df_loo,
                                      REML = FALSE,
                                      control = lmerControl(optimizer = opt_final)))
    lrt_l    <- anova(m_null_l, m_full_l)
    lrt_l$`Pr(>Chisq)`[2]
  }, error = function(e) NA)

  data.frame(
    dropped_pub = pub,
    N_obs       = nrow(df_loo),
    Beta        = round(b_l,      6),
    SE          = round(se_l,     6),
    Wald_p      = round(p_l,      4),
    LRT_p       = round(lrt_p_l,  4),
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
cat(sprintf("  Final-model β       = %+.6f\n", beta))
cat(sprintf("  LOPO models (valid) = %d / %d\n\n", length(beta_lopo), nrow(lopo_df)))
if (length(beta_lopo) > 0) {
  cat(sprintf("  Mean    = %+.6f\n",   mean(beta_lopo)))
  cat(sprintf("  Median  = %+.6f\n",   median(beta_lopo)))
  cat(sprintf("  SD      =  %.6f\n",   sd(beta_lopo)))
  cat(sprintf("  Q25–Q75 = [%+.6f, %+.6f]\n",
              quantile(beta_lopo, 0.25), quantile(beta_lopo, 0.75)))
  cat(sprintf("  Range   = [%+.6f, %+.6f]\n", min(beta_lopo), max(beta_lopo)))
  cat(sprintf("  N positive (β > 0)          = %d / %d\n",
              sum(beta_lopo > 0), length(beta_lopo)))
  n_lrt_sig  <- sum(!is.na(lopo_df$LRT_p)  & lopo_df$LRT_p  < 0.05)
  n_wald_sig <- sum(!is.na(lopo_df$Wald_p) & lopo_df$Wald_p < 0.05)
  cat(sprintf("  N LRT-significant  (p < 0.05) = %d / %d\n", n_lrt_sig,  nrow(lopo_df)))
  cat(sprintf("  N Wald-significant (p < 0.05) = %d / %d\n\n", n_wald_sig, nrow(lopo_df)))

  cat("  Wald p per LOPO model (dropped publication → Wald p):\n")
  lopo_wald <- lopo_df[order(lopo_df$Wald_p, na.last = TRUE),
                       c("dropped_pub", "Beta", "Wald_p", "sig_Wald")]
  for (i in seq_len(nrow(lopo_wald))) {
    r <- lopo_wald[i, ]
    p_str <- if (is.na(r$Wald_p)) "NA" else sprintf("%.4f", r$Wald_p)
    cat(sprintf("    %-15s  β = %+.6f  Wald p = %s  %s\n",
                r$dropped_pub, ifelse(is.na(r$Beta), NA_real_, r$Beta),
                p_str, r$sig_Wald))
  }
  cat("\n")
}

lopo_df$beta_shift <- abs(lopo_df$Beta - beta)
most_inf <- lopo_df[which.max(lopo_df$beta_shift), ]
cat(sprintf("  Most influential publication: %s\n", most_inf$dropped_pub))
cat(sprintf("  β without it: %.6f (shift = %+.6f from final-model β = %.6f)\n\n",
            most_inf$Beta, most_inf$Beta - beta, beta))

cat("================================================================================\n")
cat("LIRAGLUTIDE DIO BASELINE WEIGHT LMER ROBUST ANALYSIS (MICE ONLY) COMPLETE\n")
cat("================================================================================\n")
