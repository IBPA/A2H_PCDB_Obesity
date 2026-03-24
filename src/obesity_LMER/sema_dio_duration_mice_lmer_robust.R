################################################################################
# SEMAGLUTIDE DIO DURATION ANALYSIS: LMER — MICE ONLY
#
# Estimand: does preclinical treatment duration predict the translational gap
# in semaglutide DIO studies, restricted to mouse models?
#
# Unit of analysis: individual preclinical-clinical arm pairs (row level).
#
# Statistical method:
#   Full model random effects: (1 | pmcid) + (1 | NCT_Number)
#   Full model fixed effects:  pre_dur_days + preclinical_administration_route +
#     pre_dose_freq + pre_age_treat_d_z + pre_age_treat_d_missing +
#     pre_dose_mgkg_log_z + pre_n_z + clin_dose_mg_z + clin_dur_days_z +
#     clinical_sample_size_z + clin_age_groups + clin_phases
#     (+ pre_sex if >1 level)
#   Categorical covariates with only 1 level in the filtered dataset are
#   automatically excluded.
#
#   Before selection, candidate covariates are screened for near-collinearity
#   with pre_dur_days: η (correlation ratio from one-way ANOVA) > 0.7
#   (categorical) or |r| > 0.7 (numeric) triggers exclusion to prevent
#   suppression bias.
#
#   Alias check: iteratively removes perfectly collinear covariates (VIF = ∞)
#   by mapping aliased model-matrix columns back to their parent variable.
#
#   VIF-based elimination: iteratively removes the covariate with the highest
#   VIF_equiv (> 10) until all retained terms have VIF_equiv ≤ 10.
#   Random effects (1 | pmcid) + (1 | NCT_Number) are always retained.
#   pre_dur_days is the estimand of interest and is always retained.
#
#   Model residuals checked for homoscedasticity and normality.
#
#   R² (marginal and conditional) estimated with MuMIn (Bartón, 2015;
#   Nakagawa et al., 2017).
#
# Note: one publication containing a genetic-background animal misclassified
# as DIO is excluded:
#   PMC10218334 (Ldlr-/-.Leiden, mice, arm 22)
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
cat("SEMAGLUTIDE DIO DURATION ANALYSIS: LMER — MICE ONLY\n")
cat("================================================================================\n\n")

args <- commandArgs(trailingOnly = TRUE)
data_file <- if (length(args) >= 1) args[1] else "../../data/obesity_a2h_v2.csv"
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
## Filter: Semaglutide, DIO model, mice only
## =============================================================================

drug_cols   <- c("pre_dose_mgkg_log", "clin_dose_mg", "clin_dur_days")
global_cols <- c("pre_age_treat_d", "clinical_sample_size", "pre_n")

df_sema <- df %>%
  filter(
    tolower(as.character(intervention)) == "semaglutide",
    pre_species == "mice",
    model_type == "DIO",
    !is.na(y),
    !is.na(pre_dur_days)
  ) %>%
  droplevels()

## Exclude arm with genetic-background animal misclassified as DIO
excluded_arm_ids <- c(22L)
df_sema <- df_sema %>%
  filter(!preclinical_arm_id %in% excluded_arm_ids) %>%
  mutate(pre_dose_mgkg_log = log(pre_dose_mgkg)) %>%
  droplevels()
cat(sprintf("NOTE: %d arm excluded — genetic strain misclassified as DIO:\n",
            length(excluded_arm_ids)))
cat("  arm 22 PMC10218334 (Ldlr-/-.Leiden, mice)\n\n")

cat(sprintf("Semaglutide DIO mice dataset: %d observations, %d publications, %d clinical trials\n\n",
            nrow(df_sema), n_distinct(df_sema$pmcid), n_distinct(df_sema$NCT_Number)))

## Duration descriptives
dur_desc <- df_sema %>%
  distinct(preclinical_arm_id, .keep_all = TRUE) %>%
  summarise(
    n_arms = n(),
    mean   = round(mean(pre_dur_days), 1),
    sd     = round(sd(pre_dur_days), 1),
    median = median(pre_dur_days),
    q25    = quantile(pre_dur_days, 0.25),
    q75    = quantile(pre_dur_days, 0.75),
    min    = min(pre_dur_days),
    max    = max(pre_dur_days)
  )
cat("Duration (days) — unique preclinical arms:\n")
print(as.data.frame(dur_desc))
cat("\n")

## Standardize numeric predictors (covariates only; pre_dur_days kept in raw days)
df_sema[paste0(global_cols, "_z")] <- scale(df_sema[global_cols])

safe_scale <- function(x) {
  s <- sd(x, na.rm = TRUE)
  if (is.na(s) || s == 0) return(x - mean(x, na.rm = TRUE))
  as.numeric(scale(x))
}
df_sema <- df_sema %>%
  mutate(across(all_of(drug_cols), safe_scale, .names = "{.col}_z"))

## =============================================================================
## Helper: fit LMER with optimizer fallback
## =============================================================================

fit_lmer_reml <- function(fml, data = df_sema, opt = NULL) {
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

fit_lmer_ml <- function(fml, data = df_sema, opt) {
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
  nlevs <- n_distinct(df_sema[[col]], na.rm = TRUE)
  if (nlevs > 1) {
    included_cats <- c(included_cats, cat_candidates[[col]])
    cat(sprintf("Including %s as fixed effect (%d levels)\n", col, nlevs))
  } else {
    cat(sprintf("%s excluded (only 1 level: %s)\n", col,
                as.character(unique(df_sema[[col]])[1])))
  }
}
cat("\n")

fe_terms <- c(
  included_cats,
  "pre_age_treat_d_z",
  "pre_age_treat_d_missing",
  "pre_dose_mgkg_log_z",
  "pre_n_z",
  "clin_dose_mg_z",
  "clin_dur_days_z",
  "clinical_sample_size_z"
)

make_formula <- function(fe, re) {
  paste(c("y ~ pre_dur_days", fe, re), collapse = " + ")
}

## =============================================================================
## PRE-SCREENING: Exclude covariates near-collinear with pre_dur_days
##
## Covariates strongly associated with the estimand act as suppressor variables,
## inflating the duration coefficient. Threshold: η (correlation ratio from
## one-way ANOVA of pre_dur_days ~ covariate) > 0.7 (categorical) or
## |r| > 0.7 (numeric, Pearson with pre_dur_days).
## =============================================================================

cat("================================================================================\n")
cat("PRE-SCREENING: Collinearity with pre_dur_days\n")
cat("================================================================================\n\n")
cat("Categorical: excluded if η (correlation ratio from ANOVA) > 0.7.\n")
cat("Numeric:     excluded if |r| > 0.7 (Pearson with pre_dur_days).\n\n")

screen_results <- lapply(fe_terms, function(term) {
  col_name <- sub("_z$", "", term)
  if (!col_name %in% names(df_sema)) {
    return(data.frame(term = term, col = col_name, type = "unknown",
                      criterion = NA_character_, value = NA_character_,
                      exclude = FALSE, stringsAsFactors = FALSE))
  }
  vals <- df_sema[[col_name]]
  if (is.factor(vals) || is.character(vals)) {
    fit_aov <- tryCatch(
      aov(df_sema$pre_dur_days ~ droplevels(factor(vals))),
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
    r <- abs(cor(as.numeric(vals), df_sema$pre_dur_days, use = "complete.obs"))
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
  cat(sprintf("Excluding (near-collinear with pre_dur_days): %s\n\n",
              paste(excluded_by_screen, collapse = ", ")))
  fe_terms <- fe_terms[!fe_terms %in% excluded_by_screen]
} else {
  cat("No covariates flagged — all retained in model.\n\n")
}

## Alias check: iteratively detect and remove perfectly collinear covariates.
## alias()$Complete returns model-matrix column names (e.g. "pre_sexUnknown"),
## which are mapped back to their parent fe_term (e.g. "pre_sex") before removal.
n_alias_removed <- 0L
repeat {
  fml_alias_check <- as.formula(
    paste("y ~ pre_dur_days +", paste(fe_terms, collapse = " + "))
  )
  m_alias_check <- tryCatch(lm(fml_alias_check, data = df_sema), error = function(e) NULL)
  if (is.null(m_alias_check)) break
  ali <- alias(m_alias_check)$Complete
  if (is.null(ali) || nrow(ali) == 0) break
  aliased_cols <- rownames(ali)
  ## Map aliased column name → parent variable in fe_terms
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
## pre_dur_days). Computed on equivalent lm() for the fixed-effects structure.
## =============================================================================

VIF_THRESHOLD <- 10L

get_vif_df <- function(fe) {
  fml <- as.formula(paste("y ~ pre_dur_days +", paste(fe, collapse = " + ")))
  m   <- tryCatch(lm(fml, data = df_sema), error = function(e) NULL)
  if (is.null(m)) return(NULL)
  v   <- tryCatch(car::vif(m), error = function(e) NULL)
  if (is.null(v)) return(NULL)
  if (is.vector(v)) {
    ## All 1-df terms: car::vif() returns a named numeric vector
    data.frame(VIF_equiv = round(v, 3), row.names = names(v))
  } else {
    ## Some multi-df terms: returns matrix with GVIF, Df, GVIF^(1/(2*Df))
    df_v <- as.data.frame(v)
    colnames(df_v)[3] <- "GVIF_1_2Df"
    df_v$VIF_equiv <- round(df_v$GVIF_1_2Df^2, 3)
    df_v
  }
}

vif_fe_terms <- fe_terms
vif_removed  <- character(0)

repeat {
  df_v <- get_vif_df(vif_fe_terms)
  if (is.null(df_v)) break
  cov_rows <- setdiff(rownames(df_v), "pre_dur_days")
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
  m_lm_vif <- lm(fml_lm_vif, data = df_sema)
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
## STEP 3: DURATION EFFECT — Wald t-test and LRT
## =============================================================================

cat("================================================================================\n")
cat("STEP 3: DURATION EFFECT — WALD t-TEST AND LRT\n")
cat("================================================================================\n\n")

ci_final  <- confint(model_final, method = "Wald", level = 0.95)
dur_term  <- "pre_dur_days"
beta <- NA; se <- NA; pval <- NA; ci_l <- NA; ci_u <- NA

if (dur_term %in% rownames(coefs_final)) {
  beta <- coefs_final[dur_term, "Estimate"]
  se   <- coefs_final[dur_term, "Std. Error"]
  pval <- coefs_final[dur_term, "Pr(>|t|)"]
  ci_l <- ci_final[dur_term, "2.5 %"]
  ci_u <- ci_final[dur_term, "97.5 %"]

  cat("DURATION EFFECT (primary result)\n")
  cat("---------------------------------\n")
  cat(sprintf("  Term: %s\n", dur_term))
  cat(sprintf("  β = %.4f per day,  SE = %.4f\n", beta, se))
  cat(sprintf("  95%% CI (Wald): [%.4f, %.4f]\n", ci_l, ci_u))
  cat(sprintf("  Wald p = %s %s\n", format_p(pval), sig_stars(pval)))
}

## LRT for duration (REML = FALSE comparison)
cat("\nLRT: DURATION FIXED EFFECT\n")
cat("---------------------------\n")

fml_full_ml_dur <- make_formula(selected_fe, re_terms)
fml_null_ml_dur <- sub("y ~ pre_dur_days \\+ ", "y ~ ", fml_full_ml_dur)
if (fml_null_ml_dur == fml_full_ml_dur) {
  fml_null_ml_dur <- paste(c("y ~ 1", re_terms), collapse = " + ")
}

lrt_p <- NA; lrt_chi <- NA; lrt_df <- NA

m_full_ml_dur <- fit_lmer_ml(fml_full_ml_dur, opt = opt_final)
m_null_ml_dur <- fit_lmer_ml(fml_null_ml_dur, opt = opt_final)

if (!is.null(m_full_ml_dur) && !is.null(m_null_ml_dur)) {
  ll_full <- as.numeric(logLik(m_full_ml_dur))
  ll_null <- as.numeric(logLik(m_null_ml_dur))
  ll_diff <- ll_full - ll_null
  lrt     <- anova(m_null_ml_dur, m_full_ml_dur)
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

if (dur_term %in% rownames(coefs_final)) {
  ref_p <- if (!is.na(lrt_p)) lrt_p else pval
  cat(sprintf("  Wald p = %s %s\n", format_p(pval), sig_stars(pval)))
  cat(sprintf("  LRT   p = %s %s  (preferred)\n\n", format_p(lrt_p), sig_stars(lrt_p)))
  if (!is.na(ref_p) && ref_p >= 0.05) {
    cat("-> NO SIGNIFICANT duration effect (Semaglutide DIO — Mice)\n\n")
  } else {
    cat("-> SIGNIFICANT duration effect: longer preclinical treatment is associated\n")
    cat("   with a larger translational gap.\n\n")
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

## --- Diagnostic plots ---
plot_file <- "sema_dio_duration_mice_lmer_robust_diagnostics.pdf"
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

if (dur_term %in% rownames(coefs_final)) {
  result_row <- data.frame(
    Drug              = "Semaglutide",
    Model             = "DIO",
    Species           = "mice",
    Final_formula     = final_formula,
    Selected_RE       = paste(re_terms, collapse = " + "),
    Selected_FE       = paste(c("pre_dur_days", selected_fe), collapse = " + "),
    N_obs             = nrow(df_sema),
    N_studies         = n_distinct(df_sema$pmcid),
    N_trials          = n_distinct(df_sema$NCT_Number),
    ICC_pmcid         = round(icc_pmcid, 4),
    ICC_NCT           = round(icc_nct, 4),
    Duration_term     = dur_term,
    Beta_per_day      = round(beta, 5),
    SE                = round(se, 5),
    CI_L_95           = round(ci_l, 5),
    CI_U_95           = round(ci_u, 5),
    Wald_p            = pval,
    LRT_chi2          = round(lrt_chi, 4),
    LRT_df            = lrt_df,
    LRT_p             = lrt_p,
    R2_marginal       = round(r2_m, 4),
    R2_conditional    = round(r2_c, 4),
    stringsAsFactors  = FALSE
  )
  write.csv(result_row, "sema_dio_duration_mice_lmer_robust_summary.csv", row.names = FALSE)
  cat("Results saved to: sema_dio_duration_mice_lmer_robust_summary.csv\n\n")
  print(result_row)
}

## =============================================================================
## DOSAGE AND AGE CONFOUNDING CHECK
## =============================================================================

cat("================================================================================\n")
cat("DOSAGE AND AGE CONFOUNDING CHECK\n")
cat("================================================================================\n\n")

sex_part   <- if ("pre_sex" %in% selected_fe) "pre_sex" else character(0)
route_part <- if ("preclinical_administration_route" %in% selected_fe)
               "preclinical_administration_route" else character(0)

fit_lmer_safe <- function(fml) {
  tryCatch(suppressWarnings(
    lmer(as.formula(fml), data = df_sema, REML = TRUE,
         control = lmerControl(optimizer = opt_final))),
    error = function(e) NULL)
}

## ---------------------------------------------------------------------------
## 1. Progressive β stability
## ---------------------------------------------------------------------------

cat("1. PROGRESSIVE β STABILITY (duration β as covariates are added)\n\n")

formula_m1 <- paste(c("y ~ pre_dur_days",
                       route_part, sex_part, re_terms),
                    collapse = " + ")
formula_m2 <- paste(c("y ~ pre_dur_days",
                       route_part, "pre_dose_mgkg_log_z",
                       sex_part, re_terms),
                    collapse = " + ")
formula_m3 <- paste(c("y ~ pre_dur_days",
                       route_part, "pre_dose_mgkg_log_z",
                       "pre_age_treat_d_z", "pre_age_treat_d_missing",
                       sex_part, re_terms),
                    collapse = " + ")

m1 <- fit_lmer_safe(formula_m1)
m2 <- fit_lmer_safe(formula_m2)
m3 <- fit_lmer_safe(formula_m3)

extract_dur_row <- function(fit, label) {
  if (is.null(fit))
    return(data.frame(Model = label, Beta = NA, SE = NA, Wald_p = NA,
                      stringsAsFactors = FALSE))
  cf <- summary(fit)$coefficients
  if (!dur_term %in% rownames(cf))
    return(data.frame(Model = label, Beta = NA, SE = NA, Wald_p = NA,
                      stringsAsFactors = FALSE))
  data.frame(
    Model  = label,
    Beta   = round(cf[dur_term, "Estimate"],   4),
    SE     = round(cf[dur_term, "Std. Error"], 4),
    Wald_p = round(cf[dur_term, "Pr(>|t|)"],  4),
    stringsAsFactors = FALSE
  )
}

stab <- rbind(
  extract_dur_row(m1,          "Duration only (+ route, sex)"),
  extract_dur_row(m2,          "+ log(dose)"),
  extract_dur_row(m3,          "+ log(dose) + age"),
  extract_dur_row(model_final, "Final selected model")
)
print(stab, row.names = FALSE)
cat("\n")

if (!is.na(stab$Beta[1]) && nrow(stab) >= 4 &&
    !is.na(stab$Beta[3]) && !is.na(stab$Beta[4]) && stab$Beta[1] != 0) {
  pct_dose_age <- abs((stab$Beta[3] - stab$Beta[1]) / stab$Beta[1]) * 100
  pct_final    <- abs((stab$Beta[4] - stab$Beta[3]) / stab$Beta[1]) * 100
  cat(sprintf("  β shift from adding dose + age:           %.1f%% (%+.4f → %+.4f)\n",
              pct_dose_age, stab$Beta[1], stab$Beta[3]))
  cat(sprintf("  β shift from adding further covariates:   %.1f%% (%+.4f → %+.4f)\n",
              pct_final, stab$Beta[3], stab$Beta[4]))
  cat("\n")
  if (pct_dose_age < 10) {
    cat("-> Duration effect ROBUST to dose and age adjustment (<10% change).\n\n")
  } else if (pct_dose_age < 30) {
    cat("-> Duration effect MODESTLY sensitive to dose/age adjustment (10–30% change).\n\n")
  } else {
    cat("-> Duration effect shows SUBSTANTIAL sensitivity to dose/age adjustment (>30% change).\n\n")
  }
}

## ---------------------------------------------------------------------------
## 2. Interaction LRTs
## ---------------------------------------------------------------------------

cat("2. INTERACTION TESTS: does log(dose) or age moderate the duration effect?\n\n")

fit_lmer_ml2 <- function(fml) {
  tryCatch(suppressWarnings(
    lmer(as.formula(fml), data = df_sema, REML = FALSE,
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

lrt_interaction("pre_dur_days:pre_dose_mgkg_log_z", "Duration × log(dose)")
lrt_interaction("pre_dur_days:pre_age_treat_d_z",   "Duration × age")
lrt_interaction("pre_dur_days:clin_dur_days_z",      "Duration × clinical duration")
cat("\n")
cat("  Non-significant interactions (p > 0.05) indicate that dosage, age,\n")
cat("  and clinical duration do not moderate the preclinical duration effect.\n\n")

## =============================================================================
## SENSITIVITY: LEAVE-ONE-PUBLICATION-OUT (LOPO)
## =============================================================================

cat("================================================================================\n")
cat("SENSITIVITY: LEAVE-ONE-PUBLICATION-OUT (LOPO)\n")
cat("================================================================================\n\n")
cat(sprintf("Refitting final selected model leaving out each of the %d publications.\n\n",
            n_distinct(df_sema$pmcid)))

pubs <- levels(df_sema$pmcid)

fml_null_lopo <- sub("y ~ pre_dur_days \\+ ", "y ~ ", final_formula)
if (fml_null_lopo == final_formula)
  fml_null_lopo <- paste(c("y ~ 1", re_terms), collapse = " + ")

lopo_results <- lapply(pubs, function(pub) {
  df_loo <- df_sema %>% filter(pmcid != pub) %>% droplevels()

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
  if (!dur_term %in% rownames(cf))
    return(data.frame(dropped_pub = pub, N_obs = nrow(df_loo),
                      Beta = NA, SE = NA, Wald_p = NA, LRT_p = NA,
                      stringsAsFactors = FALSE))

  b_l  <- cf[dur_term, "Estimate"]
  se_l <- cf[dur_term, "Std. Error"]
  p_l  <- cf[dur_term, "Pr(>|t|)"]

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
    Beta        = round(b_l,      4),
    SE          = round(se_l,     4),
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
cat(sprintf("  Final-model β       = %+.4f\n", beta))
cat(sprintf("  LOPO models (valid) = %d / %d\n\n", length(beta_lopo), nrow(lopo_df)))
if (length(beta_lopo) > 0) {
  cat(sprintf("  Mean    = %+.4f\n",   mean(beta_lopo)))
  cat(sprintf("  Median  = %+.4f\n",   median(beta_lopo)))
  cat(sprintf("  SD      =  %.4f\n",   sd(beta_lopo)))
  cat(sprintf("  Q25–Q75 = [%+.4f, %+.4f]\n",
              quantile(beta_lopo, 0.25), quantile(beta_lopo, 0.75)))
  cat(sprintf("  Range   = [%+.4f, %+.4f]\n", min(beta_lopo), max(beta_lopo)))
  cat(sprintf("  N positive (β > 0) = %d / %d\n\n",
              sum(beta_lopo > 0), length(beta_lopo)))
}

lopo_df$beta_shift <- abs(lopo_df$Beta - beta)
most_inf <- lopo_df[which.max(lopo_df$beta_shift), ]
cat(sprintf("  Most influential publication: %s\n", most_inf$dropped_pub))
cat(sprintf("  β without it: %.4f (shift = %+.4f from final-model β = %.4f)\n\n",
            most_inf$Beta, most_inf$Beta - beta, beta))

cat("================================================================================\n")
cat("SEMAGLUTIDE DIO DURATION LMER ROBUST ANALYSIS (MICE ONLY) COMPLETE\n")
cat("================================================================================\n")
