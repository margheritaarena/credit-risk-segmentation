## ============================================================
## PROJECT: Customer Risk & Value Segmentation
## 2 Feature Engineering
## Input: data/processed/loans_clean.rds (from 01 setup and cleaning.R)
## ============================================================

library(tidyverse)
library(here)

packages <- c("scorecard")
new_packages <- packages[!(packages %in% installed.packages()[, "Package"])]
if (length(new_packages) > 0) install.packages(new_packages)

library(scorecard)

loans_clean <- readRDS(here("data", "processed", "loans_clean.rds"))


## ------------------------------------------------------------
## DERIVED FINANCIAL VARIABLES
## ------------------------------------------------------------

# DTI (Debt-to-Income) is already provided by Lending Club as `dti`. 
# Check its scale and quality before using it.
summary(loans_clean$dti)

# dti values <0 or >100 have no established meaning in Lending Club's official data dictionary and are not confirmed as 
# coded missing/NA values: they are treated as data quality errors rather than genuine extreme observations. 
# Given the dataset size (1.3M+ rows), affected rows are dropped rather than imputed, to avoid introducing an arbitrary 
# assumption about their "true" value. All derived variables are built in a single chained pipeline starting from loans_clean, 
# so no intermediate step gets accidentally overwritten.

n_before <- nrow(loans_clean)

loans_features <- loans_clean %>%
  filter(dti >= 0 & dti <= 100) %>%
  mutate(
    # Credit utilization rate: already present as `revol_util` (%
    # of revolving credit limit currently used). Make sure it's
    # numeric first (sometimes imported as character with a
    # trailing "%").
    revol_util = if (is.character(revol_util)) {
      as.numeric(str_remove(revol_util, "%"))
    } else {
      revol_util
    }
  ) %>%
  filter(
    # annual_inc == 0 (or near-zero) is economically implausible
    # for a standalone applicant and distorts loan_to_income by
    # near-zero division: treated as a data quality issue,
    # consistent with the dti approach.
    annual_inc >= 1000,
    # revol_util above 150% has no plausible real-world
    # interpretation (utilization can exceed 100% due to accrued
    # interest, but not by this margin), same reasoning as dti.
    revol_util <= 150 | is.na(revol_util)
  ) %>%
  mutate(
    # Loan-to-Income ratio: how much the loan "weighs" relative to
    # the borrower's annual income. Higher values = higher relative
    # burden. Safe now that annual_inc >= 1000 is enforced above.
    loan_to_income = loan_amnt / annual_inc
  ) %>%
  mutate(
    # Employment stability: bucket emp_length into a small number
    # of meaningful groups instead of using the raw categorical
    # string (proxy for income stability: a standard feature in
    # credit scoring models).
    emp_length_years = case_when(
      emp_length == "< 1 year" ~ 0,
      emp_length == "10+ years" ~ 10,
      is.na(emp_length) | emp_length == "Unknown" ~ NA_real_,
      TRUE ~ as.numeric(str_extract(emp_length, "\\d+"))
    ),
    employment_stability = case_when(
      is.na(emp_length_years) ~ "Unknown",
      emp_length_years < 2 ~ "Low (<2y)",
      emp_length_years < 5 ~ "Medium (2-5y)",
      TRUE ~ "High (5y+)"
    )
  ) %>%
  mutate(
    # home_ownership: ANY/NONE/OTHER are rare categories (a few
    # hundred rows combined) that risk producing unstable WOE bins,
    # merged into a single "OTHER" bucket.
    home_ownership = if_else(
      home_ownership %in% c("ANY", "NONE", "OTHER"),
      "OTHER", home_ownership
    )
  )

n_after <- nrow(loans_features)
cat("Rows dropped (invalid dti, annual_inc, or revol_util):",
    n_before - n_after,
    "(", round((n_before - n_after) / n_before * 100, 3), "%)\n")

# Verify after correction
summary(loans_features$dti)
summary(loans_features$annual_inc)
summary(loans_features$revol_util)
count(loans_features, home_ownership)

# Loan grade / subgrade: already present in the dataset ("grade",
# "sub_grade") as Lending Club's own risk rating. Keep it as is,
# it will be a very useful benchmark in Phase 4 (compare the
# model's discrimination power against LC's own grading).


## ------------------------------------------------------------
# DATA QUALITY CHECK ON CANDIDATE VARIABLES
## ------------------------------------------------------------

# Same logic applied to dti: before trusting any variable for modeling, check its range for implausible values
# (not just statistical outliers, but values that make no economic sense: negative balances, sentinel codes like -1/999, 
# percentages outside 0-100, etc.). This is a systematic pass across all numeric candidates, not a one-variable-at-a-time inspection.

numeric_candidates <- c(
  "loan_amnt", "int_rate", "installment", "annual_inc", "dti",
  "delinq_2yrs", "open_acc", "pub_rec", "revol_bal", "revol_util",
  "total_acc", "loan_to_income", "emp_length_years"
)

quality_check <- loans_features %>%
  select(all_of(numeric_candidates)) %>%
  map_dfr(function(x) {
    tibble(
      n_missing = sum(is.na(x)),
      pct_missing = round(mean(is.na(x)) * 100, 2),
      min = suppressWarnings(min(x, na.rm = TRUE)),
      p01 = quantile(x, 0.01, na.rm = TRUE),
      median = median(x, na.rm = TRUE),
      p99 = quantile(x, 0.99, na.rm = TRUE),
      max = suppressWarnings(max(x, na.rm = TRUE)),
      n_negative = sum(x < 0, na.rm = TRUE)
    )
  }, .id = "variable")

quality_check

# Flags to look at manually:
# - n_negative > 0 on a variable that should never be negative (e.g. revol_util, annual_inc, revol_bal)
# - max far above p99 (classic sign of a sentinel/placeholder value, exactly like what happened with dti)
# - revol_util outside a plausible 0-150 range (utilization can technically exceed 100% but not by an absurd margin)
# - unusually high pct_missing on a variable you plan to rely on.

quality_check %>%
  mutate(max_to_p99_ratio = round(max / p99, 1)) %>%
  arrange(desc(max_to_p99_ratio))

# Categorical candidates: check for unexpected/rare categories (e.g. typos, placeholder strings) before they go into woebin()

categorical_candidates <- c(
  "term", "grade", "sub_grade", "home_ownership",
  "verification_status", "purpose", "employment_stability"
)

walk(categorical_candidates, function(v) {
  cat("\n---", v, "---\n")
  print(loans_features %>% count(.data[[v]], sort = TRUE))
})


## ------------------------------------------------------------
## WOE (Weight of Evidence) AND IV (Information Value)
## ------------------------------------------------------------

# WOE/IV is the standard approach in real-world credit scorecards (banks, rating agencies). 
# It bins each variable, then measures how well each bin separates good payers from defaulters.

# IV interpretation (industry rule of thumb):
#   < 0.02        -> not predictive, drop
#   0.02 - 0.1    -> weak predictor
#   0.1  - 0.3    -> medium predictor
#   0.3  - 0.5    -> strong predictor
#   > 0.5         -> suspicious, check for data leakage / redundancy

# Select candidate predictors for the scorecard step. Drop columns that are IDs, dates, free text, or that would leak the outcome
# (e.g. recoveries, total_pymnt happen AFTER the loan is issued).

candidate_vars_v1 <- loans_features %>%
  select(
    default,
    loan_amnt, term, int_rate, installment, grade, sub_grade,
    emp_length_years, home_ownership, annual_inc, verification_status,
    purpose, dti, delinq_2yrs, open_acc, pub_rec, revol_bal,
    revol_util, total_acc, loan_to_income, employment_stability
  )

# Optimal binning based on WOE, using default as the target.
# NOTE: woebin() automatically drops near-constant columns: delinq_2yrs and pub_rec were removed here (>98% of values equal
# to 0, consistent with the earlier data quality check), which also explains their IV of 0 below.

bins_v1 <- woebin(candidate_vars_v1, y = "default")

iv_summary_v1 <- iv(candidate_vars_v1, y = "default") %>%
  arrange(desc(info_value))

iv_summary_v1


## ------------------------------------------------------------
## MULTICOLLINEARITY CHECK ON TOP PREDICTORS
## ------------------------------------------------------------

# The first-pass IV table flags installment (0.63), int_rate (0.53), sub_grade (0.50), and grade (0.46) as very strong, high enough
# to be suspicious (see >0.5 rule of thumb above). This is not temporal data leakage (int_rate/installment/grade are all known
# at loan origination, unlike recoveries/total_pymnt), but it is a sign of REDUNDANCY: these variables may be encoding the same
# underlying risk signal multiple times, which would inflate coefficient variance in the Phase 4 logistic regression.

# Check 1: installment vs loan_amnt vs int_rate (term converted to numeric only for this check)

loans_features %>%
  mutate(term_months = as.numeric(str_extract(term, "\\d+"))) %>%
  select(int_rate, installment, loan_amnt, term_months) %>%
  cor(use = "complete.obs")

# Result: installment ~ loan_amnt correlation = 0.957 -> near-duplicate information. 
# int_rate correlates only ~0.14-0.15 with both -> NOT redundant with loan amount/installment as initially suspected.

# Check 2: is int_rate essentially determined by sub_grade?

loans_features %>%
  group_by(sub_grade) %>%
  summarise(mean_int_rate = mean(int_rate), sd_int_rate = sd(int_rate)) %>%
  print(n = 35)

# Result: within-subgrade sd is small relative to the gap between consecutive subgrades for low-risk grades (A-C), 
# confirming int_rate is largely determined by sub_grade. sd increases noticeably for riskier grades (E-G, up to ~2.6), 
# meaning int_rate still carries some residual signal beyond sub_grade, the overlap is strong but not a perfect duplicate.

# CONCLUSION — two independent redundancy pairs identified:
#   1. installment <-> loan_amnt (corr. 0.957): near-duplicate, keep loan_amnt (more primary/interpretable)
#   2. int_rate <-> grade/sub_grade: sub_grade explains most, but not all, of int_rate's variance (residual signal grows with risk).
#      Conservative choice: keep sub_grade only, drop int_rate and grade, to avoid inflating coefficient variance in Phase 4. 
#      (Alternative for later: fit two candidate models, one with sub_grade and one with int_rate, and compare
#      out-of-sample performance, noted as a possible extension.)


## ------------------------------------------------------------
## FINAL PREDICTOR SELECTION
## ------------------------------------------------------------

# Final list, informed by:
# - IV from the first pass (dropped: emp_length_years/employment_stability, open_acc, total_acc — IV < 0.02;
#   delinq_2yrs, pub_rec — IV = 0, already excluded by woebin())
# - Multicollinearity check above (dropped: installment, grade, int_rate, redundant with loan_amnt / sub_grade)

candidate_vars <- loans_features %>%
  select(
    default,
    loan_amnt, term, sub_grade, annual_inc, verification_status,
    purpose, dti, revol_bal, revol_util, loan_to_income,
    home_ownership
  )

# Re-run WOE binning and IV on the final, non-redundant predictor set
bins <- woebin(candidate_vars, y = "default")

# Information Value summary, sorted.
# IMPORTANT: iv() and woebin() can produce different bin breakpoints for continuous variables when called independently,
# even on the same data (categorical variables are largely unaffected, few natural categories, but continuous ones can
# diverge a lot). iv() does not accept a breaks_list argument to force consistency, so instead the IV is extracted directly from
# the "bins" object itself via its "total_iv" column, this is guaranteed to match the bins actually applied downstream (woebin_ply) and shown in woebin_plot.

iv_summary <- map_dfr(bins, function(b) {
  tibble(info_value = unique(b$total_iv))
}, .id = "variable") %>%
  arrange(desc(info_value))

iv_summary

# purpose (0.0159) and revol_bal (0.0038) fall below the IV > 0.02 "not predictive" threshold once measured on the bins actually
# used downstream (they had looked stronger, 0.019 and 0.175 respectively, under iv()'s own independent binning, another
# instance of the iv()/woebin() discrepancy above). Dropped from the final predictor set below.

candidate_vars <- loans_features %>%
  select(
    default,
    loan_amnt, term, sub_grade, annual_inc, verification_status,
    dti, revol_util, loan_to_income, home_ownership
  )

bins <- woebin(candidate_vars, y = "default")

iv_summary <- map_dfr(bins, function(b) {
  tibble(info_value = unique(b$total_iv))
}, .id = "variable") %>%
  arrange(desc(info_value))

iv_summary

# Visual inspection of the top predictors.

woebin_plot(bins[c("sub_grade", "loan_to_income", "dti")])


## ------------------------------------------------------------
## APPLY WOE TRANSFORMATION
## ------------------------------------------------------------

# Convert original variables into their WOE values. 
# This is what feeds a scorecard-style logistic regression in Phase 4, using WOE-transformed variables makes the logistic regression more
# stable and standard practice in banking (monotonic relationship with the target, easier regulatory interpretation).

loans_woe <- woebin_ply(candidate_vars, bins)


## ------------------------------------------------------------
## SAVE OUTPUTS
## ------------------------------------------------------------


saveRDS(loans_features, here("data", "processed", "loans_features.rds"))
saveRDS(bins, here("data", "processed", "woe_bins.rds"))
saveRDS(loans_woe, here("data", "processed", "loans_woe.rds"))

write_csv(iv_summary, here("outputs", "iv_summary.csv"))

# Summary

cat(
  "Number of candidate predictors evaluated (first pass):", ncol(candidate_vars_v1) - 1, "\n",
  "Number of final predictors after multicollinearity + corrected IV check:", ncol(candidate_vars) - 1, "\n",
  "Predictors with IV > 0.1 (medium/strong):",
  sum(iv_summary$info_value > 0.1), "\n",
  "Top 3 predictors by IV (final set):\n"
)
print(head(iv_summary, 3))

## ------------------------------------------------------------
## NEXT STEP: Phase 3 EDA
## ------------------------------------------------------------
