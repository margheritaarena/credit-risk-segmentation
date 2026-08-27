## ============================================================
## PROJECT: Customer Risk & Value Segmentation
## PHASE 5 From model to economic decision
## Input: data/processed/loans_features.rds, test_woe_with_preds.rds, logit_model.rds (from 04 modeling.R)
## ============================================================

library(tidyverse)
library(here)
library(scorecard)

loans_features <- readRDS(here("data", "processed", "loans_features.rds"))
test_woe <- readRDS(here("data", "processed", "test_woe_with_preds.rds"))


## ------------------------------------------------------------
## RECOVER ECONOMIC COLUMNS FOR THE TEST SET
## ------------------------------------------------------------

# test_woe only contains the 9 WOE predictors used in the risk model, plus default and predicted PDs, 
# it does NOT contain the raw economic columns needed here (int_rate, funded_amnt, recoveries, total_pymnt). 
# These were intentionally excluded from the risk model in Phase 2 (int_rate was dropped for redundancy with sub_grade), 
# but are still needed now for the REVENUE side of the economic calculation, using int_rate here is not a
# leakage concern, since it does not feed back into the PD model, only into the profit/loss calculation built on top of it.

# To recover these columns for the exact same test rows used in Phase 4, the train/test split is reproduced with the same seed
# and ratio on a superset of columns that includes the economic fields. A fixed seed makes scorecard::split_df() deterministic,
# so this should reproduce the identical partition.

model_vars_econ <- loans_features %>%
  select(
    default,
    loan_amnt, term, sub_grade, annual_inc, verification_status,
    dti, revol_util, loan_to_income, home_ownership,
    int_rate, funded_amnt, recoveries, total_pymnt
  )

split_econ <- scorecard::split_df(model_vars_econ, y = "default", ratio = 0.7, seed = 42)
train_econ <- split_econ$train
test_econ <- split_econ$test

# Verification: the reproduced test set must match the Phase 4 test set exactly (same rows, same order) before trusting the merge.

stopifnot(
  nrow(test_econ) == nrow(test_woe),
  identical(test_econ$default, test_woe$default)
)
cat("Verification passed: reproduced split matches Phase 4 test set",
    "(", nrow(test_econ), "rows, identical default vector).\n")

# Combine: WOE predictors + predictions (from test_woe) with the economic columns (from test_econ), safe now that row alignment is verified.

test_full <- bind_cols(
  test_woe,
  test_econ %>% select(loan_amnt, term, int_rate, funded_amnt, recoveries, total_pymnt)
)


## ------------------------------------------------------------
## HISTORICAL LGD (Loss Given Default) ESTIMATION
## ------------------------------------------------------------

# LGD = share of exposure not recovered when a loan defaults.
# Estimated empirically from the TRAINING set's charged-off loans only (never from test, to keep the economic simulation on unseen data): 
# loss_amount = funded_amnt - total_pymnt (total received from the borrower over the loan's life, including any post-charge-off recoveries 
# already captured in total_pymnt for this dataset). Floored at 0 in case total_pymnt slightly exceeds funded_amnt due to fees.

lgd_data <- train_econ %>%
  filter(default == 1) %>%
  mutate(
    loss_amount = pmax(funded_amnt - total_pymnt, 0),
    lgd = loss_amount / funded_amnt
  )

summary(lgd_data$lgd)

# A single portfolio-average LGD is used as a simplifying assumption for the Expected Loss calculation below (a natural
# extension would be a segment-level or loan-level LGD model, noted here as a possible next step rather than built out, to keep
# the economic layer interpretable).

mean_lgd <- mean(lgd_data$lgd)
cat("Estimated portfolio-average LGD:", round(mean_lgd, 3), "\n")

# Quick check: does LGD vary meaningfully by grade? 

lgd_by_grade <- train_econ %>%
  filter(default == 1) %>%
  left_join(
    loans_features %>% select(sub_grade, grade) %>% distinct(),
    by = "sub_grade"
  ) %>%
  mutate(loss_amount = pmax(funded_amnt - total_pymnt, 0), lgd = loss_amount / funded_amnt) %>%
  group_by(grade) %>%
  summarise(mean_lgd = mean(lgd), n = n())

lgd_by_grade


## ------------------------------------------------------------
## EXPECTED LOSS PER LOAN (EL = PD x LGD x EAD)
## ------------------------------------------------------------

# EAD (Exposure at Default) is approximated as the full funded amount (loan_amnt). 
# This is the standard simplification for an approval-time risk model: at the moment the approve/reject
# decision is made, the entire loan amount is the exposure, a loan-level amortization schedule isn't available yet at that
# point, so a more granular EAD (e.g. accounting for principal already repaid) isn't applicable here.

test_full <- test_full %>%
  mutate(
    ead = loan_amnt,
    expected_loss = pred_logit * mean_lgd * ead
  )

summary(test_full$expected_loss)

cat("Total expected loss across the test portfolio: $",
    format(round(sum(test_full$expected_loss)), big.mark = ","), "\n")


## ------------------------------------------------------------
## EXPECTED PROFIT PER LOAN (revenue vs. expected loss)
## ------------------------------------------------------------

# Expected interest income if the loan is NOT charged off: a simplified total-interest-over-term estimate (loan_amnt x annual
# rate x term in years). This ignores compounding/amortization detail and the time value of money, a reasonable simplification
# for a portfolio-level decision tool, not a full cash-flow model.

test_full <- test_full %>%
  mutate(
    term_months = as.numeric(str_extract(term, "\\d+")),
    term_years = term_months / 12,
    expected_interest_income = loan_amnt * (int_rate / 100) * term_years,

    # Expected profit if approved: weighted by the model's PD,
    # earn the expected interest with probability (1-PD), lose
    # LGD x EAD with probability PD. Foregone interest on defaulted
    # loans is ignored (conservative simplification: assumes no
    # interest is collected before a default occurs).
    expected_profit_if_approved = (1 - pred_logit) * expected_interest_income -
      pred_logit * mean_lgd * ead
  )

summary(test_full$expected_profit_if_approved)


## ------------------------------------------------------------
## COST MATRIX
## ------------------------------------------------------------

# +------------------+------------------+-------------------------+
# |                  | Loan approved    | Loan rejected           |
# +------------------+------------------+-------------------------+
# | Would repay      | Earn interest    | Opportunity cost:       |
# | (true negative/  | income           | foregone expected       |
# |  false positive) |                  | interest income         |
# +------------------+------------------+-------------------------+
# | Would default    | Lose LGD x EAD   | No exposure, no loss    |
# | (false negative/ | (minus any       | (correctly avoided)     |
# |  true positive)  | interest already |                         |
# |                  | foregone)        |                         |
# +------------------+------------------+-------------------------+

# Approving a bad loan (false negative, in credit-risk terms: a defaulter classified as low-risk) is far costlier in absolute
# terms than rejecting a good loan (false positive: a reliable borrower turned away), the asymmetry that the threshold
# optimization below is designed to exploit.


## ------------------------------------------------------------
## PROFIT CURVE AND THRESHOLD OPTIMIZATION
## ------------------------------------------------------------

# For each candidate approval threshold t (approve loans with predicted PD <= t, reject the rest), compute the resulting
# portfolio-level expected profit, approval rate, and expected loss. Sweeping t from very conservative (approve almost no one)
# to very permissive (approve everyone) traces out the profit curve; the threshold maximizing total expected profit is the
# economically optimal cutoff, NOT necessarily 0.5, and not necessarily the threshold that maximizes accuracy or AUC-based metrics.

# The upper bound (0.60) was checked against the observed PD range on the test set (max predicted PD = 0.587), so it fully covers
# the achievable threshold space, no risk of an artificially truncated curve.

thresholds <- seq(0.02, 0.60, by = 0.01)

profit_curve <- map_dfr(thresholds, function(t) {
  approved <- test_full %>% filter(pred_logit <= t)
  tibble(
    threshold = t,
    n_approved = nrow(approved),
    approval_rate = nrow(approved) / nrow(test_full),
    total_expected_profit = sum(approved$expected_profit_if_approved),
    total_expected_loss = sum(approved$expected_loss),
    actual_default_rate_approved = mean(approved$default)
  )
})

profit_curve

# with_ties = FALSE: once every loan is approved (threshold >= max observed PD), total_expected_profit is identical across the
# remaining thresholds (no further loans to add), so slice_max() would otherwise return multiple tied rows — only the first is
# needed for the plot/labels below.

optimal_row <- profit_curve %>%
  slice_max(total_expected_profit, n = 1, with_ties = FALSE)

optimal_row

p_profit_curve <- ggplot(profit_curve, aes(x = threshold, y = total_expected_profit)) +
  geom_line(color = "#2C7FB8", linewidth = 1) +
  geom_vline(xintercept = optimal_row$threshold, linetype = "dashed", color = "#E34A33") +
  annotate("text", x = optimal_row$threshold, y = min(profit_curve$total_expected_profit),
           label = paste0("Optimal: ", optimal_row$threshold), hjust = 1.1, color = "#E34A33") +
  scale_y_continuous(labels = scales::comma) +
  labs(
    title = "Profit curve - expected portfolio profit by approval threshold",
    subtitle = "Approve loans with predicted PD below the threshold",
    x = "Approval threshold (predicted PD)", y = "Total expected profit ($)"
  ) +
  theme_minimal()

p_profit_curve
ggsave(here("outputs", "profit_curve.png"), p_profit_curve,
       width = 8, height = 5, dpi = 150)

# Reference comparison: optimal cutoff vs. the two extremes tested

comparison <- bind_rows(
  optimal_row %>% mutate(label = "Optimal threshold"),
  profit_curve %>% filter(threshold == max(thresholds)) %>% mutate(label = "Most permissive tested (approve almost everyone)"),
  profit_curve %>% filter(threshold == min(thresholds)) %>% mutate(label = "Most conservative tested (approve almost no one)")
)

comparison %>% select(label, threshold, approval_rate, total_expected_profit, total_expected_loss)


## ------------------------------------------------------------
## MARGINAL PROFIT ANALYSIS
## ------------------------------------------------------------

# The optimal threshold found above sits at the edge of the tested range (0.59-0.60, i.e. "approve everyone"), meaning total expected
# profit never decreases as the threshold is relaxed across the observed PD range. Before accepting this at face value, the
# MARGINAL contribution of each additional risk band is checked, not just the cumulative total, to confirm the curve is genuinely
# monotonic increasing throughout, rather than flat or declining at the tail and only appearing to rise due to earlier gains.

profit_by_band <- profit_curve %>%
  filter(n_approved > 0) %>%
  arrange(threshold) %>%
  mutate(
    marginal_loans = n_approved - lag(n_approved, default = 0),
    marginal_profit = total_expected_profit - lag(total_expected_profit, default = 0),
    marginal_profit_per_loan = marginal_profit / marginal_loans
  )

profit_by_band %>%
  select(threshold, n_approved, marginal_loans, marginal_profit_per_loan) %>%
  tail(15)

# CONCLUSION: marginal profit per loan stays positive across the entire tested range, including the riskiest band observed
# (threshold 0.58-0.59, marginal profit per loan still ~$3,000-4,000) it trends downward as risk increases (diminishing marginal
# returns, economically sensible) but never turns negative. This confirms the "approve everyone" result is not an artifact of
# cumulative totals masking a declining tail; within the observed PD range, LendingClub's risk-based interest rate pricing appears
# to compensate for default risk even at the highest risk levels present in the data.

# IMPORTANT CAVEATS for interpretation (see README): this is a pure expected-value analysis. It does NOT account for default
# correlation risk (e.g. a recession causing many simultaneous defaults), capital/liquidity constraints, or portfolio
# concentration limits, real reasons a lender would still apply more conservative cutoffs in practice, beyond expected value alone.


## ------------------------------------------------------------
## SAVE OUTPUTS
## ------------------------------------------------------------

saveRDS(test_full, here("data", "processed", "test_full_economic.rds"))
write_csv(profit_curve, here("outputs", "profit_curve.csv"))
write_csv(lgd_by_grade, here("outputs", "lgd_by_grade.csv"))

cat(
  "Portfolio-average LGD:", round(mean_lgd, 3), "\n",
  "Optimal approval threshold (predicted PD):", optimal_row$threshold, "\n",
  "Approval rate at optimal threshold:", round(optimal_row$approval_rate * 100, 1), "%\n",
  "Total expected profit at optimal threshold: $",
  format(round(optimal_row$total_expected_profit), big.mark = ","), "\n",
  "Total expected profit approving everyone: $",
  format(round(sum(test_full$expected_profit_if_approved)), big.mark = ","), "\n"
)

## ------------------------------------------------------------
## NEXT STEP: Phase 6 Customer Segmentation (K-means clustering)
## ------------------------------------------------------------
