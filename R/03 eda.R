## ============================================================
## PROJECT: Customer Risk & Value Segmentation
## 3 EDA (Exploratory Data Analysis)
## Input: data/processed/loans_features.rds (from 02 feature engineering.R)
## ============================================================

library(tidyverse)
library(here)

packages <- c("corrplot", "scales")
new_packages <- packages[!(packages %in% installed.packages()[, "Package"])]
if (length(new_packages) > 0) install.packages(new_packages)

library(corrplot)
library(scales)

loans_features <- readRDS(here("data", "processed", "loans_features.rds"))

# NOTE: a large part of the exploratory work for this project was already done out of necessity during the Phase 2 data quality
# checks and WOE binning (distributions, missingness, outliers, and the relationship between dti/sub_grade/loan_to_income and the target). 
# This phase focuses on what hasn't been covered yet: overall target balance, relationships not shown by woebin_plot, a temporal view, 
# and clean exportable figures for the dashboard.


## ------------------------------------------------------------
## TARGET OVERVIEW
## ------------------------------------------------------------

# Overall default rate (already computed in Phase 1, confirmed here on the final cleaned dataset used for modeling)

loans_features %>%
  count(default) %>%
  mutate(pct = round(n / sum(n) * 100, 1))

# Default rate by grade: the "official" LC risk rating, useful as a benchmark against your own model's discrimination power later (Phase 4)

default_by_grade <- loans_features %>%
  group_by(grade) %>%
  summarise(
    n = n(),
    default_rate = mean(default) * 100
  ) %>%
  arrange(grade)

default_by_grade

p_default_by_grade <- ggplot(default_by_grade, aes(x = grade, y = default_rate)) +
  geom_col(fill = "#2C7FB8") +
  geom_text(aes(label = paste0(round(default_rate, 1), "%")), vjust = -0.5) +
  labs(
    title = "Default rate by Lending Club grade",
    x = "Grade", y = "Default rate (%)"
  ) +
  theme_minimal()

p_default_by_grade
ggsave(here("outputs", "default_rate_by_grade.png"), p_default_by_grade,
       width = 7, height = 5, dpi = 150)


## ------------------------------------------------------------
## LOAN PURPOSE AND AMOUNT
## ------------------------------------------------------------

# Distribution of loan purposes: useful context even though "purpose" itself was dropped from the model (IV < 0.02, Phase 2)

p_purpose <- loans_features %>%
  count(purpose, sort = TRUE) %>%
  mutate(purpose = fct_reorder(purpose, n)) %>%
  ggplot(aes(x = purpose, y = n)) +
  geom_col(fill = "#41AB5D") +
  coord_flip() +
  scale_y_continuous(labels = comma) +
  labs(title = "Number of loans by purpose", x = NULL, y = "Number of loans") +
  theme_minimal()

p_purpose
ggsave(here("outputs", "loans_by_purpose.png"), p_purpose,
       width = 7, height = 5, dpi = 150)

# Loan amount distribution, split by outcome: a quick visual check on whether defaulted loans tend to be larger 
# (economically plausible: larger loans -> larger monthly burden)

p_amount_by_outcome <- loans_features %>%
  mutate(outcome = if_else(default == 1, "Charged Off", "Fully Paid")) %>%
  ggplot(aes(x = loan_amnt, fill = outcome)) +
  geom_density(alpha = 0.5) +
  scale_x_continuous(labels = comma) +
  scale_fill_manual(values = c("Charged Off" = "#E34A33", "Fully Paid" = "#2C7FB8")) +
  labs(title = "Loan amount distribution by outcome", x = "Loan amount ($)", fill = NULL) +
  theme_minimal()

p_amount_by_outcome
ggsave(here("outputs", "loan_amount_by_outcome.png"), p_amount_by_outcome,
       width = 7, height = 5, dpi = 150)


## ------------------------------------------------------------
## CORRELATION AMONG FINAL NUMERIC PREDICTORS
## ------------------------------------------------------------

# Quick sanity check on the final predictor set selected in Phase 2 (after removing installment/grade/int_rate for redundancy)
# confirms no residual strong correlations remain among the numeric variables that made it into the model.

numeric_final <- loans_features %>%
  select(loan_amnt, annual_inc, dti, revol_util, loan_to_income)

corr_matrix <- cor(numeric_final, use = "complete.obs")
corr_matrix

png(here("outputs", "correlation_final_predictors.png"), width = 700, height = 700)
corrplot(corr_matrix, method = "color", type = "upper",
         addCoef.col = "black", tl.col = "black", tl.srt = 45,
         title = "Correlation among final numeric predictors", mar = c(0, 0, 2, 0))
dev.off()


## ------------------------------------------------------------
## TEMPORAL TREND
## ------------------------------------------------------------

# issue_d is typically in "Mon-YYYY" format (e.g. "Dec-2015") in the raw Lending Club export. 
# Parse it and look at how the default rate evolves over time: useful context for the
# dashboard's Overview tab, and a sanity check for macro effects (e.g. 2008-2009 financial crisis, if present in the data range).

if ("issue_d" %in% names(loans_features)) {
  
  loans_temporal <- loans_features %>%
    mutate(issue_date = my(issue_d)) %>%   # lubridate::my() parses "Mon-YYYY"
    filter(!is.na(issue_date))
  
  default_by_quarter <- loans_temporal %>%
    mutate(issue_quarter = floor_date(issue_date, "quarter")) %>%
    group_by(issue_quarter) %>%
    summarise(
      n = n(),
      default_rate = mean(default) * 100
    )
  
  # Diagnostic: check the most recent quarters before trusting the
  # raw trend. Loans are only included here if they already reached
  # a final outcome (Fully Paid/Charged Off, filtered in Phase 1):
  # loans still "Current" were excluded. Very recent loans haven't
  # had time to mature/default yet, so the final-outcome loans left
  # in the most recent quarters are a biased, fast-resolving subset
  # (mostly early payoffs), not a representative sample of that
  # vintage. This is a classic right-censoring effect in credit data.
  
  default_by_quarter %>%
    arrange(issue_quarter) %>%
    tail(10)
  
  # -> confirms the bias: n drops from 60,884 (2016-Q3) to 5,006
  #    (2018-Q4), a >90% collapse, with default_rate artificially
  #    falling to ~2% in the last quarter: not a real risk
  #    improvement.
  
  # Corrected view: exclude quarters too recent to have matured,
  # so the trend reflects true vintage risk rather than censoring.
  
  max_date <- max(loans_temporal$issue_date)
  cutoff_date <- max_date %m-% months(24)
  
  default_by_quarter_adj <- loans_temporal %>%
    filter(issue_date <= cutoff_date) %>%
    mutate(issue_quarter = floor_date(issue_date, "quarter")) %>%
    group_by(issue_quarter) %>%
    summarise(
      n = n(),
      default_rate = mean(default) * 100
    )
  
  # Sanity check: n should stay comparatively stable through the
  # last included quarter, with no artificial collapse
  
  default_by_quarter_adj %>%
    arrange(issue_quarter) %>%
    tail(10)
  
  # Further check: the earliest quarters (2007) have tiny sample
  # sizes (n = 1, 81, 169 loans): Lending Club's earliest history,
  # before volume ramped up. Default rates computed on such small
  # samples are noise, not signal, and would distort the start of
  # the trend line. Excluded via a minimum observation threshold,
  # consistent with the same data-reliability standard applied
  # throughout this project.
  
  default_by_quarter_adj %>%
    arrange(issue_quarter) %>%
    head(5)
  
  default_by_quarter_adj <- default_by_quarter_adj %>%
    filter(n >= 500)
  
  p_temporal_adj <- ggplot(default_by_quarter_adj, aes(x = issue_quarter, y = default_rate)) +
    geom_line(color = "#2C7FB8", linewidth = 1) +
    geom_point(color = "#2C7FB8") +
    labs(
      title = "Default rate over time (matured vintages only)",
      subtitle = "Loans issued 24+ months before dataset cutoff, quarters with n \u2265 500",
      x = NULL, y = "Default rate (%)"
    ) +
    theme_minimal()
  
  p_temporal_adj
  ggsave(here("outputs", "default_rate_over_time_adjusted.png"), p_temporal_adj,
         width = 8, height = 5, dpi = 150)
  
} else {
  cat("issue_d column not found: skipping temporal analysis.\n",
      "Check column name if a time trend is needed for the dashboard.\n")
}


## ------------------------------------------------------------
## SUMMARY
## ------------------------------------------------------------

cat(
  "Overall default rate:", round(mean(loans_features$default) * 100, 1), "%\n",
  "Default rate range across grades:",
  round(min(default_by_grade$default_rate), 1), "% (grade",
  default_by_grade$grade[which.min(default_by_grade$default_rate)], ") to",
  round(max(default_by_grade$default_rate), 1), "% (grade",
  default_by_grade$grade[which.max(default_by_grade$default_rate)], ")\n",
  "Most common loan purpose:",
  loans_features %>% count(purpose, sort = TRUE) %>% slice(1) %>% pull(purpose), "\n"
)

## ------------------------------------------------------------
## NEXT STEP: Phase 4 Predictive Modeling
## (Logistic Regression on WOE-transformed variables, Random Forest,
## ROC-AUC, KS statistic, Gini coefficient)
## ------------------------------------------------------------
