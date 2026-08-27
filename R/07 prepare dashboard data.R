## ============================================================
## PROJECT: Customer Risk & Value Segmentation
## PHASE 7.0 Prepare lean data for the Shiny dashboard
## Input: outputs from Phases 4, 5, 6
## Output: app/data/ (self-contained data for deployment)
## ============================================================

library(tidyverse)
library(here)

test_full <- readRDS(here("data", "processed", "test_full_segmented.rds"))
cluster_profile <- read_csv(here("outputs", "cluster_profile.csv"), show_col_types = FALSE)
profit_curve <- read_csv(here("outputs", "profit_curve.csv"), show_col_types = FALSE)

dir.create(here("app", "data"), recursive = TRUE, showWarnings = FALSE)


## ------------------------------------------------------------
## LEAN DASHBOARD DATASET
## ------------------------------------------------------------

# Keep only the columns the app actually uses. Loading the full test_full_segmented.rds (25+ columns, including all WOE variables
# not needed for display) into a Shiny app is unnecessary overhead, both for local performance and for shinyapps.io's free-tier
# memory limit (1GB).

dashboard_data <- test_full %>%
  select(
    default, pred_logit, loan_amnt, annual_inc, dti_woe,
    expected_profit_if_approved, expected_loss, cluster
  ) %>%
  rename(pd = pred_logit, loan_amount = loan_amnt, income = annual_inc)

# A grade/sub_grade-equivalent isn't in this lean set (dropped after Phase 2 due to redundancy with the WOE-transformed
# sub_grade_woe, not human-readable) pull the original sub_grade back in from loans_features for display purposes only (not used
# in any model computation here, purely for the Overview tab's "default rate by grade" chart).

loans_features <- readRDS(here("data", "processed", "loans_features.rds"))
default_by_grade <- loans_features %>%
  group_by(grade) %>%
  summarise(default_rate = mean(default) * 100, n = n()) %>%
  arrange(grade)

saveRDS(dashboard_data, here("app", "data", "dashboard_data.rds"))
write_csv(cluster_profile, here("app", "data", "cluster_profile.csv"))
write_csv(profit_curve, here("app", "data", "profit_curve.csv"))
write_csv(default_by_grade, here("app", "data", "default_by_grade.csv"))

cat(
  "Dashboard data prepared:\n",
  "- dashboard_data.rds:", nrow(dashboard_data), "rows,", ncol(dashboard_data), "columns\n",
  "- cluster_profile.csv:", nrow(cluster_profile), "rows\n",
  "- profit_curve.csv:", nrow(profit_curve), "rows\n",
  "- default_by_grade.csv:", nrow(default_by_grade), "rows\n",
  "Saved to app/data/\n"
)
