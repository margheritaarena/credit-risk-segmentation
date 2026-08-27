## ============================================================
## PROJECT: Customer Risk & Value Segmentation
## Dataset: Lending Club Loan Data (Kaggle)
## ============================================================

## ------------------------------------------------------------
## SETUP
## ------------------------------------------------------------

#renv::init()

# Required packages for this phase
#packages <- c("tidyverse", "here", "janitor", "naniar", "skimr")

#new_packages <- packages[!(packages %in% installed.packages()[, "Package"])]
#if (length(new_packages) > 0) install.packages(new_packages)

library(tidyverse)  # dplyr, ggplot2, readr, purrr, etc.
library(here)         # robust relative path handling
library(janitor)       # column name cleaning
library(naniar)          # missing values analysis
library(skimr)             # quick summary statistics

#dir.create(here("data", "raw"), recursive = TRUE, showWarnings = FALSE)
#dir.create(here("data", "processed"), recursive = TRUE, showWarnings = FALSE)
#dir.create(here("outputs"), showWarnings = FALSE)


## ------------------------------------------------------------
## IMPORT
## ------------------------------------------------------------

# Path
raw_path <- here("data", "raw", "accepted_2007_to_2018Q4.csv.gz")

# Import
loans_raw <- read_csv(
  raw_path,
  guess_max = 100000,
  show_col_types = FALSE
)

# Quick first look: dimensions, types, missingness at a glance
dim(loans_raw)
skim(loans_raw)

# Clean column names (consistent snake_case, no spaces/mixed case)
loans <- loans_raw %>%
  clean_names()


## TARGET VARIABLE DEFINITION

# The dataset contains loans in different states. 
# For a probability of default model we must keep ONLY final outcomes: "Fully Paid" and "Charged Off". 
# We exclude "Current", "In Grace Period", etc. because their outcome is not yet known,
# including them would introduce data leakage (the model would "see" loans whose future result is not yet determined).

table(loans$loan_status, useNA = "ifany")

loans_final <- loans %>%
  filter(loan_status %in% c("Fully Paid", "Charged Off")) %>%
  mutate(
    default = if_else(loan_status == "Charged Off", 1L, 0L)
  )

# Class balance check: if default is rare (typical in credit risk, often 15-20%) 
# accuracy alone will be a misleading metric later on (Phase 4).
loans_final %>%
  count(default) %>%
  mutate(pct = n / sum(n) * 100)


## MISSING VALUES ANALYSIS AND TREATMENT

# Visualize the missingness pattern 
loans_final %>%
  sample_n(5000) %>%
  vis_miss(warn_large_data = FALSE)

# Percentage of missing values per column, sorted (useful to decide
# which columns to drop entirely: typical threshold 40-50% missing)
miss_summary <- loans_final %>%
  miss_var_summary() %>%
  arrange(desc(pct_miss))

miss_summary %>% head(20)

# Drop columns with too many missing values (threshold: 40%)
columns_to_keep <- miss_summary %>%
  filter(pct_miss < 40) %>%
  pull(variable)

loans_reduced <- loans_final %>%
  select(all_of(columns_to_keep))

# Reasoned imputation for remaining columns with residual missing values:
# - numeric: median (robust to outliers, better than the mean for financial variables that are typically right-skewed, e.g. income)
# - categorical: explicit "Unknown" category (never drop whole rows without justification — you lose information)
loans_clean <- loans_reduced %>%
  mutate(across(where(is.numeric), ~ replace_na(.x, median(.x, na.rm = TRUE)))) %>%
  mutate(across(where(is.character), ~ replace_na(.x, "Unknown")))


## OUTLIER TREATMENT

outlier_diagnostics <- loans_clean %>%
  select(where(is.numeric), -default) %>%   # exclude the target variable
  map_dfr(function(x) {
    q <- quantile(x, probs = c(0.25, 0.75), na.rm = TRUE)
    iqr <- q[2] - q[1]
    lower <- q[1] - 1.5 * iqr
    upper <- q[2] + 1.5 * iqr
    tibble(
      pct_outliers = round(mean(x < lower | x > upper, na.rm = TRUE) * 100, 2),
      skewness = round(e1071::skewness(x, na.rm = TRUE), 2),
      min = min(x, na.rm = TRUE),
      max = max(x, na.rm = TRUE)
    )
  }, .id = "variable") %>%
  arrange(desc(pct_outliers))

# Identify variables with pct_outliers > 2%
high_outlier_vars <- outlier_diagnostics %>%
  filter(pct_outliers > 2) %>%
  pull(variable)

high_outlier_vars

# IQR method: values beyond 1.5*IQR from the quartiles are capped (winsorizing) rather than removed, to avoid losing rows.

loans_clean <- loans_clean %>%
  mutate(across(
    all_of(high_outlier_vars),
    function(x) {
      q <- quantile(x, probs = c(0.25, 0.75), na.rm = TRUE)
      iqr <- q[2] - q[1]
      lower <- q[1] - 1.5 * iqr
      upper <- q[2] + 1.5 * iqr
      pmin(pmax(x, lower), upper)
    }
  ))

# Should return 0 (or very close to 0) for all treated variables
loans_clean %>%
  select(all_of(high_outlier_vars)) %>%
  map_dbl(function(x) {
    q <- quantile(x, probs = c(0.25, 0.75), na.rm = TRUE)
    iqr <- q[2] - q[1]
    sum(x < (q[1] - 1.5*iqr) | x > (q[2] + 1.5*iqr), na.rm = TRUE)
  })


## SAVE CLEANED DATASET

# Save as .rds (preserves R types, more efficient than CSV for a dataset this size)
saveRDS(loans_clean, here("data", "processed", "loans_clean.rds"))

# Final summary to report in the README (concrete numbers, not just "I cleaned the data")
cat(
  "Original rows:", nrow(loans_raw), "\n",
  "Rows after filtering final outcomes:", nrow(loans_final), "\n",
  "Columns after dropping >40% missing:", ncol(loans_clean), "\n",
  "Final dataset rows:", nrow(loans_clean), "\n",
  "Default rate:", round(mean(loans_clean$default) * 100, 1), "%\n"
)

## ------------------------------------------------------------
## NEXT STEP: Phase 2 Feature Engineering
## (DTI, loan-to-income, WOE/IV with the scorecard package)
## ------------------------------------------------------------
