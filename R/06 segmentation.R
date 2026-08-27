## ============================================================
## PROJECT: Customer Risk & Value Segmentation
## PHASE 6 Customer Segmentation (K-means clustering)
## Input: data/processed/test_full_economic.rds (from 05 economic decision.R)
## ============================================================

library(tidyverse)
library(here)

packages <- c("factoextra")
new_packages <- packages[!(packages %in% installed.packages()[, "Package"])]
if (length(new_packages) > 0) install.packages(new_packages)

library(factoextra)

test_full <- readRDS(here("data", "processed", "test_full_economic.rds"))

set.seed(42)


## ------------------------------------------------------------
## VARIABLE SELECTION AND STANDARDIZATION
## ------------------------------------------------------------

# Clustering is run on the test set, since it already carries both the risk signal (pred_logit, the modeled PD) and the economic
# layer (expected_profit_if_approved) built in Phase 5, combining both a "risk" and a "value" dimension is exactly what a
# risk/value segmentation needs, and it keeps the clustering variables independent from the ones used to fit the risk model
# itself (avoiding circular reasoning: sub_grade/dti/etc. were already used to predict PD, so re-clustering on the same raw
# predictors would mostly just re-discover risk grades, not add a new "value" dimension).

# DEFENSIVE CHECK: earlier versions of 05 economic decision.R did not carry annual_inc into test_full_economic.rds (only added in
# a later fix). If you re-ran 05 end-to-end AFTER that fix, this block does nothing and simply proceeds. If test_full_economic.rds
# still predates the fix, annual_inc is reconstructed here the same way it was recovered in Phase 5: reproducing the train/test
# split with the identical seed on a superset of columns from loans_features, verifying row alignment before merging, and
# guarding against accidentally binding a duplicate column (the issue that produced annual_inc...25/26/27 during development).

if (!"annual_inc" %in% names(test_full)) {
  
  cat("annual_inc not found in test_full — reconstructing from",
      "loans_features via the reproduced Phase 5 split.\n")
  
  loans_features <- readRDS(here("data", "processed", "loans_features.rds"))
  
  model_vars_econ <- loans_features %>%
    select(
      default,
      loan_amnt, term, sub_grade, annual_inc, verification_status,
      dti, revol_util, loan_to_income, home_ownership,
      int_rate, funded_amnt, recoveries, total_pymnt
    )
  
  split_econ <- scorecard::split_df(model_vars_econ, y = "default", ratio = 0.7, seed = 42)
  test_econ <- split_econ$test
  
  # Verification: same check used in Phase 5, must match before
  # trusting any merge based on row order/position.
  stopifnot(
    nrow(test_econ) == nrow(test_full),
    identical(test_econ$default, test_full$default)
  )
  
  # Explicitly drop any pre-existing annual_inc-like column first,
  # so re-running this block never creates annual_inc...N duplicates.
  test_full <- test_full %>%
    select(-any_of("annual_inc")) %>%
    bind_cols(test_econ %>% select(annual_inc))
  
  cat("annual_inc successfully reconstructed and merged.\n")
  
} else {
  cat("annual_inc already present in test_full — no action needed.\n")
}

# Should show exactly one "annual_inc" column, no numbered suffixes

stopifnot(sum(names(test_full) == "annual_inc") == 1)

cluster_vars <- test_full %>%
  select(pred_logit, loan_amnt, annual_inc, expected_profit_if_approved) %>%
  rename(
    risk_pd = pred_logit,
    loan_amount = loan_amnt,
    income = annual_inc,
    expected_profit = expected_profit_if_approved
  )

summary(cluster_vars)

# K-means is distance-based, so all variables must be standardized first, otherwise a variable on a large scale (e.g. income, in
# tens of thousands) would dominate the distance calculation over a variable on a small scale (e.g. risk_pd, between 0 and 1).

cluster_vars_scaled <- scale(cluster_vars)


## ------------------------------------------------------------
## CHOOSING THE NUMBER OF CLUSTERS (elbow method)
## ------------------------------------------------------------

# NOTE: fviz_nbclust() runs k-means repeatedly (for k = 1 to 10 by default) on the full dataset — with ~400k rows this can take a
# few minutes. A random subsample is used here to speed up this diagnostic step only; the final clustering still
# runs on the full test set once k is chosen.

set.seed(42)
elbow_sample <- cluster_vars_scaled[sample(nrow(cluster_vars_scaled), 20000), ]

p_elbow <- fviz_nbclust(elbow_sample, kmeans, method = "wss", k.max = 10) +
  labs(title = "Elbow method — choosing the number of clusters",
       subtitle = "Computed on a 20,000-row subsample for speed")

p_elbow
ggsave(here("outputs", "elbow_method.png"), p_elbow, width = 7, height = 5, dpi = 150)


## ------------------------------------------------------------
## K-MEANS CLUSTERING
## ------------------------------------------------------------

# k is set based on the elbow plot above. 4 is used as a starting assumption consistent with the plan's four illustrative segments
# (e.g. low risk/high value, low risk/low value, high risk/high value, high risk/low value), and re-evaluated against the actual
# elbow shape once the plot is inspected.

k_final <- 4

kmeans_result <- kmeans(cluster_vars_scaled, centers = k_final, nstart = 25)

test_full$cluster <- as.factor(kmeans_result$cluster)

# Cluster sizes: check for any extremely small/unstable cluster

table(test_full$cluster)


## ------------------------------------------------------------
## CLUSTER PROFILING (business-friendly interpretation)
## ------------------------------------------------------------

# Average value of each clustering variable per cluster, on the ORIGINAL (unscaled) scale, scaled values are useful for the
# algorithm, not for reading the results.

cluster_profile <- test_full %>%
  group_by(cluster) %>%
  summarise(
    n = n(),
    pct_of_portfolio = n() / nrow(test_full) * 100,
    avg_pd = mean(pred_logit),
    avg_loan_amount = mean(loan_amnt),
    avg_income = mean(annual_inc),
    avg_expected_profit = mean(expected_profit_if_approved),
    actual_default_rate = mean(default)
  ) %>%
  arrange(desc(avg_expected_profit))

cluster_profile

# Business-friendly labels, assigned by reading the profile table above (median PD and median expected profit as thresholds)

# NOTE: run this after inspecting cluster_profile, and adjust the thresholds/labels manually if the actual cluster centers don't
# cleanly split into four quadrants.

# NOTE: an initial automatic quadrant labeling (median PD x median expected profit) produced two clusters both landing in
# "High Risk - High Value" and none in "High Risk - Low Value", not a bug, but a genuine pattern consistent with Phase 5's
# finding that Lending Club's risk-based interest pricing keeps expected value positive even in the riskiest segments observed.
# The two "high risk" clusters were then manually differentiated by scale (loan size / income), which is what actually separates
# them, rather than forcing a generic quadrant label that would hide that distinction.

cluster_profile <- cluster_profile %>%
  mutate(
    segment_label = case_when(
      cluster == 3 ~ "High Risk - Premium Value (large loans, high income)",
      cluster == 2 ~ "High Risk - Volume Value (smaller loans, mass market)",
      cluster == 1 ~ "Low Risk - High Value",
      cluster == 4 ~ "Low Risk - Low Value (mass market)",
      TRUE ~ NA_character_
    )
  )

cluster_profile %>% select(cluster, segment_label, n, pct_of_portfolio, avg_pd,
                           avg_loan_amount, avg_income, avg_expected_profit,
                           actual_default_rate)


## ------------------------------------------------------------
## VISUALIZATION
## ------------------------------------------------------------

# Risk vs. value scatter, colored by cluster: the core visual for the dashboard's Segmentation tab. 
# Sampled for plotting (400k points would be an unreadable, overplotted blob).

plot_sample <- test_full %>% slice_sample(n = 10000)

p_clusters <- ggplot(plot_sample, aes(x = pred_logit, y = expected_profit_if_approved, color = cluster)) +
  geom_point(alpha = 0.4, size = 1) +
  scale_y_continuous(labels = scales::comma) +
  labs(
    title = "Customer segments - risk vs. expected value",
    subtitle = "10,000-point sample from the test set",
    x = "Predicted probability of default (PD)",
    y = "Expected profit if approved ($)",
    color = "Cluster"
  ) +
  theme_minimal()

p_clusters
ggsave(here("outputs", "customer_segments.png"), p_clusters, width = 8, height = 6, dpi = 150)


## ------------------------------------------------------------
## SAVE OUTPUTS
## ------------------------------------------------------------

saveRDS(kmeans_result, here("data", "processed", "kmeans_result.rds"))
saveRDS(test_full, here("data", "processed", "test_full_segmented.rds"))
write_csv(cluster_profile, here("outputs", "cluster_profile.csv"))

cat(
  "Number of clusters:", k_final, "\n",
  "Cluster sizes:\n"
)
print(table(test_full$cluster))

## ------------------------------------------------------------
## NEXT STEP: Phase 7 Shiny Dashboard
## (Overview, Segmentation, Threshold Simulator tabs)
## ------------------------------------------------------------
