## ============================================================
## PROJECT: Customer Risk & Value Segmentation
## 4 Predictive Modeling
## Input: data/processed/loans_features.rds (from 02 feature engineering.R)
## ============================================================

library(tidyverse)
library(here)
library(scorecard)

packages <- c("pROC", "ranger")
new_packages <- packages[!(packages %in% installed.packages()[, "Package"])]
if (length(new_packages) > 0) install.packages(new_packages)

library(pROC)
library(ranger)

loans_features <- readRDS(here("data", "processed", "loans_features.rds"))

set.seed(42)  # reproducibility for the split, RF, etc.


## ------------------------------------------------------------
## TRAIN/TEST SPLIT
## ------------------------------------------------------------

# Final predictor set from Phase 2 (raw scale, before WOE)
model_vars <- loans_features %>%
  select(
    default,
    loan_amnt, term, sub_grade, annual_inc, verification_status,
    dti, revol_util, loan_to_income, home_ownership
  )

# 70/30 split, stratified by default to preserve the ~20% default rate in both sets 
# (important for reliable metric estimation on an imbalanced target)

train_idx <- scorecard::split_df(model_vars, y = "default", ratio = 0.7, seed = 42)
train_raw <- train_idx$train
test_raw <- train_idx$test

cat(
  "Train rows:", nrow(train_raw), " | Train default rate:",
  round(mean(train_raw$default) * 100, 1), "%\n",
  "Test rows:", nrow(test_raw), " | Test default rate:",
  round(mean(test_raw$default) * 100, 1), "%\n"
)


## ------------------------------------------------------------
## WOE BINNING REFIT ON TRAINING DATA ONLY
## ------------------------------------------------------------

# IMPORTANT: in Phase 2, WOE bins were computed on the entire cleaned dataset, since no train/test split existed yet at that stage. 
# Now that a split exists, reusing those bins would leak information from the test set into the WOE values seen during training 
# (the bin boundaries and each bin's WOE would have been influenced by test-set observations). 
# To avoid this, WOE bins are refit here using the TRAINING set only, and then applied identically (same breakpoints) to both train and test.

bins_train <- woebin(train_raw, y = "default")

train_woe <- woebin_ply(train_raw, bins_train)
test_woe <- woebin_ply(test_raw, bins_train)  # same bins, applied to unseen data

# Quick check: IV on training data only (for reference: expected to be close to, but not identical to, the full-dataset IV computed in Phase 2)

iv_train <- map_dfr(bins_train, function(b) {
  tibble(info_value = unique(b$total_iv))
}, .id = "variable") %>%
  arrange(desc(info_value))

iv_train


## ------------------------------------------------------------
## LOGISTIC REGRESSION (on WOE-transformed variables)
## ------------------------------------------------------------

# Standard practice in banking credit scorecards: fitting a logistic regression on WOE-transformed variables keeps the model both interpretable 
# (monotonic relationship between each predictor and risk) and stable (WOE compresses outliers/extreme values into a bounded scale),
# which matters for regulatory explainability.

logit_model <- glm(default ~ ., data = train_woe, family = "binomial")

summary(logit_model)

# Predicted probability of default on the test set
test_woe$pred_logit <- predict(logit_model, newdata = test_woe, type = "response")


## ------------------------------------------------------------
## RANDOM FOREST (on raw, non-WOE variables)
## ------------------------------------------------------------

# Random Forest doesn't need WOE transformation, it captures non-linear relationships and interactions natively. 
# Categorical variables are converted to factors. 
# "ranger" is used instead of the base "randomForest" package for speed on a dataset this size.

train_rf <- train_raw %>%
  mutate(across(where(is.character), as.factor),
         default = as.factor(default))

test_rf <- test_raw %>%
  mutate(across(where(is.character), as.factor),
         default = as.factor(default))

# NOTE: training on ~940k rows (70% of 1.34M). This may take a few minutes depending on your machine. 
# num.trees kept moderate (300) as a reasonable trade-off between performance and runtime for a first model,
# increase later if needed once the pipeline works end-to-end.

rf_model <- ranger(
  default ~ .,
  data = train_rf,
  num.trees = 300,
  probability = TRUE,      # get predicted probabilities, not just class
  importance = "impurity", # for a variable importance plot later
  seed = 42
)

rf_pred <- predict(rf_model, data = test_rf)$predictions[, "1"]
test_woe$pred_rf <- rf_pred


## ------------------------------------------------------------
## EVALUATION METRICS
## ------------------------------------------------------------

# ROC-AUC: overall discriminative power of each model

roc_logit <- roc(test_woe$default, test_woe$pred_logit, quiet = TRUE)
roc_rf <- roc(test_woe$default, test_woe$pred_rf, quiet = TRUE)

auc_logit <- auc(roc_logit)
auc_rf <- auc(roc_rf)

# Gini coefficient: standard credit risk metric, derived directly from AUC (2*AUC - 1). 
# Often reported alongside/instead of AUC in banking risk reports.

gini_logit <- 2 * auc_logit - 1
gini_rf <- 2 * auc_rf - 1

# KS (Kolmogorov-Smirnov) statistic: maximum separation between the cumulative distributions of good and bad payers' predicted scores,
# a classic credit scoring metric, often more intuitive to risk teams than AUC alone.

ks_stat <- function(actual, predicted) {
  roc_obj <- roc(actual, predicted, quiet = TRUE)
  max(abs(roc_obj$sensitivities - (1 - roc_obj$specificities)))
}

ks_logit <- ks_stat(test_woe$default, test_woe$pred_logit)
ks_rf <- ks_stat(test_woe$default, test_woe$pred_rf)

# Benchmark: how well does Lending Club's own sub_grade alone discriminate risk, as a point of comparison for both models
# (higher sub_grade letter/number = worse -> convert to a numeric risk score for AUC calculation)

subgrade_rank <- test_raw %>%
  mutate(subgrade_score = as.numeric(factor(sub_grade, levels = sort(unique(sub_grade)))))

roc_subgrade <- roc(subgrade_rank$default, subgrade_rank$subgrade_score, quiet = TRUE)
auc_subgrade <- auc(roc_subgrade)

# Summary table

model_comparison <- tibble(
  model = c("Logistic Regression (WOE)", "Random Forest", "sub_grade only (LC benchmark)"),
  auc = c(auc_logit, auc_rf, auc_subgrade),
  gini = c(gini_logit, gini_rf, 2 * auc_subgrade - 1),
  ks = c(ks_logit, ks_rf, NA)
)

model_comparison

# ROC curve comparison plot

png(here("outputs", "roc_curve_comparison.png"), width = 700, height = 700)
plot(roc_logit, col = "#2C7FB8", main = "ROC Curve — Model Comparison")
plot(roc_rf, col = "#E34A33", add = TRUE)
legend("bottomright",
       legend = c(
         paste0("Logistic Regression (AUC=", round(auc_logit, 3), ")"),
         paste0("Random Forest (AUC=", round(auc_rf, 3), ")")
       ),
       col = c("#2C7FB8", "#E34A33"), lwd = 2)
dev.off()


## ------------------------------------------------------------
## RANDOM FOREST VARIABLE IMPORTANCE
## ------------------------------------------------------------

rf_importance <- tibble(
  variable = names(rf_model$variable.importance),
  importance = rf_model$variable.importance
) %>%
  arrange(desc(importance))

rf_importance

p_importance <- ggplot(rf_importance, aes(x = fct_reorder(variable, importance), y = importance)) +
  geom_col(fill = "#41AB5D") +
  coord_flip() +
  labs(title = "Random Forest — Variable Importance", x = NULL, y = "Importance (impurity)") +
  theme_minimal()

p_importance
ggsave(here("outputs", "rf_variable_importance.png"), p_importance,
       width = 7, height = 5, dpi = 150)


## ------------------------------------------------------------
## CALIBRATION CHECK (Logistic Regression)
## ------------------------------------------------------------

# Checks whether predicted probabilities are actually reliable (e.g. among loans predicted at ~20% PD, is the observed default rate really close to 20%?)
# matters more than ranking ability alone once we move to Phase 5 
# (Expected Loss calculations depend on the predicted PD being well-calibrated, not just well-ranked).

calibration_data <- test_woe %>%
  mutate(pd_bucket = ntile(pred_logit, 10)) %>%
  group_by(pd_bucket) %>%
  summarise(
    predicted_pd = mean(pred_logit),
    observed_default_rate = mean(as.numeric(as.character(default)))
  )

calibration_data

p_calibration <- ggplot(calibration_data, aes(x = predicted_pd, y = observed_default_rate)) +
  geom_point(size = 3, color = "#2C7FB8") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50") +
  labs(
    title = "Calibration check - Logistic Regression",
    subtitle = "Points near the diagonal indicate well-calibrated probabilities",
    x = "Predicted PD (decile average)", y = "Observed default rate"
  ) +
  theme_minimal()

p_calibration
ggsave(here("outputs", "calibration_check.png"), p_calibration,
       width = 7, height = 5, dpi = 150)


## ------------------------------------------------------------
## PHASE 4.8 — SAVE OUTPUTS
## ------------------------------------------------------------

saveRDS(logit_model, here("data", "processed", "logit_model.rds"))
saveRDS(rf_model, here("data", "processed", "rf_model.rds"))
saveRDS(bins_train, here("data", "processed", "woe_bins_train.rds"))
saveRDS(test_woe, here("data", "processed", "test_woe_with_preds.rds"))

write_csv(model_comparison, here("outputs", "model_comparison.csv"))

cat(
  "Logistic Regression — AUC:", round(auc_logit, 3),
  "| Gini:", round(gini_logit, 3), "| KS:", round(ks_logit, 3), "\n",
  "Random Forest       — AUC:", round(auc_rf, 3),
  "| Gini:", round(gini_rf, 3), "| KS:", round(ks_rf, 3), "\n",
  "sub_grade benchmark — AUC:", round(auc_subgrade, 3), "\n"
)

## ------------------------------------------------------------
## NEXT STEP: Phase 5 From model to economic decision
## (Expected Loss = PD x LGD x EAD, cost matrix, profit curve,
## threshold optimization)
## ------------------------------------------------------------
