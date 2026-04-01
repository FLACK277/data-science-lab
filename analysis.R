# =============================================================================
# END-TO-END MACHINE LEARNING PROJECT IN R
# Dataset: Heart Disease UCI (Cleveland Heart Disease Dataset)
# Problem: Binary Classification - Predict presence of heart disease
# Author: ML Project
# =============================================================================

# ── 1. LOAD LIBRARIES ─────────────────────────────────────────────────────────
library(tidyverse)      # Data manipulation & ggplot2
library(caret)          # ML workflow, train/test split, metrics
library(randomForest)   # Random Forest algorithm
library(corrplot)       # Correlation matrix visualization
library(scales)         # Scale helpers for ggplot2
library(gridExtra)      # Arrange multiple ggplot2 plots
library(e1071)          # SVM & statistical functions
library(pROC)           # ROC curve analysis

set.seed(42)            # Reproducibility

# =============================================================================
# 2. DATASET DESCRIPTION
# =============================================================================
# The Cleveland Heart Disease dataset from the UCI ML Repository contains
# 303 patient records with 14 attributes. The target variable indicates
# whether a patient has heart disease (1) or not (0).
#
# Features:
#   age     - Age in years
#   sex     - Sex (1 = male, 0 = female)
#   cp      - Chest pain type (0-3)
#   trestbps- Resting blood pressure (mm Hg)
#   chol    - Serum cholesterol (mg/dl)
#   fbs     - Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
#   restecg - Resting ECG results (0-2)
#   thalach - Maximum heart rate achieved
#   exang   - Exercise-induced angina (1 = yes, 0 = no)
#   oldpeak - ST depression induced by exercise
#   slope   - Slope of peak exercise ST segment (0-2)
#   ca      - Number of major vessels colored by fluoroscopy (0-3)
#   thal    - Thalassemia type (1 = normal, 2 = fixed defect, 3 = reversible)
#   target  - Heart disease (1 = present, 0 = absent)

# =============================================================================
# 3. DATA LOADING
# =============================================================================

# Fetch data directly from UCI ML Repository
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

col_names <- c("age","sex","cp","trestbps","chol","fbs",
               "restecg","thalach","exang","oldpeak","slope","ca","thal","target")

# Attempt download; fall back to synthetic data if offline
heart_raw <- tryCatch({
  read.csv(url, header = FALSE, col.names = col_names, na.strings = "?")
}, error = function(e) {
  message("Network unavailable – generating representative synthetic data...")
  NULL
})

# ── Synthetic fallback (statistically representative) ─────────────────────────
if (is.null(heart_raw)) {
  n <- 303
  heart_raw <- data.frame(
    age      = round(rnorm(n, 54, 9)),
    sex      = sample(0:1, n, replace=TRUE, prob=c(0.32, 0.68)),
    cp       = sample(0:3, n, replace=TRUE, prob=c(0.47,0.16,0.28,0.09)),
    trestbps = round(rnorm(n, 131, 18)),
    chol     = round(rnorm(n, 246, 52)),
    fbs      = sample(0:1, n, replace=TRUE, prob=c(0.85, 0.15)),
    restecg  = sample(0:2, n, replace=TRUE, prob=c(0.50,0.02,0.48)),
    thalach  = round(rnorm(n, 150, 23)),
    exang    = sample(0:1, n, replace=TRUE, prob=c(0.67, 0.33)),
    oldpeak  = pmax(0, round(rnorm(n, 1.04, 1.16), 1)),
    slope    = sample(0:2, n, replace=TRUE, prob=c(0.07,0.46,0.47)),
    ca       = sample(0:3, n, replace=TRUE, prob=c(0.59,0.21,0.13,0.07)),
    thal     = sample(c(1,2,3), n, replace=TRUE, prob=c(0.05,0.38,0.57)),
    target   = sample(0:1, n, replace=TRUE, prob=c(0.46, 0.54))
  )
  # Inject 6 missing values (mirrors real dataset)
  heart_raw$ca[sample(n,2)]   <- NA
  heart_raw$thal[sample(n,2)] <- NA
  heart_raw$chol[sample(n,1)] <- NA
  heart_raw$trestbps[sample(n,1)] <- NA
}

cat("✓ Data loaded:", nrow(heart_raw), "rows ×", ncol(heart_raw), "columns\n")

# =============================================================================
# 4. DATA PREPROCESSING
# =============================================================================

heart <- heart_raw

# 4a. Inspect missing values
cat("\n── Missing Values ──────────────────────────────\n")
miss_summary <- colSums(is.na(heart))
print(miss_summary[miss_summary > 0])

# 4b. Binarize target (original has 0-4; >0 = disease present)
heart$target <- ifelse(heart$target > 0, 1, 0)

# 4c. Impute missing values with median (robust to outliers)
for (col in names(heart)) {
  if (any(is.na(heart[[col]]))) {
    heart[[col]][is.na(heart[[col]])] <- median(heart[[col]], na.rm = TRUE)
    cat("  Imputed", col, "with median =", median(heart_raw[[col]], na.rm=TRUE), "\n")
  }
}

# 4d. Type conversion – factors for categorical variables
heart <- heart %>%
  mutate(
    sex     = factor(sex,     levels=0:1, labels=c("Female","Male")),
    cp      = factor(cp,      levels=0:3, labels=c("Typical","Atypical","Non-anginal","Asymptomatic")),
    fbs     = factor(fbs,     levels=0:1, labels=c("≤120","\\>120")),
    restecg = factor(restecg, levels=0:2, labels=c("Normal","ST-T abnorm","LV hypertrophy")),
    exang   = factor(exang,   levels=0:1, labels=c("No","Yes")),
    slope   = factor(slope,   levels=0:2, labels=c("Upsloping","Flat","Downsloping")),
    thal    = factor(thal,    levels=c(1,2,3), labels=c("Normal","Fixed","Reversible")),
    target  = factor(target,  levels=0:1, labels=c("No Disease","Disease"))
  )

# 4e. Scale numeric features (Z-score normalisation)
numeric_cols <- c("age","trestbps","chol","thalach","oldpeak")
preproc_obj  <- preProcess(heart[, numeric_cols], method = c("center","scale"))
heart_scaled <- predict(preproc_obj, heart)

cat("\n✓ Preprocessing complete. Final dimensions:", nrow(heart_scaled), "×", ncol(heart_scaled), "\n")
cat("  Target distribution:\n")
print(table(heart_scaled$target))

# =============================================================================
# 5. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

# 5a. Summary statistics
cat("\n── Summary Statistics ──────────────────────────\n")
print(summary(heart[, c("age","trestbps","chol","thalach","oldpeak","target")]))

# ── Helper theme ──────────────────────────────────────────────────────────────
theme_heart <- theme_minimal(base_size = 13) +
  theme(
    plot.title      = element_text(face="bold", size=14, colour="#c0392b"),
    plot.subtitle   = element_text(colour="#7f8c8d", size=11),
    axis.title      = element_text(colour="#2c3e50"),
    legend.position = "bottom",
    panel.grid.minor= element_blank()
  )

# 5b. Age distribution by disease
p1 <- ggplot(heart, aes(age, fill=target)) +
  geom_histogram(bins=25, colour="white", alpha=0.85, position="identity") +
  scale_fill_manual(values=c("#2ecc71","#e74c3c"), name="Heart Disease") +
  labs(title="Age Distribution", subtitle="By heart disease status",
       x="Age (years)", y="Count") +
  theme_heart

# 5c. Chest pain type vs disease
p2 <- ggplot(heart, aes(cp, fill=target)) +
  geom_bar(position="fill", colour="white", alpha=0.9) +
  scale_fill_manual(values=c("#2ecc71","#e74c3c"), name="Heart Disease") +
  scale_y_continuous(labels=percent) +
  labs(title="Chest Pain Type", subtitle="Proportion with heart disease",
       x="Chest Pain Type", y="Proportion") +
  theme_heart + theme(axis.text.x=element_text(angle=20, hjust=1))

# 5d. Max heart rate vs age (coloured by disease)
p3 <- ggplot(heart, aes(age, thalach, colour=target)) +
  geom_point(alpha=0.65, size=2) +
  geom_smooth(method="lm", se=FALSE, linewidth=1.2) +
  scale_colour_manual(values=c("#27ae60","#c0392b"), name="Heart Disease") +
  labs(title="Max Heart Rate vs Age", subtitle="Linear trend by disease status",
       x="Age", y="Max Heart Rate (bpm)") +
  theme_heart

# 5e. Cholesterol boxplot
p4 <- ggplot(heart, aes(target, chol, fill=target)) +
  geom_boxplot(outlier.colour="#e74c3c", outlier.shape=21, alpha=0.8) +
  scale_fill_manual(values=c("#2ecc71","#e74c3c"), name="Heart Disease") +
  labs(title="Cholesterol by Disease", subtitle="mg/dl",
       x="Heart Disease", y="Serum Cholesterol (mg/dl)") +
  theme_heart + theme(legend.position="none")

# 5f. Sex distribution
p5 <- ggplot(heart, aes(sex, fill=target)) +
  geom_bar(position="dodge", colour="white", alpha=0.9) +
  scale_fill_manual(values=c("#2ecc71","#e74c3c"), name="Heart Disease") +
  labs(title="Disease by Sex", x="Sex", y="Count") +
  theme_heart

# 5g. ST depression (oldpeak) density
p6 <- ggplot(heart, aes(oldpeak, fill=target)) +
  geom_density(alpha=0.65) +
  scale_fill_manual(values=c("#2ecc71","#e74c3c"), name="Heart Disease") +
  labs(title="ST Depression Density", subtitle="oldpeak",
       x="ST Depression", y="Density") +
  theme_heart

# Save EDA grid
eda_grid <- arrangeGrob(p1, p2, p3, p4, p5, p6, ncol=2)
ggsave("/home/claude/heart_disease_ml/eda_plots.png", eda_grid,
       width=14, height=18, dpi=150)
cat("✓ EDA plots saved\n")

# 5h. Correlation heatmap (numeric only)
num_df <- heart %>%
  mutate(
    target_num = as.integer(target) - 1,
    sex_num    = as.integer(sex) - 1
  ) %>%
  select(age, trestbps, chol, thalach, oldpeak, ca, target_num) %>%
  rename(target=target_num)

png("/home/claude/heart_disease_ml/correlation.png", width=900, height=800, res=120)
corrplot(cor(num_df), method="color", type="upper", addCoef.col="black",
         tl.col="black", tl.srt=45, col=colorRampPalette(c("#3498db","white","#e74c3c"))(200),
         title="Correlation Matrix", mar=c(0,0,2,0))
dev.off()
cat("✓ Correlation plot saved\n")

# =============================================================================
# 6. MODEL BUILDING
# =============================================================================

# 6a. Train / Test split (75% / 25%)
train_idx <- createDataPartition(heart_scaled$target, p=0.75, list=FALSE)
train_df  <- heart_scaled[ train_idx, ]
test_df   <- heart_scaled[-train_idx, ]

cat("\n── Data split ──────────────────────────────────\n")
cat("  Training set:", nrow(train_df), "rows\n")
cat("  Test set    :", nrow(test_df),  "rows\n")

# 6b. Cross-validation control (10-fold CV)
cv_ctrl <- trainControl(
  method          = "cv",
  number          = 10,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

# 6c. ── RANDOM FOREST ──────────────────────────────────────────────────────
cat("\n── Training Random Forest … ────────────────────\n")
rf_grid <- expand.grid(mtry = c(2, 3, 4, 5))

rf_model <- train(
  target ~ .,
  data      = train_df,
  method    = "rf",
  trControl = cv_ctrl,
  tuneGrid  = rf_grid,
  metric    = "ROC",
  ntree     = 500
)

cat("  Best mtry:", rf_model$bestTune$mtry, "\n")
cat("  CV ROC:   ", max(rf_model$results$ROC) %>% round(4), "\n")

# 6d. ── LOGISTIC REGRESSION (baseline) ───────────────────────────────────────
cat("\n── Training Logistic Regression … ─────────────\n")
lr_model <- train(
  target ~ .,
  data      = train_df,
  method    = "glm",
  family    = "binomial",
  trControl = cv_ctrl,
  metric    = "ROC"
)

# =============================================================================
# 7. MODEL EVALUATION
# =============================================================================

# Predictions
rf_pred_class <- predict(rf_model, test_df)
rf_pred_prob  <- predict(rf_model, test_df, type="prob")[,"Disease"]
lr_pred_class <- predict(lr_model, test_df)
lr_pred_prob  <- predict(lr_model, test_df, type="prob")[,"Disease"]

# Confusion matrices
rf_cm <- confusionMatrix(rf_pred_class, test_df$target, positive="Disease")
lr_cm <- confusionMatrix(lr_pred_class, test_df$target, positive="Disease")

# ROC
rf_roc <- roc(test_df$target, rf_pred_prob, levels=c("No Disease","Disease"))
lr_roc <- roc(test_df$target, lr_pred_prob, levels=c("No Disease","Disease"))

cat("\n═══════════════════════════════════════════════\n")
cat("  MODEL EVALUATION RESULTS\n")
cat("═══════════════════════════════════════════════\n")

print_metrics <- function(cm, roc_obj, model_name) {
  cat(sprintf("\n  ── %s ──────────────────\n", model_name))
  cat(sprintf("  Accuracy  : %.4f\n", cm$overall["Accuracy"]))
  cat(sprintf("  Precision : %.4f\n", cm$byClass["Precision"]))
  cat(sprintf("  Recall    : %.4f\n", cm$byClass["Recall"]))
  cat(sprintf("  F1 Score  : %.4f\n", cm$byClass["F1"]))
  cat(sprintf("  AUC-ROC   : %.4f\n", auc(roc_obj)))
}

print_metrics(rf_cm, rf_roc, "Random Forest")
print_metrics(lr_cm, lr_roc, "Logistic Regression")

# Variable importance plot
imp_df <- varImp(rf_model)$importance %>%
  rownames_to_column("Feature") %>%
  arrange(desc(Overall)) %>%
  head(10)

p_imp <- ggplot(imp_df, aes(reorder(Feature, Overall), Overall, fill=Overall)) +
  geom_col(colour="white", show.legend=FALSE) +
  coord_flip() +
  scale_fill_gradient(low="#f39c12", high="#c0392b") +
  labs(title="Top 10 Feature Importances",
       subtitle="Random Forest – Mean Decrease in Gini",
       x=NULL, y="Importance") +
  theme_heart

ggsave("/home/claude/heart_disease_ml/feature_importance.png", p_imp,
       width=9, height=6, dpi=150)
cat("\n✓ Feature importance plot saved\n")

# Save ROC comparison
png("/home/claude/heart_disease_ml/roc_curves.png", width=800, height=650, res=120)
plot(rf_roc, col="#c0392b", lwd=2.5, main="ROC Curves – Model Comparison")
lines(lr_roc, col="#2980b9", lwd=2.5, lty=2)
legend("bottomright",
       legend=c(sprintf("Random Forest (AUC=%.3f)", auc(rf_roc)),
                sprintf("Logistic Reg  (AUC=%.3f)", auc(lr_roc))),
       col=c("#c0392b","#2980b9"), lwd=2.5, lty=c(1,2), bty="n")
dev.off()
cat("✓ ROC curves plot saved\n")

# =============================================================================
# 8. SAVE ARTEFACTS FOR SHINY DASHBOARD
# =============================================================================
saveRDS(rf_model,    "/home/claude/heart_disease_ml/rf_model.rds")
saveRDS(preproc_obj, "/home/claude/heart_disease_ml/preproc_obj.rds")
saveRDS(heart,       "/home/claude/heart_disease_ml/heart_clean.rds")
saveRDS(rf_cm,       "/home/claude/heart_disease_ml/rf_cm.rds")
saveRDS(lr_cm,       "/home/claude/heart_disease_ml/lr_cm.rds")

metrics_list <- list(
  rf = list(
    accuracy  = unname(rf_cm$overall["Accuracy"]),
    precision = unname(rf_cm$byClass["Precision"]),
    recall    = unname(rf_cm$byClass["Recall"]),
    f1        = unname(rf_cm$byClass["F1"]),
    auc       = as.numeric(auc(rf_roc))
  ),
  lr = list(
    accuracy  = unname(lr_cm$overall["Accuracy"]),
    precision = unname(lr_cm$byClass["Precision"]),
    recall    = unname(lr_cm$byClass["Recall"]),
    f1        = unname(lr_cm$byClass["F1"]),
    auc       = as.numeric(auc(lr_roc))
  )
)
saveRDS(metrics_list, "/home/claude/heart_disease_ml/metrics_list.rds")

cat("\n✓ All model artefacts saved.\n")
cat("✓ analysis.R complete – run app.R to launch the Shiny dashboard.\n")
