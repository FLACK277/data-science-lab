# Heart Disease ML Project – R

## Dataset
**UCI Cleveland Heart Disease Dataset**  
303 patients · 13 clinical features · Binary classification (Disease / No Disease)

## Problem Statement
Predict whether a patient has heart disease based on clinical measurements such as
age, cholesterol, resting blood pressure, max heart rate, and ECG results.

---

## Project Structure

```
heart_disease_ml/
├── analysis.R          ← Full ML pipeline (preprocessing, EDA, modelling, evaluation)
├── app.R               ← Interactive Shiny dashboard
├── README.md           ← This file
├── rf_model.rds        ← Trained Random Forest (auto-generated)
├── preproc_obj.rds     ← Scaler object (auto-generated)
├── heart_clean.rds     ← Cleaned dataset (auto-generated)
├── metrics_list.rds    ← Evaluation metrics (auto-generated)
├── rf_cm.rds           ← Confusion matrix (auto-generated)
├── lr_cm.rds           ← LR confusion matrix (auto-generated)
├── eda_plots.png       ← EDA grid (auto-generated)
├── correlation.png     ← Correlation heatmap (auto-generated)
├── feature_importance.png
└── roc_curves.png
```

---

## Quick Start

### Step 1 – Install R packages

```r
install.packages(c(
  "tidyverse", "caret", "randomForest", "corrplot",
  "scales", "gridExtra", "e1071", "pROC",
  "shiny", "shinydashboard", "plotly", "DT"
))
```

### Step 2 – Run the analysis pipeline

```r
source("analysis.R")   # takes ~60-90 seconds (cross-validation)
```

This will:
- Download the dataset from UCI (or use synthetic fallback)
- Preprocess and clean the data
- Run EDA and save visualisations
- Train Random Forest + Logistic Regression with 10-fold CV
- Print evaluation metrics
- Save all artefacts for the dashboard

### Step 3 – Launch the Shiny dashboard

```r
shiny::runApp("app.R")
```

Open the URL printed in the console (typically `http://127.0.0.1:XXXX`).

---

## Dashboard Tabs

| Tab | Contents |
|-----|----------|
| **Overview** | KPI metrics, target & age distributions |
| **EDA** | Interactive histograms, boxplots, scatter plots |
| **Model Results** | Confusion matrix, feature importance, metric comparison |
| **Live Predictor** | Enter patient values → get real-time risk prediction with gauge |
| **Dataset** | Searchable, downloadable data table |

---

## Model Performance (Random Forest)

| Metric    | Score  |
|-----------|--------|
| Accuracy  | ~0.868 |
| Precision | ~0.882 |
| Recall    | ~0.882 |
| F1 Score  | ~0.882 |
| AUC-ROC   | ~0.930 |

*(exact values depend on random seed and train/test split)*

---

## Key Insights

1. **Asymptomatic chest pain** is the strongest predictor of heart disease.
2. **Thalassemia type** (reversible defect) strongly correlates with disease.
3. **Max heart rate** is inversely correlated with disease — lower rate = higher risk.
4. **ST depression** (oldpeak) increases significantly with disease presence.
5. **Males** in this dataset have a higher disease prevalence than females.
6. Random Forest outperforms Logistic Regression by ~3-5% across all metrics.

---

## Requirements

- R ≥ 4.2.0
- Internet access (for dataset download; synthetic fallback available offline)

---

## Deploying This App

### Important: Vercel limitation

This project is an **R Shiny server app**. Vercel does not provide an R/Shiny runtime,
so direct deployment to Vercel will fail.

### Recommended deployment (Render / Railway / Fly.io)

This repo now includes a `Dockerfile` so you can deploy as a container.

1. Push your repo to GitHub
2. On Render/Railway/Fly.io, create a new service from this repo
3. Select **Docker** deployment
4. Expose port `3838`
5. Deploy

The app starts with:

```bash
R -q -e "shiny::runApp('/app', host='0.0.0.0', port=3838)"
```

### Local Docker test

```bash
docker build -t heart-disease-shiny .
docker run --rm -p 3838:3838 heart-disease-shiny
```

Then open `http://127.0.0.1:3838`.

### Alternative: Deploy to shinyapps.io

This platform is purpose-built for R Shiny and is the fastest managed option.

1. Create an account on shinyapps.io
2. In shinyapps.io, open Account -> Tokens and create a new token
3. Copy `.Renviron.example` to `.Renviron` and fill in your values
4. Load the variables in your shell and run the deploy script

```bash
set -a
source .Renviron
set +a
Rscript deploy_shinyapps.R
```

The deploy script uses:

- `SHINYAPPS_NAME`
- `SHINYAPPS_TOKEN`
- `SHINYAPPS_SECRET`

and deploys this repo as app name `heart-disease-ml-shiny`.
