# =============================================================================
# HEART DISEASE ML – INTERACTIVE SHINY DASHBOARD
# Run: shiny::runApp("app.R")
# =============================================================================

library(shiny)
library(shinydashboard)
library(tidyverse)
library(caret)
library(randomForest)
library(ggplot2)
library(scales)
library(DT)
library(plotly)

# ── Load pre-computed artefacts ───────────────────────────────────────────────
base_dir    <- dirname(rstudioapi::getSourceEditorContext()$path)
rf_model    <- readRDS(file.path(base_dir, "rf_model.rds"))
preproc_obj <- readRDS(file.path(base_dir, "preproc_obj.rds"))
heart       <- readRDS(file.path(base_dir, "heart_clean.rds"))
metrics     <- readRDS(file.path(base_dir, "metrics_list.rds"))

# ── Helper: metric value box ──────────────────────────────────────────────────
metric_box <- function(value, label, icon_name, color) {
  valueBox(
    value  = sprintf("%.3f", value),
    subtitle = label,
    icon   = icon(icon_name),
    color  = color
  )
}

theme_dash <- theme_minimal(base_size = 13) +
  theme(
    plot.title      = element_text(face = "bold", colour = "#c0392b"),
    panel.grid.minor= element_blank(),
    legend.position = "bottom"
  )

# =============================================================================
# UI
# =============================================================================
ui <- dashboardPage(
  skin = "red",

  # ── Header ─────────────────────────────────────────────────────────────────
  dashboardHeader(
    title = tags$span(icon("heartbeat"), " Heart Disease ML"),
    titleWidth = 280
  ),

  # ── Sidebar ────────────────────────────────────────────────────────────────
  dashboardSidebar(
    width = 240,
    sidebarMenu(
      menuItem("Overview",        tabName="overview",    icon=icon("dashboard")),
      menuItem("EDA",             tabName="eda",         icon=icon("chart-bar")),
      menuItem("Model Results",   tabName="model",       icon=icon("robot")),
      menuItem("Live Predictor",  tabName="predictor",   icon=icon("stethoscope")),
      menuItem("Dataset",         tabName="data",        icon=icon("table"))
    )
  ),

  # ── Body ───────────────────────────────────────────────────────────────────
  dashboardBody(
    tags$head(tags$style(HTML("
      .skin-red .main-header .navbar { background-color:#c0392b; }
      .skin-red .main-header .logo   { background-color:#922b21; }
      .content-wrapper               { background-color:#f5f6fa; }
      .box { border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,.08); }
      .value-box .inner h3 { font-size:2rem; }
      .predict-card { background:#fff; border-radius:10px;
                      padding:20px; box-shadow:0 2px 12px rgba(0,0,0,.1); }
    "))),

    tabItems(

      # ── TAB 1: OVERVIEW ────────────────────────────────────────────────────
      tabItem(tabName="overview",
        fluidRow(
          box(width=12, status="danger", solidHeader=TRUE,
              title=tags$span(icon("info-circle"), " Project Summary"),
              p("This dashboard presents an end-to-end machine learning pipeline for",
                strong("Heart Disease Prediction"), "using the UCI Cleveland dataset."),
              tags$ul(
                tags$li("303 patient records with 13 clinical features"),
                tags$li("Binary classification: Disease vs No Disease"),
                tags$li("Algorithm: Random Forest (with Logistic Regression baseline)"),
                tags$li("10-fold cross-validation for hyperparameter tuning")
              )
          )
        ),
        fluidRow(
          metric_box(metrics$rf$accuracy,  "RF Accuracy",  "bullseye",       "red"),
          metric_box(metrics$rf$precision, "RF Precision", "filter",         "orange"),
          metric_box(metrics$rf$recall,    "RF Recall",    "search",         "yellow"),
          metric_box(metrics$rf$auc,       "RF AUC-ROC",   "chart-line",     "green")
        ),
        fluidRow(
          box(width=6, title="Target Distribution", status="danger",
              plotlyOutput("overviewTarget", height=280)),
          box(width=6, title="Age Distribution", status="warning",
              plotlyOutput("overviewAge",    height=280))
        )
      ),

      # ── TAB 2: EDA ─────────────────────────────────────────────────────────
      tabItem(tabName="eda",
        fluidRow(
          box(width=4, status="danger", title="EDA Controls",
              selectInput("eda_x", "X-Axis Variable",
                          choices = c("age","trestbps","chol","thalach","oldpeak"),
                          selected = "age"),
              selectInput("eda_plot_type", "Plot Type",
                          choices = c("Histogram","Boxplot","Scatter vs thalach"),
                          selected = "Histogram"),
              checkboxInput("eda_by_target", "Color by Heart Disease?", TRUE)
          ),
          box(width=8, status="danger", title="Exploratory Plot",
              plotlyOutput("edaPlot", height=380))
        ),
        fluidRow(
          box(width=6, title="Chest Pain Type vs Disease", status="warning",
              plotlyOutput("edaCp", height=300)),
          box(width=6, title="Thalassemia vs Disease", status="warning",
              plotlyOutput("edaThal", height=300))
        )
      ),

      # ── TAB 3: MODEL RESULTS ───────────────────────────────────────────────
      tabItem(tabName="model",
        fluidRow(
          box(width=6, title="Confusion Matrix – Random Forest", status="danger",
              plotlyOutput("cmPlot", height=350)),
          box(width=6, title="Feature Importance (Top 10)", status="warning",
              plotlyOutput("impPlot", height=350))
        ),
        fluidRow(
          box(width=12, title="Model Metrics Comparison", status="danger",
              plotlyOutput("metricsBar", height=300))
        )
      ),

      # ── TAB 4: LIVE PREDICTOR ──────────────────────────────────────────────
      tabItem(tabName="predictor",
        fluidRow(
          box(width=4, status="danger", solidHeader=TRUE,
              title=tags$span(icon("user-md"), " Patient Information"),
              sliderInput("p_age",      "Age (years)",        30, 80, 55, 1),
              selectInput("p_sex",      "Sex",                c("Male","Female")),
              selectInput("p_cp",       "Chest Pain Type",
                          c("Typical","Atypical","Non-anginal","Asymptomatic")),
              sliderInput("p_trestbps", "Resting BP (mm Hg)", 90, 200, 130, 1),
              sliderInput("p_chol",     "Cholesterol (mg/dl)", 130, 565, 245, 1),
              selectInput("p_fbs",      "Fasting BS > 120",   c("No"="≤120","Yes"="\\>120")),
              selectInput("p_restecg",  "Resting ECG",
                          c("Normal","ST-T abnorm","LV hypertrophy")),
              sliderInput("p_thalach",  "Max Heart Rate",     70, 210, 150, 1),
              selectInput("p_exang",    "Exercise Angina",    c("No","Yes")),
              sliderInput("p_oldpeak",  "ST Depression",      0, 6.2, 1.0, 0.1),
              selectInput("p_slope",    "Slope of ST",        c("Upsloping","Flat","Downsloping")),
              sliderInput("p_ca",       "# Major Vessels (0-3)", 0, 3, 0, 1),
              selectInput("p_thal",     "Thalassemia",        c("Normal","Fixed","Reversible")),
              br(),
              actionButton("predict_btn", "Predict", icon=icon("heartbeat"),
                           class="btn-danger btn-lg", width="100%")
          ),
          box(width=8, status="danger", solidHeader=TRUE,
              title=tags$span(icon("chart-pie"), " Prediction Result"),
              br(),
              uiOutput("predResult"),
              br(),
              plotlyOutput("predGauge", height=320),
              br(),
              verbatimTextOutput("predDetails")
          )
        )
      ),

      # ── TAB 5: DATASET ─────────────────────────────────────────────────────
      tabItem(tabName="data",
        fluidRow(
          box(width=12, title="Heart Disease Dataset (processed)", status="danger",
              DTOutput("dataTable"))
        )
      )
    )
  )
)

# =============================================================================
# SERVER
# =============================================================================
server <- function(input, output, session) {

  # ── OVERVIEW ───────────────────────────────────────────────────────────────
  output$overviewTarget <- renderPlotly({
    df <- heart %>% count(target)
    plot_ly(df, labels=~target, values=~n, type="pie",
            marker=list(colors=c("#2ecc71","#e74c3c")),
            textinfo="label+percent") %>%
      layout(showlegend=TRUE,
             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
  })

  output$overviewAge <- renderPlotly({
    plot_ly(heart, x=~age, color=~target, type="histogram", alpha=0.75,
            colors=c("#2ecc71","#e74c3c")) %>%
      layout(barmode="overlay",
             xaxis=list(title="Age"),
             yaxis=list(title="Count"),
             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
  })

  # ── EDA ────────────────────────────────────────────────────────────────────
  output$edaPlot <- renderPlotly({
    xv  <- input$eda_x
    col <- if (input$eda_by_target) heart$target else NULL

    if (input$eda_plot_type == "Histogram") {
      plot_ly(heart, x=~get(xv), color=col, type="histogram", alpha=0.75,
              colors=c("#2ecc71","#e74c3c")) %>%
        layout(barmode="overlay",
               xaxis=list(title=xv), yaxis=list(title="Count"),
               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

    } else if (input$eda_plot_type == "Boxplot") {
      plot_ly(heart, y=~get(xv), x=~target, color=~target, type="box",
              colors=c("#2ecc71","#e74c3c")) %>%
        layout(xaxis=list(title="Heart Disease"), yaxis=list(title=xv),
               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

    } else {
      plot_ly(heart, x=~get(xv), y=~thalach, color=~target,
              type="scatter", mode="markers", alpha=0.7,
              colors=c("#2ecc71","#e74c3c")) %>%
        layout(xaxis=list(title=xv), yaxis=list(title="Max Heart Rate"),
               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    }
  })

  output$edaCp <- renderPlotly({
    df <- heart %>% count(cp, target) %>%
      group_by(cp) %>% mutate(pct=n/sum(n))
    plot_ly(df, x=~cp, y=~pct, color=~target, type="bar",
            colors=c("#2ecc71","#e74c3c")) %>%
      layout(barmode="stack", yaxis=list(title="Proportion", tickformat=".0%"),
             xaxis=list(title="Chest Pain Type"),
             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
  })

  output$edaThal <- renderPlotly({
    df <- heart %>% count(thal, target) %>%
      group_by(thal) %>% mutate(pct=n/sum(n))
    plot_ly(df, x=~thal, y=~pct, color=~target, type="bar",
            colors=c("#2ecc71","#e74c3c")) %>%
      layout(barmode="stack", yaxis=list(title="Proportion", tickformat=".0%"),
             xaxis=list(title="Thalassemia"),
             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
  })

  # ── MODEL RESULTS ──────────────────────────────────────────────────────────
  output$cmPlot <- renderPlotly({
    cm_tbl <- readRDS(file.path(base_dir, "rf_cm.rds"))$table
    df <- as.data.frame(cm_tbl)
    names(df) <- c("Predicted","Actual","Freq")
    plot_ly(df, x=~Actual, y=~Predicted, z=~Freq, type="heatmap",
            colorscale=list(c(0,"#fff5f5"), c(1,"#c0392b")),
            text=~Freq, texttemplate="%{text}") %>%
      layout(xaxis=list(title="Actual"),
             yaxis=list(title="Predicted"),
             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
  })

  output$impPlot <- renderPlotly({
    imp <- varImp(rf_model)$importance %>%
      rownames_to_column("Feature") %>%
      arrange(desc(Overall)) %>%
      head(10)
    plot_ly(imp, x=~Overall, y=~reorder(Feature,Overall), type="bar",
            orientation="h",
            marker=list(color=~Overall,
                        colorscale=list(c(0,"#f39c12"),c(1,"#c0392b")))) %>%
      layout(xaxis=list(title="Importance"),
             yaxis=list(title=NULL),
             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
  })

  output$metricsBar <- renderPlotly({
    df <- data.frame(
      Metric = c("Accuracy","Precision","Recall","F1","AUC"),
      RF     = c(metrics$rf$accuracy, metrics$rf$precision,
                 metrics$rf$recall,   metrics$rf$f1, metrics$rf$auc),
      LR     = c(metrics$lr$accuracy, metrics$lr$precision,
                 metrics$lr$recall,   metrics$lr$f1, metrics$lr$auc)
    ) %>% pivot_longer(c(RF,LR), names_to="Model", values_to="Value")

    plot_ly(df, x=~Metric, y=~Value, color=~Model, type="bar",
            colors=c("#c0392b","#2980b9")) %>%
      layout(barmode="group", yaxis=list(title="Score", range=c(0,1)),
             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
  })

  # ── LIVE PREDICTOR ─────────────────────────────────────────────────────────
  pred_result <- eventReactive(input$predict_btn, {
    # Build a new patient row matching the original heart data structure
    new_patient <- data.frame(
      age      = input$p_age,
      sex      = factor(input$p_sex,    levels=c("Female","Male")),
      cp       = factor(input$p_cp,     levels=c("Typical","Atypical","Non-anginal","Asymptomatic")),
      trestbps = input$p_trestbps,
      chol     = input$p_chol,
      fbs      = factor(input$p_fbs,    levels=c("≤120","\\>120")),
      restecg  = factor(input$p_restecg,levels=c("Normal","ST-T abnorm","LV hypertrophy")),
      thalach  = input$p_thalach,
      exang    = factor(input$p_exang,  levels=c("No","Yes")),
      oldpeak  = input$p_oldpeak,
      slope    = factor(input$p_slope,  levels=c("Upsloping","Flat","Downsloping")),
      ca       = input$p_ca,
      thal     = factor(input$p_thal,   levels=c("Normal","Fixed","Reversible")),
      stringsAsFactors = FALSE
    )

    # Scale numeric columns using preproc_obj
    num_cols  <- c("age","trestbps","chol","thalach","oldpeak")
    new_scaled <- new_patient
    new_scaled[, num_cols] <- predict(preproc_obj, new_patient[, num_cols, drop=FALSE])

    # Predict
    pred_class <- predict(rf_model, new_scaled)
    pred_prob  <- predict(rf_model, new_scaled, type="prob")

    list(
      class    = as.character(pred_class),
      prob_dis = pred_prob[1, "Disease"],
      prob_no  = pred_prob[1, "No Disease"]
    )
  })

  output$predResult <- renderUI({
    req(pred_result())
    res <- pred_result()
    if (res$class == "Disease") {
      div(class="predict-card",
          style="border-left:6px solid #e74c3c; background:#fff5f5;",
          h2(icon("exclamation-triangle", style="color:#e74c3c"),
             " High Risk: Heart Disease Detected",
             style="color:#c0392b;"),
          p(sprintf("Probability of heart disease: %.1f%%", res$prob_dis * 100),
            style="font-size:1.2rem; color:#555;"),
          p(em("Please consult a cardiologist immediately."),
            style="color:#888;")
      )
    } else {
      div(class="predict-card",
          style="border-left:6px solid #2ecc71; background:#f0fff4;",
          h2(icon("check-circle", style="color:#2ecc71"),
             " Low Risk: No Heart Disease",
             style="color:#27ae60;"),
          p(sprintf("Probability of heart disease: %.1f%%", res$prob_dis * 100),
            style="font-size:1.2rem; color:#555;"),
          p(em("Maintain a healthy lifestyle. Regular check-ups recommended."),
            style="color:#888;")
      )
    }
  })

  output$predGauge <- renderPlotly({
    req(pred_result())
    prob <- pred_result()$prob_dis * 100

    plot_ly(
      type  = "indicator",
      mode  = "gauge+number+delta",
      value = round(prob, 1),
      number= list(suffix="%", font=list(size=40)),
      delta = list(reference=50, increasing=list(color="#e74c3c"),
                   decreasing=list(color="#2ecc71")),
      gauge = list(
        axis  = list(range=list(0,100), ticksuffix="%"),
        bar   = list(color=if(prob>=50) "#e74c3c" else "#2ecc71"),
        steps = list(
          list(range=c(0,30),  color="#d5f5e3"),
          list(range=c(30,60), color="#fef9e7"),
          list(range=c(60,100),color="#fadbd8")
        ),
        threshold=list(line=list(color="black",width=3), value=50)
      ),
      title=list(text="Disease Probability", font=list(size=18))
    ) %>%
      layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
             margin=list(t=80))
  })

  output$predDetails <- renderPrint({
    req(pred_result())
    res <- pred_result()
    cat("═══════════════════════════════════════\n")
    cat(" PREDICTION DETAILS\n")
    cat("═══════════════════════════════════════\n")
    cat(sprintf(" Predicted Class : %s\n",   res$class))
    cat(sprintf(" P(Disease)      : %.4f\n", res$prob_dis))
    cat(sprintf(" P(No Disease)   : %.4f\n", res$prob_no))
    cat(" Model           : Random Forest (500 trees)\n")
    cat(sprintf(" RF CV AUC       : %.4f\n", metrics$rf$auc))
    cat("═══════════════════════════════════════\n")
  })

  # ── DATASET TABLE ──────────────────────────────────────────────────────────
  output$dataTable <- renderDT({
    datatable(heart,
              options  = list(pageLength=15, scrollX=TRUE,
                              dom="Bfrtip", buttons=c("csv","excel")),
              extensions="Buttons",
              rownames  = FALSE,
              class     = "stripe hover compact") %>%
      formatStyle("target",
                  backgroundColor = styleEqual(
                    c("No Disease","Disease"), c("#d5f5e3","#fadbd8")))
  })
}

# =============================================================================
# RUN
# =============================================================================
shinyApp(ui=ui, server=server)
