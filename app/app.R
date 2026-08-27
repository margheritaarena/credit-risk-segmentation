## ============================================================
## PROJECT: Customer Risk & Value Segmentation
## PHASE 7 — Shiny Dashboard
## Data: app/data/ (prepared by 07_prepare_dashboard_data.R)
## ============================================================

library(shiny)
library(shinydashboard)
library(tidyverse)
library(plotly)
library(DT)
library(scales)

## ------------------------------------------------------------
## DATA (loaded once at app startup, not per-session)
## ------------------------------------------------------------

dashboard_data <- readRDS("data/dashboard_data.rds")
cluster_profile <- read_csv("data/cluster_profile.csv", show_col_types = FALSE)
profit_curve <- read_csv("data/profit_curve.csv", show_col_types = FALSE)
default_by_grade <- read_csv("data/default_by_grade.csv", show_col_types = FALSE)

# Pre-sample once for scatter plots — rendering 400k points in a
# browser is slow and visually just an overplotted blob; a fixed
# sample is representative and keeps the app responsive.
set.seed(42)
plot_sample <- dashboard_data %>% slice_sample(n = 8000)

cluster_labels <- cluster_profile %>% select(cluster, segment_label)


## ============================================================
## UI
## ============================================================

ui <- dashboardPage(

  dashboardHeader(title = "Credit Risk & Segmentation"),

  dashboardSidebar(
    sidebarMenu(
      menuItem("Overview", tabName = "overview", icon = icon("chart-line")),
      menuItem("Segmentation", tabName = "segmentation", icon = icon("users")),
      menuItem("Threshold Simulator", tabName = "simulator", icon = icon("sliders-h"))
    )
  ),

  dashboardBody(
    tags$head(tags$style(HTML("
      .content-wrapper { background-color: #f4f6f9; }
      .box { border-top-color: #2C7FB8; }
    "))),

    tabItems(

      ## -------------------- OVERVIEW TAB --------------------
      tabItem(
        tabName = "overview",

        fluidRow(
          valueBoxOutput("kpi_total_loans", width = 3),
          valueBoxOutput("kpi_default_rate", width = 3),
          valueBoxOutput("kpi_avg_pd", width = 3),
          valueBoxOutput("kpi_total_el", width = 3)
        ),

        fluidRow(
          box(
            title = "Default rate by Lending Club grade", width = 6, status = "primary",
            plotlyOutput("plot_default_by_grade", height = 320)
          ),
          box(
            title = "Predicted PD distribution", width = 6, status = "primary",
            plotlyOutput("plot_pd_distribution", height = 320)
          )
        ),

        fluidRow(
          box(
            title = "About this project", width = 12, status = "primary",
            p("This dashboard summarizes a credit risk and customer segmentation model built on the Lending Club Loan Data (2007-2018, ~1.34M loans after cleaning). A logistic regression on WOE-transformed variables predicts each loan's probability of default (PD), which feeds into an Expected Loss calculation and a customer segmentation. All figures shown here are computed on the held-out test set (~404k loans, never used to fit the model)."),
            p("See the project README on GitHub for full methodology, data quality checks, and modeling details.")
          )
        )
      ),

      ## -------------------- SEGMENTATION TAB --------------------
      tabItem(
        tabName = "segmentation",

        fluidRow(
          box(
            title = "Filter segments", width = 12, status = "primary",
            checkboxGroupInput(
              "selected_clusters", "Show segments:",
              choices = setNames(cluster_labels$cluster, cluster_labels$segment_label),
              selected = cluster_labels$cluster,
              inline = TRUE
            )
          )
        ),

        fluidRow(
          box(
            title = "Customer segments — risk vs. expected value", width = 8, status = "primary",
            plotlyOutput("plot_segments", height = 450)
          ),
          box(
            title = "Segment sizes", width = 4, status = "primary",
            plotlyOutput("plot_segment_sizes", height = 450)
          )
        ),

        fluidRow(
          box(
            title = "Segment profile", width = 12, status = "primary",
            DTOutput("table_cluster_profile")
          )
        )
      ),

      ## -------------------- THRESHOLD SIMULATOR TAB --------------------
      tabItem(
        tabName = "simulator",

        fluidRow(
          box(
            title = "Approval threshold", width = 12, status = "primary",
            p("Loans with a predicted probability of default (PD) at or below this threshold are approved; loans above it are rejected. Move the slider to see the effect on approval rate, expected profit, and expected loss, computed live on the test-set portfolio (~404k loans)."),
            sliderInput(
              "pd_threshold", NULL,
              min = 0.03, max = 0.60, value = 0.20, step = 0.01,
              width = "100%"
            )
          )
        ),

        fluidRow(
          valueBoxOutput("sim_n_approved", width = 3),
          valueBoxOutput("sim_approval_rate", width = 3),
          valueBoxOutput("sim_expected_profit", width = 3),
          valueBoxOutput("sim_default_rate_approved", width = 3)
        ),

        fluidRow(
          box(
            title = "Profit curve", width = 12, status = "primary",
            p("Precomputed across the full threshold range — the dashed line marks your current selection. Total expected profit rises across the whole observed risk range in this dataset (see README for the full methodology and caveats behind this result)."),
            plotlyOutput("plot_profit_curve", height = 380)
          )
        )
      )
    )
  )
)


## ============================================================
## SERVER
## ============================================================

server <- function(input, output, session) {

  ## -------------------- OVERVIEW --------------------

  output$kpi_total_loans <- renderValueBox({
    valueBox(comma(nrow(dashboard_data)), "Loans in test set", icon = icon("file-invoice-dollar"), color = "blue")
  })

  output$kpi_default_rate <- renderValueBox({
    valueBox(paste0(round(mean(dashboard_data$default) * 100, 1), "%"),
             "Overall default rate", icon = icon("exclamation-triangle"), color = "orange")
  })

  output$kpi_avg_pd <- renderValueBox({
    valueBox(paste0(round(mean(dashboard_data$pd) * 100, 1), "%"),
             "Average predicted PD", icon = icon("chart-bar"), color = "purple")
  })

  output$kpi_total_el <- renderValueBox({
    valueBox(paste0("$", comma(round(sum(dashboard_data$expected_loss)))),
             "Total expected loss (if all approved)", icon = icon("piggy-bank"), color = "red")
  })

  output$plot_default_by_grade <- renderPlotly({
    p <- ggplot(default_by_grade, aes(x = grade, y = default_rate, text = paste0(grade, ": ", round(default_rate, 1), "%"))) +
      geom_col(fill = "#2C7FB8") +
      labs(x = "Grade", y = "Default rate (%)") +
      theme_minimal()
    ggplotly(p, tooltip = "text")
  })

  output$plot_pd_distribution <- renderPlotly({
    p <- ggplot(plot_sample, aes(x = pd)) +
      geom_histogram(bins = 30, fill = "#41AB5D") +
      labs(x = "Predicted PD", y = "Count (sample)") +
      theme_minimal()
    ggplotly(p)
  })

  ## -------------------- SEGMENTATION --------------------

  filtered_sample <- reactive({
    req(input$selected_clusters)
    plot_sample %>% filter(as.character(cluster) %in% input$selected_clusters)
  })

  output$plot_segments <- renderPlotly({
    p <- ggplot(filtered_sample(), aes(x = pd, y = expected_profit_if_approved, color = as.factor(cluster))) +
      geom_point(alpha = 0.4, size = 1) +
      scale_y_continuous(labels = comma) +
      labs(x = "Predicted PD", y = "Expected profit if approved ($)", color = "Cluster") +
      theme_minimal()
    ggplotly(p)
  })

  output$plot_segment_sizes <- renderPlotly({
    plot_data <- cluster_profile %>% filter(as.character(cluster) %in% input$selected_clusters)
    p <- ggplot(plot_data, aes(x = reorder(segment_label, pct_of_portfolio), y = pct_of_portfolio,
                                text = paste0(round(pct_of_portfolio, 1), "% of portfolio"))) +
      geom_col(fill = "#756BB1") +
      coord_flip() +
      labs(x = NULL, y = "% of portfolio") +
      theme_minimal()
    ggplotly(p, tooltip = "text")
  })

  output$table_cluster_profile <- renderDT({
    cluster_profile %>%
      mutate(
        avg_pd = paste0(round(avg_pd * 100, 1), "%"),
        actual_default_rate = paste0(round(actual_default_rate * 100, 1), "%"),
        avg_loan_amount = paste0("$", comma(round(avg_loan_amount))),
        avg_income = paste0("$", comma(round(avg_income))),
        avg_expected_profit = paste0("$", comma(round(avg_expected_profit))),
        pct_of_portfolio = paste0(round(pct_of_portfolio, 1), "%")
      ) %>%
      select(segment_label, pct_of_portfolio, avg_pd, actual_default_rate,
             avg_loan_amount, avg_income, avg_expected_profit) %>%
      rename(
        Segment = segment_label, `% Portfolio` = pct_of_portfolio,
        `Avg. PD` = avg_pd, `Actual Default Rate` = actual_default_rate,
        `Avg. Loan` = avg_loan_amount, `Avg. Income` = avg_income,
        `Avg. Expected Profit` = avg_expected_profit
      )
  }, options = list(dom = "t", pageLength = 10), rownames = FALSE)

  ## -------------------- THRESHOLD SIMULATOR --------------------

  approved <- reactive({
    dashboard_data %>% filter(pd <= input$pd_threshold)
  })

  output$sim_n_approved <- renderValueBox({
    valueBox(comma(nrow(approved())), "Loans approved", icon = icon("check-circle"), color = "green")
  })

  output$sim_approval_rate <- renderValueBox({
    rate <- nrow(approved()) / nrow(dashboard_data) * 100
    valueBox(paste0(round(rate, 1), "%"), "Approval rate", icon = icon("percent"), color = "blue")
  })

  output$sim_expected_profit <- renderValueBox({
    profit <- sum(approved()$expected_profit_if_approved)
    valueBox(paste0("$", comma(round(profit))), "Total expected profit", icon = icon("sack-dollar"), color = "purple")
  })

  output$sim_default_rate_approved <- renderValueBox({
    rate <- if (nrow(approved()) > 0) mean(approved()$default) * 100 else 0
    valueBox(paste0(round(rate, 1), "%"), "Default rate among approved", icon = icon("exclamation-triangle"), color = "orange")
  })

  output$plot_profit_curve <- renderPlotly({
    p <- ggplot(profit_curve, aes(x = threshold, y = total_expected_profit,
                                   text = paste0("Threshold: ", threshold, "\nProfit: $", comma(round(total_expected_profit))))) +
      geom_line(color = "#2C7FB8", linewidth = 1) +
      geom_vline(xintercept = input$pd_threshold, linetype = "dashed", color = "#E34A33") +
      scale_y_continuous(labels = comma) +
      labs(x = "Approval threshold (predicted PD)", y = "Total expected profit ($)") +
      theme_minimal()
    ggplotly(p, tooltip = "text")
  })
}


## ============================================================
## RUN APP
## ============================================================

shinyApp(ui = ui, server = server)
