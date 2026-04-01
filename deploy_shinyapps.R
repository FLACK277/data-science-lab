required_pkgs <- c("rsconnect")
for (pkg in required_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}

library(rsconnect)

account <- Sys.getenv("SHINYAPPS_NAME", unset = "")
token <- Sys.getenv("SHINYAPPS_TOKEN", unset = "")
secret <- Sys.getenv("SHINYAPPS_SECRET", unset = "")

if (account == "" || token == "" || secret == "") {
  stop(
    "Missing shinyapps.io credentials. Set SHINYAPPS_NAME, SHINYAPPS_TOKEN, and SHINYAPPS_SECRET.",
    call. = FALSE
  )
}

rsconnect::setAccountInfo(name = account, token = token, secret = secret)

# Deploy the current project directory as a Shiny app.
rsconnect::deployApp(
  appDir = ".",
  appName = "heart-disease-ml-shiny",
  launch.browser = FALSE,
  forceUpdate = TRUE
)
