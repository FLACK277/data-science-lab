FROM rocker/r-ver:4.5.3

WORKDIR /app

# System dependencies needed by common R packages used in this project.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libcairo2-dev \
    libxt-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libjpeg-dev \
    libtiff5-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install required R packages for analysis + Shiny app.
RUN R -q -e "install.packages(c('shiny','shinydashboard','tidyverse','caret','randomForest','corrplot','scales','gridExtra','e1071','pROC','DT','plotly','rstudioapi'), repos='https://cloud.r-project.org')"

COPY . /app

EXPOSE 3838

CMD ["R", "-q", "-e", "shiny::runApp('/app', host='0.0.0.0', port=3838)"]
