# Traffic Analysis and Environmental Impact Dashboard

## Overview
This project provides an in-depth analysis of traffic flow, road accidents, and their environmental impact in Irish cities for the year 2022. It integrates data from traffic monitoring systems, road collision reports, and environmental sensors to identify patterns, correlations, and trends. The interactive dashboard allows users to explore insights through visualizations and machine learning forecasts, offering actionable recommendations for urban planners and policymakers.

## Features

### 1. Traffic Flow Analysis
- Evaluates vehicle movement efficiency and congestion trends over time.
- Uses **time-series decomposition** to analyze daily, weekly, and seasonal traffic patterns.
- Implements **Facebook Prophet** for forecasting traffic congestion.

### 2. Collision Data Study
- Examines historical traffic accidents (2013–2022) across Irish regions.
- Visualizes accident trends with interactive plots (heatmaps, time-series charts).
- Predicts accident rates using **Random Forest Regressor** trained on historical data.

### 3. Environmental Impact Assessment
- Analyzes emissions of CO, NOx, and particulate matter (PM).
- Identifies correlations between traffic congestion and air quality degradation.
- Uses **Statsmodels** for statistical analysis and decomposition of emissions trends.

### 4. Forecasting Models
- **Facebook Prophet**: Forecasts traffic flow and congestion patterns.
- **Random Forest Regressor**: Predicts collision likelihood based on weather, time, and location.
- **Statsmodels**: Analyzes long-term emissions trends.
- Model performance evaluated using **R², MAE, and MSE**.

### 5. Interactive Visualizations
- Built with **Streamlit** for dynamic data exploration.
- Visualizes data using **Plotly, Matplotlib, and Seaborn**.
- Includes filters for time periods, regions, and metrics.

## Technologies Used
- **Python**: Data processing, analysis, and machine learning.
- **Streamlit**: Interactive web dashboard.
- **Pandas & NumPy**: Data manipulation.
- **Plotly, Matplotlib, Seaborn**: Visualization.
- **SQLAlchemy & PostgreSQL**: Database integration.
- **Scikit-learn**: Random Forest Regressor implementation.
- **Statsmodels**: Statistical analysis and decomposition.
- **GeoJSON & GIS**: Geospatial mapping.

## Data Sources
- **Traffic Flow Data**: Sourced from Irish transport authorities.
- **Road Collision Data**: Government reports and police records (2013–2022).
- **Environmental Data**: Air quality monitoring systems (CO, NOx, PM levels).
- **PostgreSQL Database**: Centralized storage for structured datasets.
