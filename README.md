# FPL Predictor

A machine learning tool to predict player performance in the Fantasy Premier League.

## Overview

The FPL Predictor uses historical player and team data from the Fantasy Premier League to train machine learning models that can predict player points for upcoming gameweeks. These predictions can help FPL managers make informed decisions about transfers, captaincy, and team selection.

## Features

- **Data Collection**: Automatically scrapes data from multiple sources:
  - Official FPL Website
  - FBRef
  - Third-party FPL data providers
  
- **Data Processing**: Cleans and transforms raw data into features suitable for machine learning
  
- **Machine Learning**: Trains models to predict player performance using TensorFlow
  
- **Team Optimization**: Suggests optimal team selections based on predictions and constraints

## Getting Started

### Prerequisites

- Python 3.8+
- Dependencies listed in `requirements.txt`


## Project Structure

```
.
├── mydata/                     # Raw and processed FPL data by season
├── scrapers/                   # Data collection scripts
│   ├── fbref/                  # FBRef statistics scraper
│   ├── fpl_data_website/       # Third-party FPL data scraper  
│   └── official_fpl_website/   # Official FPL API scraper
├── utils/                      # Utility functions
├── predictions/                # Generated player predictions
├── model_plots/                # Performance visualizations
├── logs/                       # Training logs
├── baseline_bps.py             # Bonus point system calculations
├── gather_data.py              # Main data collection script
├── preprocessing.py            # Data cleaning and feature engineering
├── model.py                    # Model training and evaluation
├── predict.py                  # Generate predictions and team suggestions
└── requirements.txt            # Project dependencies
```

## Model Performance

The model is evaluated using multiple metrics:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Huber Loss
- Log-Cosh Loss

Performance details are tracked in `model_results.md`.

## Workflow

1. **Data Gathering**: Scripts in `scrapers/` and `gather_data.py` collect data from various sources
2. **Preprocessing**: `preprocessing.py` cleans the data and engineers features
3. **Exploratory Data Analysis**: `eda.ipynb` visualizes patterns and relationships
4. **Model Training**: Neural network models are built and evaluated in `model.py`
5. **Prediction**: `predict.py` generates predictions and optimizes team selection

## Data Sources

The project utilizes data from:
- Official FPL website API
- FBRef for advanced football statistics
- FPL Data Website (third-party)