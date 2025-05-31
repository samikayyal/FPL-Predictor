# FPL Predictor

## Description

This project aims to predict Fantasy Premier League (FPL) player points using historical data and machine learning. It involves gathering data from various sources, preprocessing it, performing exploratory data analysis, and building predictive models.

## Project Structure

The project is organized into the following main directories and files:

```
.
├── mydata/                     # Contains raw and processed FPL data, organized by season and gameweek
│   ├── 2024-25/
│   │   ├── fixtures.csv
│   │   ├── gws/                # Player data per gameweek
│   │   ├── team_gws/           # Team data per gameweek
│   │   └── ...                 # Season data / data not split into gameweeks
├── scrapers/                   # Python scripts for collecting data from different sources
│   ├── fbref/
│   ├── fpl_data_website/
│   └── official_fpl_website/
├── utils/                      # Utility scripts for constants, general functions, and ID retrieval
├── baseline_bps.py             # Script related to baseline BPS calculation
├── best_model.keras            # Saved best performing Keras model
├── eda.ipynb                   # Jupyter Notebook for Exploratory Data Analysis
├── final_df.csv                # Final processed DataFrame for modeling
├── gather_data.py              # Main script to orchestrate data gathering
├── model.ipynb                 # Jupyter Notebook for model development and training
├── preprocessing.py            # Script for data cleaning and feature engineering
├── requirements.txt            # Python dependencies for the project
└── README.md                   # This file
```

## Data Sources

The project utilizes data from:
*   Official FPL website
*   FBRef
*   FPL Data Website (third-party)

Scripts for scraping data from these sources are located in the `scrapers/` directory.

## Workflow

1.  **Data Gathering**: Scripts in `scrapers/` and `gather_data.py` are used to collect raw data.
2.  **Preprocessing**: `preprocessing.py` cleans the raw data, handles missing values, and engineers features.
3.  **Exploratory Data Analysis (EDA)**: `eda.ipynb` is used to understand data distributions, correlations, and identify patterns.
4.  **Modeling**: `model.ipynb` details the process of building, training, and evaluating predictive models. The best model is saved as `best_model.keras`.
5.  **Prediction**: The trained model can be used to predict player points for future gameweeks.

## Usage

1.  **Gather Data**:
    Run the main data gathering script:
    ```bash
    python gather_data.py
    ```
    This will execute the necessary scrapers and store the data in the `mydata/` directory.

2.  **Preprocess Data**:
    Execute the preprocessing script:
    ```bash
    python preprocessing.py
    ```
    This will generate `final_df.csv`.

3.  **Exploratory Data Analysis**:
    Open and run the `eda.ipynb` notebook in a Jupyter environment to explore the data.

4.  **Model Training**:
    Open and run the `model.ipynb` notebook to train models and evaluate their performance. The best model will be saved.